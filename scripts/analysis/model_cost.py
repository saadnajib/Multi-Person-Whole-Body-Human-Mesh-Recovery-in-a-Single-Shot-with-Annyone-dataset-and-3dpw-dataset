"""
Parameters, FLOPs and (optionally) GPU latency of this repo's Model (model.py) for a DINOv2 ViT-S/B/L backbone:
the cost side of supervisor requirement 1b ("a significantly smaller model: fewer params AND FLOPs, faster
inference"), used by slurm/experiments/exp8_distill_vitb.sh.

    python scripts/analysis/model_cost.py --no-build --all             # analytic backbone table; runs anywhere,
                                                                        # no torch / anny / repo imports
    python scripts/analysis/model_cost.py --all                        # build Model on CPU for ViT-S/B/L (needs the
                                                                        # cluster env: torch, anny, the DINOv2 hub cache)
    python scripts/analysis/model_cost.py --backbone dinov2_vitb14 --head_dim 1024
    python scripts/analysis/model_cost.py --ckpt TEACHER.pt --ckpt STUDENT.pt --bench --json cost.json
                                                                        # architecture from each checkpoint's saved
                                                                        # args; --bench times it on the GPU

BACKBONE FLOPs (analytic; DINOv2 ViT as built by torch.hub 'facebookresearch/dinov2', dinov2_vit{s,b,l}14):
    Np  = (img / 14)^2 patch tokens (672 px: 48 x 48 = 2304); N = Np + 1 cls (+ R registers; R = 0 for the hub
          dinov2_vit*14 models this repo loads, blocks/dinov2.py)
    per block, d = width, h = MLP hidden (4d):
        MACs  = N * (4 d^2 + 2 d h)  [qkv 3Nd^2 + proj Nd^2 + fc1/fc2 2Ndh; = 12 N d^2 for h = 4d]
              + 2 N^2 d               [QK^T and attention x V, summed over heads]
        FLOPs = 2 * MACs = 24 N d^2 + 4 N^2 d     (one multiply-add = 2 FLOPs)
    + patch embedding (14x14 conv, stride 14): MACs = Np * 3 * 14^2 * d
    not counted: LayerNorm, GELU, softmax, bias adds, LayerScale, the bicubic resize of pos_embed (< 1% together).
    NOTE: 24 N d^2 + 4 N^2 d is the FLOP count of a block. Read as MACs (as the exp8 task text did) it double-counts;
    this script reports MACs and FLOPs separately so the two cannot be confused.
    Backbone parameters (analytic): patch conv 3*14^2*d + d, cls d, mask_token d, pos_embed 1370 d (the hub models
    are built at 518 px = 37 x 37 + 1 positions and interpolate), per block 12 d^2 + 15 d (h = 4d), final norm 2d.
    Results at 672 px: ViT-S 22.06 M / 196.9 GFLOPs, ViT-B 86.58 M / 589.5, ViT-L 304.37 M / 1917.2.

HEAD FLOPs (build mode): forward hooks on every nn.Linear outside the backbone during one forward of the built Model
(is_training=True with a fixed detection index, so the count does not depend on random detections), plus the
cross-attention matmuls of the HPH head analytically (2 * queries * context tokens * inner dim per layer). The head
cost is reported per image with --people persons in it (default 1): the dense parts (feat_proj, mlp_classif over all
Np tokens, the HPH context K/V projection, one per image) dominate, so it barely grows with the number of people.
The Anny body model (blend shapes + LBS, per person) has no nn.Linear and is NOT in these numbers.
In build mode the backbone linears are also counted with hooks on one real backbone forward and printed next to the
analytic linear part (12 N d^2 per block) as a check (--skip_backbone_forward skips it; ViT-L takes ~15 s on CPU).

--bench (CUDA only): batch 1 at img_size, fp32 with TF32 matmuls (train.py --tf32 1, evaluate() runs amp 0),
20 warm-up + 50 timed forwards (torch.cuda.synchronize around each), backbone alone and the full forward with one
fixed person (heads + Anny mesh). Reports mean / median ms per image and peak memory. Weights do not change the
timing (fixed person, no detection threshold involved); with --ckpt the checkpoint's weights are loaded anyway.

Checkpoints of the ORIGINAL repo class (multi_hmr_anny/multi_hmr.py, e.g. NAVER's multiHMR_672_L_anny.pt; keys
encoder.backbone.* / decoder.*) are recognised by their keys, as demo.py does, and built with that class
(xat_dim 1024, as demo.py): parameters per component and backbone FLOPs are reported; head FLOPs and --bench are
skipped for that class (printed as '-').

Exit status: 0 = done; 2 = build mode impossible here (missing torch / anny / repo import; message printed) or bad
arguments.
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# name -> (width d, depth L, heads); MLP ratio 4, patch 14, no registers (torch.hub dinov2_vit*14)
VIT = {'dinov2_vits14': (384, 12, 6), 'dinov2_vitb14': (768, 12, 12), 'dinov2_vitl14': (1024, 24, 16)}
SHORT = {'dinov2_vits14': 'ViT-S/14', 'dinov2_vitb14': 'ViT-B/14', 'dinov2_vitl14': 'ViT-L/14'}
PATCH = 14
HUB_POS_TOKENS = 37 * 37 + 1


# --------------------------------------------------------------------------- analytic (no imports)
def vit_block_macs(n_tokens, d, hidden):
    """(linear MACs, attention-matmul MACs) of one ViT block."""
    lin = n_tokens * (4 * d * d + 2 * d * hidden)
    att = 2 * n_tokens * n_tokens * d
    return lin, att


def analytic_backbone(name=None, img=672, d=None, depth=None, hidden=None, n_reg=0, patch=PATCH):
    if name is not None:
        d, depth, _ = VIT[name]
    hidden = hidden or 4 * d
    if img % patch:
        raise ValueError(f"img_size {img} is not a multiple of the patch size {patch}")
    n_patch = (img // patch) ** 2
    n_tok = n_patch + 1 + n_reg
    lin, att = vit_block_macs(n_tok, d, hidden)
    macs_patch = n_patch * 3 * patch * patch * d
    macs_blocks = depth * (lin + att)
    params = (3 * patch * patch * d + d) + d + d + HUB_POS_TOKENS * d + n_reg * d \
        + depth * (4 * d * d + 2 * d * hidden + 11 * d + hidden) + 2 * d
    # per block: qkv 3d^2+3d, proj d^2+d, fc1 d*h+h, fc2 h*d+d, 2 norms 4d, 2 LayerScales 2d -> 4d^2 + 2dh + 11d + h
    macs = macs_patch + macs_blocks
    return dict(name=name, img_size=img, width=d, depth=depth, mlp_hidden=hidden, registers=n_reg,
                patch_tokens=n_patch, tokens=n_tok, params=params,
                macs_linear=depth * lin, macs_attention=depth * att, macs_patch_embed=macs_patch,
                macs=macs, flops=2 * macs)


def fmt_m(x, nd=2):
    return '-' if x is None else f"{x / 1e6:.{nd}f}"


def fmt_g(x, nd=1):
    return '-' if x is None else f"{x / 1e9:.{nd}f}"


def print_analytic(names, img):
    rows = [analytic_backbone(n, img) for n in names]
    ref = next((r for r in rows if r['name'] == 'dinov2_vitl14'), None) or analytic_backbone('dinov2_vitl14', img)
    print(f"### Analytic backbone cost at {img}x{img} (tokens N = {rows[0]['patch_tokens']} patches + 1 cls; "
          f"FLOPs = 2 x MACs)")
    print()
    print("| backbone | width | depth | params (M) | linear GMACs | attention GMACs | patch-embed GMACs | GMACs | GFLOPs |"
          " params vs ViT-L | FLOPs vs ViT-L |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {SHORT[r['name']]} | {r['width']} | {r['depth']} | {fmt_m(r['params'])} | {fmt_g(r['macs_linear'])} | "
              f"{fmt_g(r['macs_attention'])} | {fmt_g(r['macs_patch_embed'], 2)} | {fmt_g(r['macs'])} | {fmt_g(r['flops'])} | "
              f"{r['params'] / ref['params']:.3f} ({ref['params'] / r['params']:.2f}x fewer) | "
              f"{r['flops'] / ref['flops']:.3f} ({ref['flops'] / r['flops']:.2f}x fewer) |")
    print()
    print("Backbone only. A student with --head_dim 1024 adds feat_proj (d x 1024 + 1024 params; "
          f"{rows[0]['patch_tokens']} x d x 1024 MACs) and keeps teacher-width heads (~14 M params, ~10-14 GFLOPs per "
          "image with one person, model_cost.py build mode). The Anny body model is not counted.")
    return rows


# --------------------------------------------------------------------------- build mode
def import_cluster_stack():
    """torch + the repo's model.py (anny, roma, einops, utils). Returns (torch, Model) or exits 2 with a message."""
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover - depends on the host
        print(f"[model_cost] build mode needs PyTorch ({type(e).__name__}: {e}).\n"
              f"  Here, use --no-build for the analytic backbone numbers; on the cluster: conda activate multihmr "
              f"and source slurm/env.sh (TORCH_HOME with the DINOv2 hub cache).")
        sys.exit(2)
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')   # utils.render imports pyrender (inspect_ckpt.py does the same)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        from model import Model
    except Exception as e:
        hint = ("the Anny body model package ('anny') is missing: it exists only in the cluster env multihmr"
                if 'anny' in str(e) else "a repo dependency failed to import")
        print(f"[model_cost] build mode cannot import model.py: {type(e).__name__}: {e}\n"
              f"  ({hint}). Use --no-build here for the analytic backbone numbers, or run on the cluster inside the "
              f"container with slurm/env.sh sourced.")
        sys.exit(2)
    return torch, Model


def sd_of(ckpt):
    for k in ('model_state_dict', 'state_dict'):
        if isinstance(ckpt, dict) and isinstance(ckpt.get(k), dict):
            return ckpt[k]
    return ckpt


def ckpt_label(path):
    """<run>_<epoch> for .../<run>[/train]/checkpoints/<epoch>.pt (exp6/exp7 tag convention), else the file stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    d = os.path.dirname(os.path.abspath(path))
    if os.path.basename(d) != 'checkpoints':
        return stem
    run = os.path.basename(os.path.dirname(d))
    if run == 'train':
        run = os.path.basename(os.path.dirname(os.path.dirname(d)))
    return f"{run}_{stem}"


def build_from_args(torch, Model, backbone, head_dim, img_size, ckpt_path=None):
    """(model, info). With ckpt_path: architecture from its saved args (diag_common.load_model / demo.py conventions),
    weights loaded strict=False; original-repo checkpoints are built with multi_hmr_anny's Multi_HMR."""
    info = dict(label=None, ckpt=ckpt_path, cls='Model')
    if ckpt_path is None:
        model = Model(backbone=backbone, pretrained_backbone=False, img_size=img_size, head_dim=head_dim,
                      person_center='head', bone_fix='none')
        info.update(label=f"{SHORT.get(backbone, backbone)} head_dim {head_dim or 0}", backbone=backbone,
                    head_dim=head_dim or 0, img_size=img_size)
        return model, info
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)   # saved args = argparse.Namespace
    a = ckpt.get('args') if isinstance(ckpt, dict) else None
    kw = dict(vars(a)) if hasattr(a, '__dict__') else dict(a or {})
    if not kw:
        raise SystemExit(f"[model_cost] {ckpt_path}: no saved 'args'; cannot rebuild the architecture")
    sd = sd_of(ckpt)
    n_model = sum(1 for k in sd if k.startswith(('backbone.encoder.', 'x_attention_head.', 'mlp_classif.')))
    n_orig = sum(1 for k in sd if k.startswith(('encoder.backbone.', 'decoder.', 'dec_to_token.', 'mlp_pose.')))
    kw.setdefault('simple_depth_encoding', 1)
    kw['pretrained_backbone'] = False
    kw['person_center'] = 'head'
    kw.pop('pretrained', None)
    if n_orig > n_model:
        from multi_hmr_anny.multi_hmr import Multi_HMR
        kw['xat_dim'] = 1024            # demo.py: the original checkpoints were trained with xat_dim 1024
        w = sd.get('decoder.transformer.layers.0.2.fn.net.0.weight')   # FFN width from the checkpoint (NAVER: 2048)
        kw['xat_mlp_dim'] = int(w.shape[0]) if hasattr(w, 'shape') and len(w.shape) == 2 else 2 * 1024
        model = Multi_HMR(**kw)
        info['cls'] = 'Multi_HMR (original repo)'
    else:
        kw.setdefault('pose_param', 'local-bone')
        kw['bone_fix'] = 'none'         # a bone fix is 3x3 products on a few bones: no cost
        kw['bone_fix_spec'] = None
        model = Model(**kw)
    msd = model.state_dict()
    matched = [k for k, v in sd.items() if k in msd and hasattr(v, 'shape') and msd[k].shape == v.shape]
    model.load_state_dict(sd, strict=False)
    info.update(label=ckpt_label(ckpt_path), backbone=kw.get('backbone'), head_dim=int(kw.get('head_dim') or 0),
                img_size=int(kw.get('img_size') if isinstance(kw.get('img_size'), int) else kw.get('img_size', [672])[0]),
                loaded=f"{len(matched)}/{len(msd)}")
    del ckpt, sd
    return model, info


def count_params(model):
    """Parameters by component (nn.Parameter only; buffers such as the Anny template are not parameters)."""
    groups = {}
    for name, p in model.named_parameters():
        top = name.split('.')[0]
        if top in ('backbone', 'encoder'):
            g = 'backbone'
            if top == 'encoder' and not name.startswith('encoder.backbone.'):
                g = 'heads'        # original class: encoder.mlp_det / mlp_fov_unique are head layers
        elif top == 'feat_proj':
            g = 'feat_proj'
        elif top == 'body_model':
            g = 'body_model'
        else:
            g = 'heads'            # mlp_classif, mlp_offset, x_attention_head, eye (Model); decoder, mlp_* (original)
        groups[g] = groups.get(g, 0) + p.numel()
    out = {k: groups.get(k, 0) for k in ('backbone', 'feat_proj', 'heads', 'body_model')}
    out['total'] = sum(groups.values())
    out['network'] = out['total'] - out['body_model']
    return out


def encoder_of(model):
    return model.backbone.encoder if hasattr(model, 'backbone') else model.encoder.backbone


def backbone_geometry(model):
    enc = encoder_of(model)
    if hasattr(enc.blocks[0], 'mlp'):
        blocks = list(enc.blocks)
    else:                                   # chunked blocks (BlockChunk, padded with nn.Identity): flatten
        blocks = [b for ch in enc.blocks for b in ch if hasattr(b, 'mlp')]
    hidden = blocks[0].mlp.fc1.out_features if hasattr(blocks[0].mlp, 'fc1') else 4 * enc.embed_dim
    return dict(d=enc.embed_dim, depth=len(blocks), hidden=hidden, n_reg=int(getattr(enc, 'num_register_tokens', 0) or 0),
                patch=int(enc.patch_size if isinstance(enc.patch_size, int) else enc.patch_size[0]))


class LinearCounter:
    """forward hooks on nn.Linear: MACs = rows x in x out per call, split by module-name prefix."""

    def __init__(self, torch, model):
        self.macs = {}
        self.handles = []
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                self.handles.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def h(mod, inp, out):
            x = inp[0]
            rows = x.numel() // x.shape[-1]
            self.macs[name] = self.macs.get(name, 0) + rows * mod.in_features * mod.out_features
        return h

    def total(self, prefix=None, exclude=None):
        return sum(v for k, v in self.macs.items()
                   if (prefix is None or k.startswith(prefix)) and (exclude is None or not k.startswith(exclude)))

    def remove(self):
        for h in self.handles:
            h.remove()


def fixed_person_inputs(torch, img, n_people, patch, device):
    g = img // patch
    x = torch.randn(1, 3, img, img, device=device)
    K = torch.tensor([[img, 0., img / 2], [0., img, img / 2], [0., 0., 1.]], device=device).unsqueeze(0)
    ys = torch.tensor([(g // 2 + i) % g for i in range(n_people)], device=device)
    xs = torch.tensor([(g // 2 + 3 * i) % g for i in range(n_people)], device=device)
    zero = torch.zeros(n_people, dtype=torch.long, device=device)
    idx = (zero, ys, xs, zero.clone())          # Model.detection reads scores[idx[0], idx[3], idx[1], idx[2]]
    return x, K, idx


def head_flops(torch, model, info, n_people, device):
    """MACs of everything outside the backbone for one image with n_people fixed persons (Model class only)."""
    if info['cls'] != 'Model':
        return None
    geo = backbone_geometry(model)
    img = model.img_size
    n_patch = (img // geo['patch']) ** 2
    x, K, idx = fixed_person_inputs(torch, img, n_people, geo['patch'], device)
    real_fwd = model.backbone.forward
    model.backbone.forward = lambda inp: torch.randn(inp.shape[0], n_patch, model.backbone_dim, device=inp.device)
    cnt = LinearCounter(torch, model)
    try:
        with torch.no_grad():
            model(x, idx=idx, K=K, is_training=True)
    finally:
        model.backbone.forward = real_fwd
        cnt.remove()
    lin = cnt.total(exclude='backbone.')
    # attention matmuls of the HPH head (not nn.Linear): per layer, self-attention over the n_people queries of the
    # image (2 n^2 inner) and cross-attention of those queries over the n_patch context tokens (2 n Np inner)
    att = 0
    for name, m in model.x_attention_head.named_modules():
        cls = type(m).__name__
        if cls == 'CrossAttention':
            att += 2 * n_people * n_patch * m.to_q.out_features
        elif cls == 'Attention':
            att += 2 * n_people * n_people * (m.to_qkv.out_features // 3)
    parts = {k: v for k, v in sorted(cnt.macs.items()) if not k.startswith('backbone.')}
    by_top = {}
    for k, v in parts.items():
        top = k.split('.')[0]
        by_top[top] = by_top.get(top, 0) + v
    return dict(macs_linear=lin, macs_attention=att, macs=lin + att, flops=2 * (lin + att), by_module=by_top,
                people=n_people)


def backbone_hook_check(torch, model, device):
    """Linear MACs of one real backbone forward, counted with hooks (should equal the analytic linear part)."""
    geo = backbone_geometry(model)
    img = model.img_size
    cnt = LinearCounter(torch, model)
    try:
        with torch.no_grad():
            model.backbone(torch.randn(1, 3, img, img, device=device)) if hasattr(model, 'backbone') \
                else model.encoder(torch.randn(1, 3, img, img, device=device))
    finally:
        cnt.remove()
    pre = 'backbone.encoder.' if hasattr(model, 'backbone') else 'encoder.backbone.'
    return cnt.total(prefix=pre), geo


def bench(torch, model, info, warm, iters, device):
    if info['cls'] != 'Model':
        return None
    geo = backbone_geometry(model)
    img = model.img_size
    x, K, idx = fixed_person_inputs(torch, img, 1, geo['patch'], device)

    def run(fn):
        times = []
        with torch.no_grad():
            for i in range(warm + iters):
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                fn()
                e.record()
                torch.cuda.synchronize()
                if i >= warm:
                    times.append(s.elapsed_time(e))
        times.sort()
        return dict(mean_ms=sum(times) / len(times), median_ms=times[len(times) // 2], n=len(times))

    def bb():
        z = model.backbone(x)
        if getattr(model, 'feat_proj', None) is not None:
            model.feat_proj(z)

    torch.cuda.reset_peak_memory_stats()
    out = dict(backbone=run(bb), full_1person=run(lambda: model(x, idx=idx, K=K, is_training=True)),
               warmup=warm, timed=iters, batch=1, precision='fp32 + TF32 matmuls',
               gpu=torch.cuda.get_device_name(0), peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    return out


def analyse(torch, Model, a, backbone=None, head_dim=0, ckpt=None):
    device = 'cuda' if a.bench else 'cpu'
    model, info = build_from_args(torch, Model, backbone, head_dim, a.img_size, ckpt)
    model = model.to(device).eval()
    if device == 'cpu':
        # the hub DINOv2 attention calls xFormers whenever it imported; its module flag also covers hub copies that
        # predate the XFORMERS_DISABLED switch
        for mname, mod in list(sys.modules.items()):
            if mname.endswith('layers.attention') and hasattr(mod, 'XFORMERS_AVAILABLE'):
                mod.XFORMERS_AVAILABLE = False
    params = count_params(model)
    geo = backbone_geometry(model)
    ana = analytic_backbone(img=model.img_size, d=geo['d'], depth=geo['depth'], hidden=geo['hidden'],
                            n_reg=geo['n_reg'], patch=geo['patch'])
    res = dict(info=info, params=params, backbone_analytic=ana, geometry=geo)
    if not a.skip_backbone_forward:
        hooked, _ = backbone_hook_check(torch, model, device)
        res['backbone_linear_macs_hooked'] = hooked
        res['backbone_linear_check'] = ('OK' if hooked == ana['macs_linear'] else
                                        f"MISMATCH (hooked {hooked} vs analytic {ana['macs_linear']})")
    res['heads'] = head_flops(torch, model, info, a.people, device)
    # original-repo class: head FLOPs not counted, so no total (a backbone-only number would read as the whole model)
    res['total_flops'] = ana['flops'] + res['heads']['flops'] if res['heads'] else None
    if a.bench:
        res['bench'] = bench(torch, model, info, a.warmup, a.iters, device)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return res


def print_built(results, ref_label=None):
    ref = None
    if ref_label:
        ref = next((r for r in results if r['info']['label'] == ref_label), None)
    if ref is None:
        ref = max(results, key=lambda r: r['params']['network'])
    people = next((r['heads']['people'] for r in results if r.get('heads')), '-')
    print(f"### Model cost (built from model.py; FLOPs at the model's img_size, heads with {people} person(s); "
          f"reference = {ref['info']['label']}; '-' = not counted)")
    print()
    print("| model | class | backbone | head_dim | backbone params (M) | feat_proj (M) | heads (M) | network params (M) |"
          " backbone GFLOPs | heads GFLOPs | total GFLOPs | params vs ref | FLOPs vs ref | backbone ms | full ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        i, p, h, b = r['info'], r['params'], r.get('heads'), r.get('bench')
        print(f"| {i['label']} | {i['cls']} | {SHORT.get(i.get('backbone'), i.get('backbone'))} | {i.get('head_dim', 0)} | "
              f"{fmt_m(p['backbone'])} | {fmt_m(p['feat_proj'])} | {fmt_m(p['heads'])} | {fmt_m(p['network'])} | "
              f"{fmt_g(r['backbone_analytic']['flops'])} | {fmt_g(h['flops'], 2) if h else '-'} | "
              f"{fmt_g(r['total_flops'])} | {p['network'] / ref['params']['network']:.3f} | "
              f"{(r['total_flops'] / ref['total_flops']) if r['total_flops'] and ref['total_flops'] else float('nan'):.3f} | "
              f"{('%.1f' % b['backbone']['median_ms']) if b else '-'} | {('%.1f' % b['full_1person']['median_ms']) if b else '-'} |")
    print()
    for r in results:
        i = r['info']
        line = f"  {i['label']}: body_model params {fmt_m(r['params']['body_model'])} M (not in 'network')"
        if 'backbone_linear_check' in r:
            line += f"; backbone linear MACs hooked vs analytic: {r['backbone_linear_check']}"
        if i.get('loaded'):
            line += f"; weights loaded {i['loaded']}"
        if r.get('heads'):
            line += "; head GMACs by module: " + ", ".join(f"{k} {v / 1e9:.2f}" for k, v in r['heads']['by_module'].items())
        print(line)
        if r.get('bench'):
            b = r['bench']
            print(f"  {i['label']}: bench on {b['gpu']}, {b['precision']}, batch 1, {b['warmup']} warm + {b['timed']} timed: "
                  f"backbone {b['backbone']['mean_ms']:.1f} ms mean / {b['backbone']['median_ms']:.1f} median, full (1 person) "
                  f"{b['full_1person']['mean_ms']:.1f} / {b['full_1person']['median_ms']:.1f} ms; peak {b['peak_mem_gb']:.2f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-build', dest='no_build', action='store_true',
                    help='analytic backbone numbers only (no torch, no repo imports; runs anywhere)')
    ap.add_argument('--all', action='store_true', help='dinov2_vits14, dinov2_vitb14 and dinov2_vitl14')
    ap.add_argument('--backbone', default='dinov2_vitb14', choices=sorted(VIT))
    ap.add_argument('--head_dim', type=int, default=1024,
                    help='build mode: Model head_dim (1024 = the projector-fed students of exp8 / v6-v8; for ViT-L it '
                         'equals the backbone width, so no feat_proj). 0 = heads at the backbone width')
    ap.add_argument('--img_size', type=int, default=672)
    ap.add_argument('--ckpt', action='append', default=[],
                    help='build from this checkpoint\'s saved args (repeatable); overrides --backbone/--all')
    ap.add_argument('--people', type=int, default=1, help='persons per image for the head FLOPs')
    ap.add_argument('--skip_backbone_forward', action='store_true',
                    help='build mode: skip the hooked backbone forward that checks the analytic linear MACs')
    ap.add_argument('--bench', action='store_true', help='time a forward on the GPU (batch 1)')
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--ref', default='', help='label of the reference row for the ratios (default: the largest model)')
    ap.add_argument('--json', default='', help='write every number to this JSON file')
    a = ap.parse_args()

    names = sorted(VIT, key=lambda n: VIT[n][0] * VIT[n][1]) if a.all else [a.backbone]
    if a.no_build:
        if a.ckpt or a.bench:
            print("[model_cost] --no-build ignores --ckpt / --bench (analytic backbone numbers only)")
        rows = print_analytic(names, a.img_size)
        if a.json:
            with open(a.json, 'w') as f:
                json.dump(dict(mode='analytic', rows=rows), f, indent=2)
            print(f"wrote {a.json}")
        return 0

    torch, Model = import_cluster_stack()
    if a.bench and not torch.cuda.is_available():
        print("[model_cost] --bench needs a CUDA GPU (none visible); rerun without --bench or inside a GPU job")
        return 2
    if not a.bench:
        # CPU forward: the hub DINOv2 uses xFormers' memory-efficient attention when xFormers imports, which has no
        # CPU kernel in the env's xformers 0.0.20 [Likely]; DINOv2 reads XFORMERS_DISABLED at import time.
        os.environ.setdefault('XFORMERS_DISABLED', '1')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if 'TORCH_HOME' not in os.environ:
        print("[model_cost] note: TORCH_HOME is not set; torch.hub looks for the DINOv2 code in ~/.cache/torch "
              "(source slurm/env.sh on the cluster)")
    results = []
    if a.ckpt:
        for c in a.ckpt:
            print(f"[model_cost] building from {c}", flush=True)
            results.append(analyse(torch, Model, a, ckpt=c))
    else:
        for n in names:
            print(f"[model_cost] building {n} head_dim {a.head_dim}", flush=True)
            results.append(analyse(torch, Model, a, backbone=n, head_dim=a.head_dim))
    print()
    print_built(results, a.ref or None)
    if a.json:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, float) and not math.isfinite(o):
                return None
            return o
        with open(a.json, 'w') as f:
            json.dump(dict(mode='built', results=clean(results)), f, indent=2)
        print(f"wrote {a.json}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
