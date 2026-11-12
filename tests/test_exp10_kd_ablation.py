"""CPU-only checks of the exp10 pieces (supervisor feedback of 12 Nov):
  - exp8's KD_MODE (full | nokd | supervised): the default mode composes exactly the commands of the commit before
    KD_MODE existed; each ablation mode's flags, RUN name and refusals (bash, LOCAL_CHECK=1);
  - exp8's job mode and POST_ONLY on a FAKE CLUSTER (a temp repo with stub srun / sbatch / squeue / python: the real
    exp8 script runs end to end, only the GPU programs are replaced by small fakes that write what the real ones write);
  - the exp10 launcher (LOCAL_CHECK and, on the fake cluster, a real submission with the record file);
  - dinov2_hub.load_dinov2 (local hub cache first, retries on HTTP 504) with a fake torch.hub;
  - train.py's kd_terms_active and the start-up lines exp8 greps for (read with ast; train.py is never imported);
  - the collectors: sweep_validation_table.py on logs in train.py's formats, kd_ablation_table.py on fake run folders,
    rh_recall_by_crowd.py on the Relative Human CSVs in docs/thesis/data;
  - eval_relative_human.sh's TAG for the exp8 layout and bone_fix.from_arg with a JSON path (the student's own fix).
"""

import ast
import csv
import io
import json
import os
import re
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dinov2_hub                                               # noqa: E402
from scripts.analysis import kd_ablation_table as abl           # noqa: E402
from scripts.analysis import rh_recall_by_crowd as crowd        # noqa: E402
from scripts.analysis import sweep_validation_table as valtab   # noqa: E402

EXP8 = ROOT / 'slurm' / 'experiments' / 'exp8_distill_vitb.sh'
EXP10 = ROOT / 'slurm' / 'experiments' / 'exp10_kd_ablation.sh'
EVAL_RH = ROOT / 'slurm' / 'diagnostics' / 'eval_relative_human.sh'
TAG = 'v5_pc_dist5_bonefix_v23_e10_00009'
PREFIX = f'student_vitb_from_{TAG}_e30'
BEFORE_KD_MODE = 'fd7f182'      # the last commit whose exp8 has no KD_MODE: the reference for "default unchanged"
CLUSTER = '/netscratch/najib/multi-hmr'
# the reported recipe's train.py command (exp8 LOCAL_CHECK at BEFORE_KD_MODE, default settings), one flag per item
GOLDEN_DEFAULT_TRAIN_CMD = (
    'python train.py --train_data AnnyOne --train_n 0 --val_anny_n 100 --person_center head --backbone dinov2_vitb14 '
    f'--img_size 672 --head_dim 1024 --pretrained {CLUSTER}/models/multiHMR/multiHMR_672_B.pt --pretrained_remap 1 '
    '--load_only_backbone 1 --min_backbone_match 90 --freeze_backbone 0 --distill_teacher_ckpt '
    f'{CLUSTER}/results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt '
    '--init_heads_from_teacher 1 --freeze_heads_epochs 5 --teacher_bf16 1 --lambda_kd 1.0 --lambda_kd_out 1.0 '
    '--kd_temperature 4.0 --kd_softmax_dim channel --bone_fix '
    f'{CLUSTER}/results/experiments/exp8_distill_vitb/{PREFIX}/teacher_bone_fix.json --mask_helper_joints_train 1 '
    '--use_anny_shape 1 --alpha_shape 20 --alpha_j2d 0.1 --alpha_v2d 0.1 --boost_neck_weight 4 --boost_hand_weight 4 '
    '--rotmat_loss l1 --alpha_rotmat 0.1 --alpha_global_rot 0 --start_2d_epoch 0 --gt_center_joint person_center '
    '--alpha_dist 5 --alpha_transl 5 --debug_anchor 2 --exclude_indices_file '
    f'{CLUSTER}/results/experiments/exp8_distill_vitb/{PREFIX}/exclude_manifest_indices.txt --mix_3dpw_labels '
    f'{CLUSTER}/threedpw_anny_v23/labels --mix_3dpw_ratio 0.3 --mix_3dpw_split train --val_3dpw_split validation '
    '--val_3dpw_n 300 --mask_3dpw_hands 1 --brightness 0.2 --contrast 0.2 --saturation 0.2 --hue 0.05 --batch_size 8 '
    '--num_workers 12 --n_iters_per_epoch 1000 --max_iter 30000 --learning_rate 3e-5 --lr_decay_every 12 '
    '--lr_decay_gamma 0.5 --amp 0 --tf32 1 --eval_freq 5 --val_loss 0 --nb_max_ckpt 3 --log_freq 50 --save_dir '
    f'{CLUSTER}/results/experiments/exp8_distill_vitb/{PREFIX} --name train')
EXP8_ENV = ('RUN', 'KD_T', 'KD_MODE', 'TEMPS', 'ARMS', 'EPOCHS', 'TIME', 'LAMBDA_KD', 'LAMBDA_KD_OUT', 'INIT_HEADS',
            'STUDENT_TRAIN_FIX', 'FREEZE_HEADS_EPOCHS', 'POST_ONLY', 'TEACHER_REMEASURE', 'TEACHER_REUSE', 'ABL_TAG',
            'SLURM_JOB_ID', 'EXP8_JOB', 'OVERWRITE', 'SMOKE', 'REPO', 'TEACHER', 'LABELS', 'EVAL_FREQ', 'VAL_LOSS',
            'TEACHER_TRACE_DIR', 'TEACHER_EVAL_LOG', 'CHECK_ONLY', 'DRY_RUN', 'LOCAL_CHECK', 'MODEL', 'TAG', 'BONE_FIX')


def run_bash(script, env, cwd=ROOT):
    full = {k: v for k, v in os.environ.items() if k not in EXP8_ENV}
    full.update(env)
    return subprocess.run(['bash', str(script)], cwd=str(cwd), env=full, capture_output=True, text=True)


def train_cmd(stdout):
    """The composed train.py command of an exp8 LOCAL_CHECK output, on one line."""
    a = stdout.index('--- train.py (job step 1')
    a = stdout.index('\n', a) + 1
    b = stdout.index('--- post-steps')
    return ' '.join(stdout[a:b].replace('\\\n', ' ').split())


def old_exp8(tmp):
    """exp8 as of the commit before KD_MODE, or None when git cannot give it (shallow clone, no git)."""
    try:
        r = subprocess.run(['git', 'show', f'{BEFORE_KD_MODE}:slurm/experiments/exp8_distill_vitb.sh'], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or 'KD_MODE' in r.stdout or 'exp8_distill_vitb' not in r.stdout:
        return None
    p = Path(tmp) / 'exp8_before_kd_mode.sh'
    p.write_text(r.stdout)
    return p


# =============================================================================================== exp8 modes (LOCAL_CHECK)
class Exp8DefaultModeUnchangedTests(unittest.TestCase):
    def test_default_train_command_is_the_reported_recipe(self):
        r = run_bash(EXP8, {'LOCAL_CHECK': '1'})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.splitlines()[0], f'=== exp8_distill_vitb {PREFIX}')
        self.assertEqual(train_cmd(r.stdout), GOLDEN_DEFAULT_TRAIN_CMD)
        self.assertNotIn('KD_MODE', r.stdout)
        self.assertNotIn('POST_ONLY', r.stdout)
        r2 = run_bash(EXP8, {'LOCAL_CHECK': '1', 'KD_MODE': 'full'})
        self.assertEqual(r2.stdout, r.stdout)

    def test_whole_local_check_output_equals_the_commit_before_kd_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = old_exp8(tmp)
            if old is None:
                self.skipTest(f'git cannot show {BEFORE_KD_MODE}:slurm/experiments/exp8_distill_vitb.sh here')
            cases = [{}, {'KD_T': '16', 'RUN': f'{PREFIX}_kdT16_sweep', 'VAL_LOSS': '1', 'EVAL_FREQ': '1', 'TIME': '16:00:00'},
                     {'KD_T': '8'}, {'SMOKE': '1'}, {'EPOCHS': '60'}, {'STUDENT_TRAIN_FIX': 'none'},
                     {'MASK_3DPW_SHAPE': '1'}, {'STUDENT_BACKBONE': 'dinov2_vits14'}, {'STUDENT_INIT': 'dinov2'},
                     {'TEACHER_REMEASURE': '0'}, {'LAMBDA_KD': '0.5', 'LAMBDA_KD_OUT': '2', 'BATCH': '4', 'N_ITERS': '500'}]
            for env in cases:
                env = dict(env, LOCAL_CHECK='1')
                new, ref = run_bash(EXP8, env), run_bash(old, env)
                self.assertEqual(ref.returncode, 0, env)
                self.assertEqual(new.returncode, 0, (env, new.stdout + new.stderr))
                self.assertEqual(new.stdout, ref.stdout, env)


class Exp8AblationModeTests(unittest.TestCase):
    def local(self, **env):
        env.setdefault('LOCAL_CHECK', '1')
        r = run_bash(EXP8, env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout.splitlines()[0].split()[-1], r.stdout

    def test_nokd_only_the_two_weights_and_the_run_folder_differ(self):
        run, out = self.local(KD_MODE='nokd')
        self.assertEqual(run, PREFIX + '_nokd')
        cmd = train_cmd(out)
        expect = (GOLDEN_DEFAULT_TRAIN_CMD.replace('--lambda_kd 1.0 --lambda_kd_out 1.0', '--lambda_kd 0 --lambda_kd_out 0')
                  .replace(f'/{PREFIX}/', f'/{PREFIX}_nokd/').replace(f'/{PREFIX} --name', f'/{PREFIX}_nokd --name'))
        self.assertEqual(cmd, expect)
        for kept in ('--distill_teacher_ckpt ', '--init_heads_from_teacher 1 --freeze_heads_epochs 5',
                     f'--bone_fix {CLUSTER}/results/experiments/exp8_distill_vitb/{PREFIX}_nokd/teacher_bone_fix.json'):
            self.assertIn(kept, cmd)
        self.assertIn('KD_MODE=nokd (ABLATION)', out)
        self.assertIn(f'--job-name=exp8-{PREFIX}_nokd ', out)
        self.assertIn('holdout_student_trainfix', out)      # the training fix (the teacher's) is still compared

    def test_supervised_has_no_teacher_flag_no_fix_and_the_same_architecture(self):
        run, out = self.local(KD_MODE='supervised')
        self.assertEqual(run, PREFIX + '_supervised')
        cmd = train_cmd(out)
        removed = ('--distill_teacher_ckpt '
                   f'{CLUSTER}/results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt '
                   '--init_heads_from_teacher 1 --freeze_heads_epochs 5 --teacher_bf16 1 --lambda_kd 1.0 --lambda_kd_out 1.0 '
                   '--kd_temperature 4.0 --kd_softmax_dim channel ')
        self.assertIn(removed, GOLDEN_DEFAULT_TRAIN_CMD)
        expect = (GOLDEN_DEFAULT_TRAIN_CMD.replace(removed, '')
                  .replace(f'--bone_fix {CLUSTER}/results/experiments/exp8_distill_vitb/{PREFIX}/teacher_bone_fix.json', '--bone_fix none')
                  .replace(f'/{PREFIX}/', f'/{PREFIX}_supervised/').replace(f'/{PREFIX} --name', f'/{PREFIX}_supervised --name'))
        self.assertEqual(cmd, expect)
        for word in ('--distill', '--init_heads', '--teacher_bf16', '--lambda_kd', '--kd_', '--freeze_heads', 'teacher_bone_fix', '_nofix'):
            self.assertNotIn(word, cmd)
        for same in ('--backbone dinov2_vitb14', '--head_dim 1024', '--load_only_backbone 1', '--start_2d_epoch 0',
                     '--max_iter 30000', '--learning_rate 3e-5', '--mix_3dpw_ratio 0.3'):
            self.assertIn(same, cmd)
        self.assertIn('KD_MODE=supervised (ABLATION)', out)
        self.assertIn('heads from teacher 0 (frozen 0 epochs), training bone fix: none', out)
        self.assertNotIn('holdout_student_trainfix', out)   # no training fix to compare with
        self.assertIn('model_cost.py --ckpt', out)          # the teacher is still the cost / table reference

    def test_suffix_order_with_smoke_and_temperature(self):
        self.assertEqual(self.local(KD_MODE='nokd', SMOKE='1')[0], f'student_vitb_from_{TAG}_e1_it20_nokd_smoke')
        self.assertEqual(self.local(KD_MODE='supervised', EPOCHS='60')[0], f'student_vitb_from_{TAG}_e60_supervised')
        self.assertEqual(self.local(KD_MODE='nokd', RUN=f'{PREFIX}_nokd_b')[0], f'{PREFIX}_nokd_b')

    def test_refusals(self):
        bad = [({'KD_MODE': 'none'}, 'KD_MODE=none'),
               ({'KD_MODE': 'nokd', 'LAMBDA_KD': '1.0'}, 'means no distillation loss'),
               ({'KD_MODE': 'supervised', 'LAMBDA_KD_OUT': '0.5'}, 'means no distillation loss'),
               ({'KD_MODE': 'supervised', 'INIT_HEADS': '1'}, 'INIT_HEADS=1 must be 0'),
               ({'KD_MODE': 'supervised', 'STUDENT_TRAIN_FIX': 'teacher'}, 'STUDENT_TRAIN_FIX=teacher must be none'),
               ({'KD_MODE': 'supervised', 'FREEZE_HEADS_EPOCHS': '5'}, 'FREEZE_HEADS_EPOCHS=5 must be 0'),
               ({'LAMBDA_KD': '0', 'LAMBDA_KD_OUT': '0.0'}, 'use KD_MODE=nokd'),
               ({'KD_MODE': 'nokd', 'RUN': PREFIX}, 'does not contain _nokd'),
               ({'KD_MODE': 'supervised', 'RUN': f'{PREFIX}_nokd'}, 'does not contain _supervised'),
               ({'RUN': f'{PREFIX}_nokd'}, 'names an ablation arm but KD_MODE is full'),
               ({'RUN': f'{PREFIX}_supervised'}, 'names an ablation arm but KD_MODE is full'),
               ({'POST_ONLY': '1', 'OVERWRITE': '1'}, 'POST_ONLY keeps the run folder'),
               ({'POST_ONLY': '2'}, 'POST_ONLY=2')]
        for env, text in bad:
            r = run_bash(EXP8, dict(env, LOCAL_CHECK='1'))
            self.assertNotEqual(r.returncode, 0, env)
            self.assertIn('ERROR', r.stdout, env)
            self.assertIn(text, r.stdout, env)
        # explicit zeros in an ablation mode and a single zero weight in full mode are fine
        self.assertEqual(run_bash(EXP8, {'LOCAL_CHECK': '1', 'KD_MODE': 'nokd', 'LAMBDA_KD': '0.0', 'LAMBDA_KD_OUT': '0'}).returncode, 0)
        self.assertEqual(run_bash(EXP8, {'LOCAL_CHECK': '1', 'LAMBDA_KD': '0'}).returncode, 0)

    def test_post_only_local_check_names_the_last_checkpoint_and_a_short_time_limit(self):
        _, out = self.local(POST_ONLY='1', KD_T='16', RUN=f'{PREFIX}_kdT16_sweep', VAL_LOSS='1', EVAL_FREQ='1')
        self.assertIn(f'POST_ONLY=1: no training; post-steps 2-6 on results/experiments/exp8_distill_vitb/{PREFIX}_kdT16_sweep/train/checkpoints/00029.pt', out)
        self.assertIn('--time=02:00:00 ', out)            # 60 min + two full 3DPW-test evaluations, no epochs
        self.assertIn('NOT run with POST_ONLY=1', out)
        self.assertIn('--kd_temperature 16 ', train_cmd(out) + ' ')
        _, out = self.local(POST_ONLY='1', EPOCHS='60')
        self.assertIn('train/checkpoints/00059.pt', out)


# =============================================================================================== exp10 launcher (local)
class Exp10LauncherLocalTests(unittest.TestCase):
    def test_local_check_runs_both_arms_through_exp8(self):
        r = run_bash(EXP10, {'LOCAL_CHECK': '1'})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = r.stdout
        flat = out.replace('\\\n', ' ')
        for arm in ('nokd', 'supervised'):
            self.assertIn(f'{arm}  ->  results/experiments/exp8_distill_vitb/{PREFIX}_{arm}', out)
            self.assertIn(f'=== exp8_distill_vitb {PREFIX}_{arm}\n', out)
            self.assertIn(f'KD_MODE={arm} (ABLATION)', out)
        self.assertNotIn(f'=== exp8_distill_vitb {PREFIX}\n', out)          # the reported folder is never a target
        self.assertEqual(flat.count('--val_loss 1 '), 2)
        self.assertEqual(flat.count('--eval_freq 1 '), 2)
        self.assertEqual(flat.count('--time=16:00:00'), 2)
        self.assertEqual(flat.count('--lambda_kd 0 '), 1)                   # nokd; supervised has no KD flag at all
        self.assertEqual(flat.count('--distill_teacher_ckpt '), 1)
        self.assertEqual(len(re.findall(r'--bone_fix none\s+--mask_helper_joints_train', flat)), 1)   # the supervised train command
        self.assertEqual(out.count('teacher re-measured 0;'), 2)            # the reported run's teacher numbers are reused
        self.assertNotIn('holdout_teacher --select', out)
        self.assertIn(f'teacher numbers: reused (TEACHER_REMEASURE=0) from {CLUSTER}/results/experiments/exp8_distill_vitb/{PREFIX}/holdout_teacher', out)
        self.assertIn(f'kd_ablation_table.py --reported results/experiments/exp8_distill_vitb/{PREFIX} --sweep_t4 '
                      f'results/experiments/exp8_distill_vitb/{PREFIX}_kdT4_sweep --nokd results/experiments/exp8_distill_vitb/{PREFIX}_nokd '
                      f'--supervised results/experiments/exp8_distill_vitb/{PREFIX}_supervised --out results/experiments/exp10_kd_ablation', out)
        self.assertIn('LOCAL_CHECK=1: nothing checked against the cluster, nothing submitted', out)

    def test_one_arm_a_tag_and_teacher_remeasure(self):
        r = run_bash(EXP10, {'LOCAL_CHECK': '1', 'ARMS': 'supervised', 'ABL_TAG': 'r2', 'TEACHER_REUSE': '0'})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(f'=== exp8_distill_vitb {PREFIX}_supervised_r2\n', r.stdout)
        self.assertNotIn('_nokd_r2\n', r.stdout.split('collect (login node')[0])
        self.assertIn('teacher re-measured 1;', r.stdout)
        self.assertIn('holdout_teacher --select holdout', r.stdout)

    def test_post_only_passes_through_without_the_training_time_limit(self):
        r = run_bash(EXP10, {'LOCAL_CHECK': '1', 'POST_ONLY': '1', 'ARMS': 'nokd'})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('POST_ONLY=1: no training; post-steps 2-6', r.stdout)
        self.assertIn('--time=01:30:00 ', r.stdout)       # 60 min + one evaluation (teacher reused), exp8's own
        self.assertNotIn('--time=16:00:00', r.stdout)

    def test_refusals(self):
        for bad in ({'RUN': 'x'}, {'KD_MODE': 'nokd'}, {'KD_T': '8'}, {'LAMBDA_KD': '0'}, {'INIT_HEADS': '0'},
                    {'STUDENT_TRAIN_FIX': 'none'}, {'FREEZE_HEADS_EPOCHS': '0'}, {'TEACHER_REMEASURE': '1'},
                    {'ARMS': 'full'}, {'ARMS': 'nokd nokd'}, {'ARMS': ' '}, {'TEACHER_REUSE': 'yes'}, {'ABL_TAG': 'a b'}):
            r = run_bash(EXP10, dict(bad, LOCAL_CHECK='1'))
            self.assertNotEqual(r.returncode, 0, bad)
            self.assertIn('ERROR', r.stdout, bad)


# =============================================================================================== the fake cluster
FAKE_PYTHON = r'''#!{real}
"""Stand-in for `python` on the fake cluster: the GPU programs exp8 starts are replaced by fakes that write what the
real ones write; everything else (python -c ..., the embedded stdin scripts) runs in the real interpreter."""
import json, os, sys
REAL = {real!r}
a = sys.argv[1:]


def opt(name, default=None):
    return a[a.index(name) + 1] if name in a else default


def real():
    os.execv(REAL, [REAL] + a)


FIX = {{'bones': {{'neck02': [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]}}}}
if not a:
    real()
if a[0] == '-c':
    if 'torch' in a[1]:
        sys.exit(0)            # the env / GPU capability probe and find_check_python
    real()
if a[0] == '-':
    if '--teacher' in a:       # TEACHER_CHECK_PY
        sys.stdin.read()
        if opt('--write_fix'):
            json.dump(FIX, open(opt('--write_fix'), 'w'))
        if opt('--write_json'):
            json.dump(dict(teacher=opt('--teacher'), backbone='dinov2_vitl14', head_dim=0, head_width=1024,
                           bones=['finger4-1.R', 'head', 'metacarpal1.L', 'neck02']), open(opt('--write_json'), 'w'))
        print('[teacher] RESULT: PASS')
        sys.exit(0)
    if '--labels' in a:        # LABEL_CHECK_PY
        sys.stdin.read()
        print('[check] RESULT: PASS')
        sys.exit(0)
    real()
if a[0] == 'train.py':
    import fake_train
    sys.exit(fake_train.main(a[1:]))
if a[0].endswith('diag_holdout_trace.py'):
    out = opt('--out')
    if os.environ.get('FAKE_TRACE_FAIL') and os.environ['FAKE_TRACE_FAIL'] in out:
        print('urllib.error.HTTPError: HTTP Error 504: Gateway Timeout')
        sys.exit(1)
    os.makedirs(out, exist_ok=True)
    teacher = 'exp7_final_model' in opt('--ckpt')
    v = dict(pve=73.7 if teacher else 118.5, pa_pve=50.3 if teacher else 80.1, mpjpe=77.9 if teacher else 126.0,
             recall=87.0 if teacher else 85.0, placement=dict(root_err_mm=700.0))
    json.dump(dict(ckpt=opt('--ckpt'), bone_fix=opt('--bone_fix', ''), variants={{'V1_eval_masked': v}}),
              open(os.path.join(out, 'summary.json'), 'w'))
    print('[load_model] bone_fix: %s' % opt('--bone_fix', 'checkpoint args (bone_fix_spec)'))
    print('=== PER-VARIANT METRICS\nvariant pve\nV1_eval_masked %.1f' % v['pve'])
    sys.exit(0)
if a[0].endswith('compute_bone_fix.py'):
    rc = int(os.environ.get('FAKE_FIX_RC', '0'))
    if rc == 0:
        json.dump(FIX, open(opt('--out'), 'w'))
    sys.exit(rc)
if a[0].endswith('model_cost.py'):
    one = lambda p, f, ms: dict(params=dict(network=p), total_flops=f, bench=dict(full_1person=dict(median_ms=ms), gpu='FAKE GPU'))
    json.dump(dict(results=[one(318.38e6, 1927.4e9, 59.7), one(101.38e6, 603.3e9, 28.5)]), open(opt('--json'), 'w'))
    print('[model_cost] fake')
    sys.exit(0)
real()
'''

FAKE_TRAIN = r'''"""Stand-in for train.py on the fake cluster: prints the start-up and per-epoch lines train.py prints for the given
flags (the formats of train.py's print() calls) and writes the checkpoints train.py keeps."""
import os, sys


def main(a):
    def opt(name, default=None):
        return a[a.index(name) + 1] if name in a else default
    save = os.path.join(opt('--save_dir'), opt('--name'))
    os.makedirs(os.path.join(save, 'checkpoints'), exist_ok=True)
    with open(os.path.join(opt('--save_dir'), 'fake_train_calls.txt'), 'a') as f:
        f.write(' '.join(a) + '\n')
    teacher = opt('--distill_teacher_ckpt')
    lam, lam_out = float(opt('--lambda_kd', 1.0)), float(opt('--lambda_kd_out', 0.0))
    kd_active = teacher is not None and (lam != 0 or lam_out > 0)
    if os.environ.get('FAKE_TRAIN_FORCE_KD'):
        kd_active = True
    fix = opt('--bone_fix', '')
    L = ['[speed] TF32 matmul/conv ENABLED (fp32 range, 10-bit mantissa inputs)',
         '[dinov2_hub] dinov2_vitb14 built from the local hub cache /fake/hub/facebookresearch_dinov2_main (no request to GitHub)']
    if fix and fix != 'none':
        L.append('[bone_fix] ON, 4 bone(s): neck02 178.8 deg, head 179.2 deg, metacarpal1.L 179.8 deg, finger4-1.R 179.7 deg (origin %s)' % fix)
    L += ['[pretrained] checkpoint contains 400 tensors', '[pretrained] backbone: 175/175 tensors (100.0%)']
    if teacher:
        L.append('[distill] teacher loaded from %s: 760/760 tensors (100.0%%)' % teacher)
        if fix and fix != 'none':
            L.append('[bone_fix] ON, 4 bone(s): neck02 178.8 deg, head 179.2 deg, metacarpal1.L 179.8 deg, finger4-1.R 179.7 deg (origin bone_fix_spec (checkpoint/run args))')
        if opt('--init_heads_from_teacher') == '1':
            L.append('[distill] heads initialised from teacher: 410/412 non-backbone tensors copied (99.5%); 343 backbone tensors intentionally left as-is')
        L.append('[distill] projector-fed heads: backbone 768 -> heads 1024 (feat_proj 787,456 params); feature KD is computed on the PROJECTED tokens')
        L.append('[distill] lambda_kd=%s T=%s softmax_dim=channel projector=no (same width)' % (lam, opt('--kd_temperature')))
        if lam == 0 and lam_out == 0 and not os.environ.get('FAKE_TRAIN_OLD'):
            L.append('[distill] NO distillation loss: lambda_kd=0 and lambda_kd_out=0. The teacher is used for the head copy only')
    elif not os.environ.get('FAKE_TRAIN_OLD'):
        L.append('[model] projector-fed heads, no teacher: backbone 768 -> heads 1024 (feat_proj 787,456 params); supervised losses only; heads at their fresh initialisation')
    L += ['[age_balance] exclusion-only: 2 images never sampled', '[mixed] training on AnnyOne + 3DPW-Anny (ratio 0.3)',
          '[loss] mask_3dpw_hands: 60 bones masked for 3DPW people']
    n_it, n_ep = int(opt('--n_iters_per_epoch')), int(opt('--max_iter')) // int(opt('--n_iters_per_epoch'))
    terms = ['total', 'bce', 'rotmat', 'v3d'] + (['kd', 'kd_out'] if kd_active else [])
    for e in range(n_ep):
        L += ['', 'TRAIN: ']
        L.append('EPOCH=%03d - i=00000/145000 -> ' % e + ' | '.join('%s: %.4f' % (t, 9.0 - e * 0.1 + k) for k, t in enumerate(terms)))
        L.append('[throughput] epoch %d: %d iters x batch 8 in 60s = 1.00 it/s, 8.00 images/s' % (e, n_it))
        L.append('[lr] epoch %d: lr = 3.00e-05' % e)
        if (e + 1) % int(opt('--eval_freq', 1)) == 0:
            for ds, base in (('annyone-holdout-1', 150.0), ('threedpw_anny-validation-1', 100.0)):
                L += ['', 'EVAL: ', '***EVAL METRICS - %s***' % ds]
                L += ['    - %s: %.1f' % (k, v) for k, v in (('pve', base - e), ('pa_pve', base - 40), ('precision', 90.0),
                                                           ('recall', 92.0), ('f1_score', 91.0), ('mpjpe', base + 5), ('pa_mpjpe', 70.0))]
                if opt('--val_loss') == '1':
                    vt = ' '.join('%s=%.4f' % (t, 8.0 - e * 0.1 + k) for k, t in enumerate(terms))
                    if teacher and not kd_active:
                        vt += ' kd=3.2100 kd_out=1.2300'
                    L.append('[val_loss] epoch=%03d ds=%s n=100 %s' % (e, ds, vt))
        open(os.path.join(save, 'checkpoints', '%05d.pt' % e), 'w').write('fake')
        for old in sorted(os.listdir(os.path.join(save, 'checkpoints')))[:-int(opt('--nb_max_ckpt', 3))]:
            os.remove(os.path.join(save, 'checkpoints', old))
    print('\n'.join(L))
    return 0
'''

FAKE_SRUN = '''#!/bin/bash
# srun on the fake cluster: no container, the command runs here, in --container-workdir
wd=""
while [ $# -gt 0 ]; do
  case "$1" in
    --container-workdir=*) wd="${1#*=}"; shift ;;
    -*) shift ;;
    *) break ;;
  esac
done
[ -n "$wd" ] && cd "$wd"
exec "$@"
'''

FAKE_EVAL = '''#!/bin/bash
# slurm/eval/run_eval_3dpw_test.sh on the fake cluster: reads MODEL_ENTRY EVAL_LABELS EVAL_BONE_FIX EVAL_3DPW_N and
# prints the header lines exp8 checks plus two metric blocks
IFS='|' read -r CK HD BB LABEL <<< "$MODEL_ENTRY"
echo "task 0: $LABEL  ->  $CK  (head_dim $HD, $BB)"
echo "labels: ${EVAL_LABELS}   run name: evaltest_${LABEL}__threedpw_anny_v23"
case "${EVAL_BONE_FIX:-}" in
  ''|inherit) D="the checkpoint's own (inherited from its saved args)" ;;
  none) D="forced off (--bone_fix none)" ;;
  *) D="${EVAL_BONE_FIX} (replaces the checkpoint's own)" ;;
esac
echo "bone fix: $D   model class: ${EVAL_MODEL_CLASS:-model}   3DPW-test images: ${EVAL_3DPW_N:-0} (0 = all)"
case "$LABEL" in *_teacher) P=89.0 ;; *) P=109.5 ;; esac
printf '***EVAL METRICS - annyone-test-1***\\n    - pve: 120.0\\n    - pa_pve: 82.0\\n    - recall: 85.0\\n    - mpjpe: 125.0\\n\\n'
printf '***EVAL METRICS - threedpw_anny-test-1***\\n    - pve: %s\\n    - pa_pve: 71.5\\n    - recall: 99.0\\n    - mpjpe: 111.0\\n' "$P"
'''


class FakeCluster:
    """A temp 'cluster': REPO with the real exp8 / exp10 / train.py / scripts (symlinks), stub slurm/env.sh, egl.sh and
    run_eval_3dpw_test.sh, the data files exp8's bash checks look for, and a bin/ with python, srun, sbatch, squeue."""

    def __init__(self, tmp):
        self.tmp = Path(tmp)
        self.repo = self.tmp / 'repo'
        self.bin = self.tmp / 'bin'
        self.repo.mkdir()
        self.bin.mkdir()
        for name in ('train.py', 'loss.py', 'bone_fix.py', 'dinov2_hub.py', 'blocks', 'scripts'):
            (self.repo / name).symlink_to(ROOT / name)
        (self.repo / 'slurm').mkdir()
        (self.repo / 'slurm' / 'experiments').symlink_to(ROOT / 'slurm' / 'experiments')
        (self.repo / 'slurm' / 'eval').mkdir()
        (self.repo / 'slurm' / 'env.sh').write_text('# fake env\n')
        (self.repo / 'slurm' / 'egl.sh').write_text('# fake egl\n')
        (self.repo / 'slurm' / 'eval' / 'run_eval_3dpw_test.sh').write_text(FAKE_EVAL)
        self.teacher = self.repo / 'results/experiments/exp7_final_model/v5_pc_dist5_bonefix_v23_e10/train/checkpoints/00009.pt'
        self.teacher.parent.mkdir(parents=True)
        self.teacher.write_text('fake teacher')
        (self.repo / 'models/multiHMR').mkdir(parents=True)
        (self.repo / 'models/multiHMR/multiHMR_672_B.pt').write_text('fake init')
        hub = self.tmp / 'torch_cache' / 'hub'
        (hub / 'checkpoints').mkdir(parents=True)
        (hub / 'checkpoints' / 'dinov2_vitb14_pretrain.pth').write_text('fake weights')
        (hub / 'facebookresearch_dinov2_main').mkdir()
        (self.repo / 'results/diagnostics').mkdir(parents=True)
        (self.repo / 'results/diagnostics/manifest.json').write_text(json.dumps(dict(entries=[dict(idx=5), dict(idx=7)])))
        labels = self.repo / 'threedpw_anny_v23' / 'labels'
        labels.mkdir(parents=True)
        (self.repo / 'configs').mkdir()
        tasks = []
        for split in ('train', 'validation', 'test'):
            (labels.parent / split).mkdir()
            fit = labels.parent / split / f'seq_{split}_p0.npz'
            fit.write_text('fake fit')
            os.utime(fit, (1_000_000_000, 1_000_000_000))          # older than the index written now
            (labels / f'index_{split}.txt').write_text(f'{labels}/seq_{split}/000000.pkl\n')
            tasks.append(f'{split} seq_{split} 0 10')
        (self.repo / 'configs' / 'threedpw_tasks.txt').write_text('\n'.join(tasks) + '\n')
        self.script('python', FAKE_PYTHON.format(real=sys.executable))
        self.script('python3', FAKE_PYTHON.format(real=sys.executable))
        (self.bin / 'fake_train.py').write_text(FAKE_TRAIN)
        self.script('srun', FAKE_SRUN)
        self.script('squeue', '#!/bin/bash\ncase " $* " in *" -j "*) echo "2:00:00" ;; esac\nexit 0\n')
        self.script('sbatch', '#!/bin/bash\n# records the submission; prints what Slurm prints\n'
                              'n=$(( $(cat "$FAKE_SBATCH_LOG" 2>/dev/null | wc -l) + 4242 ))\n'
                              'echo "$* | KD_MODE=${KD_MODE:-} RUN=${RUN:-} TEACHER_REMEASURE=${TEACHER_REMEASURE:-} '
                              'VAL_LOSS=${VAL_LOSS:-} EVAL_FREQ=${EVAL_FREQ:-} POST_ONLY=${POST_ONLY:-}" >> "$FAKE_SBATCH_LOG"\n'
                              'echo "Submitted batch job ${n}"\n')
        self.sbatch_log = self.tmp / 'sbatch_calls.txt'
        self.results = self.repo / 'results/experiments/exp8_distill_vitb'

    def script(self, name, text):
        p = self.bin / name
        p.write_text(text)
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def env(self, **extra):
        e = dict(REPO=str(self.repo), TORCH_CACHE=str(self.tmp / 'torch_cache'), USER='tester',
                 PATH=f'{self.bin}{os.pathsep}{os.environ["PATH"]}', PYTHONPATH=str(self.bin),
                 FAKE_SBATCH_LOG=str(self.sbatch_log), EPOCHS='3', N_ITERS='20', VAL_LOSS='1', EVAL_FREQ='1')
        e.update(extra)
        return e

    def job(self, script='slurm/experiments/exp8_distill_vitb.sh', **extra):
        """exp8 in job mode (as sbatch would start it)."""
        return run_bash(script, self.env(SLURM_JOB_ID='777', EXP8_JOB='1', **extra), cwd=self.repo)

    def login(self, script, **extra):
        return run_bash(script, self.env(**extra), cwd=self.repo)

    def run_dir(self, suffix=''):
        return self.results / f'student_vitb_from_{TAG}_e3_it20{suffix}'


class FakeClusterJobTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='exp10-cluster-')
        self.addCleanup(self._tmp.cleanup)
        self.c = FakeCluster(self._tmp.name)

    def assert_final_table(self, run, kd_mode):
        ft = json.loads((run / 'final_table.json').read_text())
        self.assertEqual(ft['inputs']['kd_mode'], kd_mode)
        rows = {(r['section'].split(' (')[0], r['metric']): r for r in ft['rows']}
        self.assertEqual(rows[('AnnyOne holdout', 'PVE')]['student'], 118.5)
        self.assertEqual(rows[('AnnyOne holdout', 'PVE')]['teacher'], 73.7)
        self.assertEqual(rows[('3DPW test', 'PVE')]['student'], 109.5)
        self.assertEqual(rows[('3DPW test', 'PVE')]['teacher'], 89.0)
        return ft

    def test_full_mode_job_runs_as_before(self):
        r = self.c.job()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        run = self.c.run_dir()
        self.assert_final_table(run, 'full')
        self.assertTrue((run / 'holdout_student_trainfix' / 'summary.json').exists())
        self.assertTrue((run / 'holdout_teacher' / 'summary.json').exists())
        self.assertEqual((run / 'last_checkpoint.txt').read_text().strip(), str(run / 'train/checkpoints/00002.pt'))
        settings = (run / 'settings.txt').read_text().splitlines()
        self.assertTrue(settings[-1].startswith('python train.py --train_data AnnyOne'))
        self.assertIn('--lambda_kd 1.0 --lambda_kd_out 1.0', settings[-1])
        self.assertIn('KD_MODE=full', settings[-2])
        self.assertNotIn('ABLATION', r.stdout)

    def test_nokd_job_trains_without_kd_terms_and_builds_the_table(self):
        r = self.c.job(KD_MODE='nokd')
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        run = self.c.run_dir('_nokd')
        self.assert_final_table(run, 'nokd')
        calls = (run / 'fake_train_calls.txt').read_text()
        self.assertIn('--lambda_kd 0 --lambda_kd_out 0', calls)
        self.assertIn('--init_heads_from_teacher 1', calls)
        self.assertIn(f'--bone_fix {run}/teacher_bone_fix.json', calls)
        self.assertIn('[distill] NO distillation loss', r.stdout)
        self.assertIn('ABLATION KD_MODE=nokd', r.stdout)
        self.assertTrue((run / 'holdout_student_trainfix' / 'summary.json').exists())

    def test_supervised_job_never_loads_the_teacher_in_training(self):
        r = self.c.job(KD_MODE='supervised', FAKE_FIX_RC='3')      # fresh heads without a flipped bone: no fix
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        run = self.c.run_dir('_supervised')
        ft = self.assert_final_table(run, 'supervised')
        calls = (run / 'fake_train_calls.txt').read_text()
        for word in ('--distill', '--init_heads', '--teacher_bf16', '--lambda_kd', '--kd_', '--freeze_heads', 'teacher_bone_fix'):
            self.assertNotIn(word, calls)
        self.assertIn('--bone_fix none', calls)
        self.assertIn('--head_dim 1024', calls)
        self.assertIn('[model] projector-fed heads, no teacher', r.stdout)
        self.assertIn('NO FLIPPED BONE', r.stdout)
        self.assertEqual((run / 'student_fix_arg.txt').read_text().strip(), 'none')
        self.assertEqual(ft['inputs']['student_fix'], 'none')
        self.assertIn('bone fix: forced off (--bone_fix none)', (run / 'eval_3dpw_test_student.log').read_text())
        self.assertFalse((run / 'holdout_student_trainfix').exists())
        self.assertEqual(json.loads((run / 'bone_fix.json').read_text())['bones'], {})

    def test_a_supervised_arm_without_an_estimation_dump_still_gets_its_table(self):
        r = self.c.job(KD_MODE='supervised', FAKE_FIX_RC='2')
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        self.assertIn('KD_MODE=supervised: no fix can be estimated', r.stdout)
        run = self.c.run_dir('_supervised')
        self.assert_final_table(run, 'supervised')
        self.assertEqual(json.loads((run / 'bone_fix.json').read_text())['status'], 'NO_ESTIMATE')
        for mode, suffix in (('full', ''), ('nokd', '_nokd')):          # every other mode stops, as before
            r = self.c.job(KD_MODE=mode, FAKE_FIX_RC='2')
            self.assertNotEqual(r.returncode, 0, mode)
            self.assertIn('ERROR: compute_bone_fix.py exit 2', r.stdout)
            self.assertFalse((self.c.run_dir(suffix) / 'final_table.json').exists())

    def test_the_job_stops_when_the_log_does_not_show_the_mode(self):
        r = self.c.job(KD_MODE='nokd', FAKE_TRAIN_FORCE_KD='1')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("a 'kd:' / 'kd_out:' term in the training loss lines", r.stdout)
        self.assertIn('stopping before the post-steps', r.stdout)
        self.assertFalse((self.c.run_dir('_nokd') / 'final_table.json').exists())
        r = self.c.job(KD_MODE='nokd', RUN='x_nokd_old', FAKE_TRAIN_OLD='1')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("'[distill] NO distillation loss'", r.stdout)
        r = self.c.job(KD_MODE='supervised', RUN='x_supervised_old', FAKE_TRAIN_OLD='1')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("'[model] projector-fed heads, no teacher'", r.stdout)

    def test_post_only_finishes_a_run_whose_post_step_died_of_a_504(self):
        env = dict(KD_T='16', RUN='student_kdT16_sweep')
        r = self.c.job(FAKE_TRACE_FAIL='holdout_student', **env)        # training fine, then a trace fails
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('HTTP Error 504: Gateway Timeout', r.stdout)
        self.assertIn('NOTE: exit 1, but the last epoch\'s checkpoint', r.stdout)
        self.assertIn('POST_ONLY=1 <the variables of this run', r.stdout)
        run = self.c.results / 'student_kdT16_sweep'
        self.assertFalse((run / 'final_table.json').exists())
        self.assertEqual((run / 'fake_train_calls.txt').read_text().count('\n'), 1)
        log_before = (run / 'train.log').read_text()
        # a plain rerun is refused (checkpoints exist); POST_ONLY with other settings is refused too
        r = self.c.job(**env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('already holds checkpoints', r.stdout)
        r = self.c.job(POST_ONLY='1', KD_T='8', RUN='student_kdT16_sweep')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('differs from the one these settings compose', r.stdout)
        self.assertIn('< --kd_temperature 16', r.stdout)
        self.assertIn('> --kd_temperature 8', r.stdout)
        r = self.c.job(POST_ONLY='1', EPOCHS='5', **env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('train/checkpoints/00004.pt (the checkpoint of epoch 5, the last) is missing', r.stdout)
        r = self.c.job(POST_ONLY='1', KD_MODE='nokd', RUN='student_kdT16_sweep_nokd')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('POST_ONLY=1 but', r.stdout)
        # the real thing: same environment + POST_ONLY=1
        r = self.c.job(POST_ONLY='1', **env)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        self.assertIn('=== 1. TRAINING SKIPPED (POST_ONLY=1)', r.stdout)
        self.assertNotIn('=== 1. TRAINING ->', r.stdout)
        self.assert_final_table(run, 'full')
        self.assertEqual((run / 'fake_train_calls.txt').read_text().count('\n'), 1)      # nothing was trained again
        self.assertEqual((run / 'train.log').read_text(), log_before)
        self.assertIn('POST_ONLY job 777', (run / 'post_only.txt').read_text())
        self.assertIn('--kd_temperature 16 ', (run / 'settings.txt').read_text().splitlines()[-1])
        self.assertNotIn('NOTE: exit', r.stdout)

    def test_post_only_refuses_an_unfinished_or_foreign_folder(self):
        r = self.c.job(POST_ONLY='1')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('POST_ONLY=1 but', r.stdout)
        self.assertIn('00002.pt (the checkpoint of epoch 3, the last) is missing', r.stdout)
        self.assertIn('REFUSED: nothing trained', r.stdout)
        # a nokd folder read as a distilled run: the settings differ, and with the override the log guard still stops it
        self.assertEqual(self.c.job(KD_MODE='nokd', RUN='abl_nokd').returncode, 0)
        r = self.c.job(POST_ONLY='1', KD_MODE='nokd', RUN='abl_nokd', TEACHER_REMEASURE='0',
                       TEACHER_TRACE_DIR=str(self.c.results / 'abl_nokd' / 'holdout_teacher'),
                       TEACHER_EVAL_LOG=str(self.c.results / 'abl_nokd' / 'eval_3dpw_test_teacher.log'))
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])              # POST_ONLY of an ablation arm works
        self.assertIn('teacher: reusing', r.stdout)


class FakeClusterSubmitTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='exp10-submit-')
        self.addCleanup(self._tmp.cleanup)
        self.c = FakeCluster(self._tmp.name)

    def test_exp10_submits_two_jobs_and_records_them(self):
        r = self.c.login('slurm/experiments/exp10_kd_ablation.sh')
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        self.assertIn('submitted 2 of 2 arm(s): nokd:4242 supervised:4243', r.stdout)
        # the reported run's teacher files do not exist on this fake cluster: the arms re-measure the teacher
        self.assertIn('teacher numbers: re-measured in each arm (TEACHER_REMEASURE=1)', r.stdout)
        calls = self.c.sbatch_log.read_text().splitlines()
        self.assertEqual(len(calls), 2)
        run = f'student_vitb_from_{TAG}_e3'
        self.assertIn(f'--job-name=exp8-{run}_nokd --time=16:00:00', calls[0])
        self.assertIn(f'| KD_MODE=nokd RUN={run}_nokd TEACHER_REMEASURE=1 VAL_LOSS=1 EVAL_FREQ=1 POST_ONLY=0', calls[0])
        self.assertIn(f'| KD_MODE=supervised RUN={run}_supervised TEACHER_REMEASURE=1', calls[1])
        records = list((self.c.repo / 'logs').glob('exp10_kd_ablation_*.txt'))
        self.assertEqual(len(records), 1)
        body = [l for l in records[0].read_text().splitlines() if not l.startswith('#')]
        self.assertEqual(body, [f'nokd  {run}_nokd  4242', f'supervised  {run}_supervised  4243'])

    def test_exp10_reuses_the_reported_teacher_numbers_when_they_match(self):
        rep = self.c.results / f'student_vitb_from_{TAG}_e3'
        (rep / 'holdout_teacher').mkdir(parents=True)
        (rep / 'holdout_teacher' / 'summary.json').write_text('{}')
        labels = self.c.repo / 'threedpw_anny_v23' / 'labels'
        (rep / 'eval_3dpw_test_teacher.log').write_text(
            f'task 0: exp8_x_teacher  ->  {self.c.teacher}  (head_dim 0, dinov2_vitl14)\nlabels: {labels}   run name: x\n')
        r = self.c.login('slurm/experiments/exp10_kd_ablation.sh', CHECK_ONLY='1')
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        self.assertIn('teacher numbers: reused (TEACHER_REMEASURE=0)', r.stdout)
        self.assertEqual(r.stdout.count('CHECK_ONLY=1: every check passed; nothing submitted'), 2)
        self.assertFalse(self.c.sbatch_log.exists())
        # another teacher in that log: not reusable, the arms fall back to re-measuring
        (rep / 'eval_3dpw_test_teacher.log').write_text(f'task 0: x  ->  /other/teacher.pt  (head_dim 0)\nlabels: {labels}   run name: x\n')
        r = self.c.login('slurm/experiments/exp10_kd_ablation.sh', CHECK_ONLY='1')
        self.assertEqual(r.returncode, 0, r.stdout[-2000:])
        self.assertIn('teacher numbers: re-measured in each arm', r.stdout)

    def test_post_only_submission_of_a_trained_run(self):
        env = dict(KD_T='16', RUN='student_kdT16_sweep')
        self.assertNotEqual(self.c.job(FAKE_TRACE_FAIL='est_trace', **env).returncode, 0)
        r = self.c.login('slurm/experiments/exp8_distill_vitb.sh', POST_ONLY='1', **env)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        self.assertIn('settings.txt records the same train.py command', r.stdout)
        self.assertIn('Submitted batch job 4242', r.stdout)
        call = self.c.sbatch_log.read_text()
        self.assertIn('--job-name=exp8-student_kdT16_sweep --time=02:00:00', call)
        self.assertIn('POST_ONLY=1', call)
        # without POST_ONLY the same folder is refused, and an untrained folder is refused with POST_ONLY
        r = self.c.login('slurm/experiments/exp8_distill_vitb.sh', **env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('NOT SUBMITTED', r.stdout)
        r = self.c.login('slurm/experiments/exp8_distill_vitb.sh', POST_ONLY='1', KD_T='2', RUN='student_kdT2_sweep')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('POST_ONLY=1 but', r.stdout)


# =============================================================================================== dinov2_hub
class FakeHub:
    def __init__(self, hub_dir, fail=0, error=None):
        self.dir, self.calls, self.fail = str(hub_dir), [], fail
        self.error = error or urllib.error.HTTPError('https://github.com/facebookresearch/dinov2/tree/main/', 504,
                                                     'Gateway Timeout', None, None)

    def get_dir(self):
        return self.dir

    def load(self, repo, name, source='github', pretrained=False):
        self.calls.append((repo, name, source, pretrained))
        if source == 'github' and len([c for c in self.calls if c[2] == 'github']) <= self.fail:
            raise self.error
        return f'model:{name}:{source}'


class Dinov2HubTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='dinov2-hub-')
        self.addCleanup(self._tmp.cleanup)
        self.hub_dir = Path(self._tmp.name)
        self._env = {k: os.environ.pop(k, None) for k in ('DINOV2_HUB_SOURCE', 'DINOV2_HUB_RETRIES', 'DINOV2_HUB_WAIT')}
        self.addCleanup(lambda: [os.environ.__setitem__(k, v) if v is not None else os.environ.pop(k, None)
                                 for k, v in self._env.items()])

    def cache(self):
        d = self.hub_dir / 'facebookresearch_dinov2_main'
        d.mkdir()
        (d / 'hubconf.py').write_text('# fake')
        return d

    def load(self, hub, **kw):
        waits = []
        with redirect_stdout(io.StringIO()) as buf:
            m = dinov2_hub.load_dinov2('dinov2_vitb14', hub=hub, sleep=waits.append, **kw)
        return m, waits, buf.getvalue()

    def test_the_cached_code_is_loaded_locally_without_any_github_call(self):
        d = self.cache()
        hub = FakeHub(self.hub_dir, fail=99)                 # every GitHub call would fail with a 504
        m, waits, out = self.load(hub, pretrained=True)
        self.assertEqual(m, 'model:dinov2_vitb14:local')
        self.assertEqual(hub.calls, [(str(d), 'dinov2_vitb14', 'local', True)])
        self.assertEqual(waits, [])
        self.assertIn('built from the local hub cache', out)

    def test_without_a_cache_the_original_call_is_retried_through_a_504(self):
        hub = FakeHub(self.hub_dir, fail=2)
        m, waits, out = self.load(hub)
        self.assertEqual(m, 'model:dinov2_vitb14:github')
        self.assertEqual([c[0] for c in hub.calls], ['facebookresearch/dinov2'] * 3)
        self.assertEqual(waits, [5.0, 10.0])
        self.assertIn('network error on attempt 1/6 (HTTPError: HTTP Error 504: Gateway Timeout)', out)

    def test_a_cache_that_appears_between_attempts_is_used(self):
        hub = FakeHub(self.hub_dir, fail=99)
        cache = self.cache
        waits = []

        def sleep(w):
            waits.append(w)
            if len(waits) == 1:
                cache()

        with redirect_stdout(io.StringIO()):
            m = dinov2_hub.load_dinov2('dinov2_vitl14', hub=hub, sleep=sleep)
        self.assertEqual(m, 'model:dinov2_vitl14:local')
        self.assertEqual(len(waits), 1)

    def test_giving_up_names_the_error_and_other_errors_are_not_swallowed(self):
        os.environ['DINOV2_HUB_RETRIES'] = '3'
        hub = FakeHub(self.hub_dir, fail=99)
        with self.assertRaises(RuntimeError) as cm, redirect_stdout(io.StringIO()):
            dinov2_hub.load_dinov2('dinov2_vitb14', hub=hub, sleep=lambda w: None)
        self.assertIn('after 3 attempt(s)', str(cm.exception))
        self.assertIn('504', str(cm.exception))
        self.assertEqual(len(hub.calls), 3)
        hub = FakeHub(self.hub_dir, fail=99, error=KeyError('not a network error'))
        with self.assertRaises(KeyError):
            dinov2_hub.load_dinov2('dinov2_vitb14', hub=hub, sleep=lambda w: None)
        self.assertEqual(len(hub.calls), 1)

    def test_source_switch(self):
        self.cache()
        os.environ['DINOV2_HUB_SOURCE'] = 'github'
        hub = FakeHub(self.hub_dir)
        m, _, out = self.load(hub)
        self.assertEqual(m, 'model:dinov2_vitb14:github')
        self.assertIn('DINOV2_HUB_SOURCE=github', out)
        shutil.rmtree(self.hub_dir / 'facebookresearch_dinov2_main')
        os.environ['DINOV2_HUB_SOURCE'] = 'local'
        with self.assertRaises(FileNotFoundError):
            dinov2_hub.load_dinov2('dinov2_vitb14', hub=FakeHub(self.hub_dir), sleep=lambda w: None)
        os.environ['DINOV2_HUB_SOURCE'] = 'bogus'
        with self.assertRaises(ValueError):
            dinov2_hub.load_dinov2('dinov2_vitb14', hub=FakeHub(self.hub_dir))

    def test_both_model_classes_build_the_vit_through_the_helper(self):
        for rel in ('blocks/dinov2.py', 'multi_hmr_anny/encoder.py'):
            src = (ROOT / rel).read_text()
            self.assertNotIn("torch.hub.load('facebookresearch/dinov2'", src, rel)
            self.assertIn('from dinov2_hub import load_dinov2', src, rel)
            self.assertIn('load_dinov2(self.name, pretrained=pretrained)', src, rel)

    def test_real_torch_hub_local_source(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch not importable here')
        d = self.cache()
        (d / 'hubconf.py').write_text('dependencies = []\n\ndef dinov2_vitb14(pretrained=False):\n    return ("vitb", pretrained)\n')
        old = torch.hub.get_dir()
        torch.hub.set_dir(str(self.hub_dir))
        try:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(dinov2_hub.load_dinov2('dinov2_vitb14', pretrained=True), ('vitb', True))
        finally:
            torch.hub.set_dir(old)


# =============================================================================================== train.py (ast)
class TrainPyKdModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / 'train.py').read_text()
        cls.tree = ast.parse(cls.src)

    def test_kd_terms_active_truth_table(self):
        fn = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kd_terms_active')
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'train.py', 'exec'), ns)
        active = ns['kd_terms_active']
        A = lambda **kw: type('Args', (), kw)()
        self.assertTrue(active(A(lambda_kd=1.0, lambda_kd_out=1.0)))        # the reported recipe
        self.assertTrue(active(A(lambda_kd=1.0, lambda_kd_out=0.0)))        # train.py's own defaults
        self.assertTrue(active(A(lambda_kd=0.0, lambda_kd_out=0.5)))
        self.assertFalse(active(A(lambda_kd=0.0, lambda_kd_out=0.0)))       # exp8 KD_MODE=nokd
        self.assertFalse(active(A(lambda_kd=0, lambda_kd_out=0)))
        self.assertFalse(active(A()))

    def test_the_training_step_runs_the_teacher_only_with_an_active_term(self):
        self.assertIn('if self.teacher is not None and self.kd_loss is not None and kd_terms_active(self.args):\n'
                      '                    with torch.no_grad():\n'
                      '                        t_out = self.teacher(x, is_training=True,', self.src)
        self.assertIn("if float(self.args.lambda_kd) != 0.0:\n                        loss = loss + self.args.lambda_kd * kd\n", self.src)
        # the output-KD term is still added only with a positive weight, in training and in the validation loss
        self.assertIn("if lko > 0 and pred['rotmat'].shape == t_out['rotmat'].shape:", self.src)
        self.assertIn('if lko > 0:\n                            loss = loss + lko * kd_out\n', self.src)

    def test_start_up_lines_exp8_greps_for_exist_in_train_py(self):
        self.assertIn('print("[distill] NO distillation loss: lambda_kd=0 and lambda_kd_out=0.', self.src)
        self.assertIn('print(f"[model] projector-fed heads, no teacher: backbone {model.backbone_dim} -> heads {model.embed_dim} "', self.src)
        self.assertIn("elif getattr(model, 'feat_proj', None) is not None and not args.eval_only:", self.src)
        exp8 = EXP8.read_text()
        for needle in ('grep -q "def kd_terms_active" "${REPO}/train.py"', 'grep -q "NO distillation loss" "${REPO}/train.py"',
                       'grep -q "projector-fed heads, no teacher" "${REPO}/train.py"',
                       r'grep -qE "^\[distill\] NO distillation loss"', r'grep -qE "^\[model\] projector-fed heads, no teacher"'):
            self.assertIn(needle, exp8)

    def test_default_flags_keep_distillation_on(self):
        defaults = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'add_argument' and node.args \
                    and isinstance(node.args[0], ast.Constant) and node.args[0].value in ('--lambda_kd', '--lambda_kd_out', '--head_dim'):
                defaults[node.args[0].value] = {k.arg: k.value for k in node.keywords}['default'].value
        self.assertEqual(defaults, {'--lambda_kd': 1.0, '--lambda_kd_out': 0.0, '--head_dim': 0})


# =============================================================================================== validation table
def write_log(run, val3, valh, val_loss=None, kd_t=4.0, train_terms=('total', 'kd', 'kd_out'), settings=True, extra=''):
    """A run folder with a train.log in train.py's formats. val3 / valh: {1-based epoch: pve} of the 3DPW-validation /
    AnnyOne holdout evaluations; val_loss: {epoch: (3dpw terms dict, holdout terms dict)}."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    n = max(list(val3) + list(valh) + [1])
    L = [f'[distill] lambda_kd=1.0 T={kd_t} softmax_dim=channel projector=no (same width)']
    for e in range(1, n + 1):
        L += ['', 'TRAIN: ', '  0%|          | 0/145000 [00:00<?, ?it/s]\r  0%|          | 1/145000 [00:01<40:00:00,  1.01s/it]']
        L.append(f'EPOCH={e - 1:03d} - i=00950/145000 -> ' + ' | '.join(f'{t}: {10.0 - e * 0.1 + k:.4f}' for k, t in enumerate(train_terms)))
        L.append(f'[throughput] epoch {e - 1}: 1000 iters x batch 8 in 690s = 1.45 it/s, 11.59 images/s')
        L.append(f'[lr] epoch {e - 1}: lr = 3.00e-05')
        for i, (ds, series) in enumerate((('annyone-holdout-1', valh), ('threedpw_anny-validation-1', val3))):
            if e not in series:
                continue
            L += ['', 'EVAL: ', f'***EVAL METRICS - {ds}***']
            L += [f'    - {k}: {v:.1f}' for k, v in (('pve', series[e]), ('pa_pve', series[e] - 30), ('precision', 60.0),
                                                     ('recall', 99.0 - i * 10), ('f1_score', 75.0), ('mpjpe', series[e] + 2), ('pa_mpjpe', 55.0))]
            if val_loss and e in val_loss:
                terms = val_loss[e][1 - i]
                L.append(f'[val_loss] epoch={e - 1:03d} ds={ds} n=100 ' + ' '.join(f'{k}={v:.4f}' for k, v in terms.items()))
                L.append(f'[val_loss] epoch={e - 1:03d} ds={ds} note has_hand_labels: not in the batch: set to 0 for every person')
    (run / 'train.log').write_text('\n'.join(L) + '\n')
    if settings:
        (run / 'settings.txt').write_text(f'exp8_distill_vitb {run.name}, job 1\nLAMBDA_KD=1.0 LAMBDA_KD_OUT=1.0 KD_T={kd_t} FREEZE_HEADS_EPOCHS=5\n{extra}python train.py --lambda_kd 1.0\n')
    return run


class SweepValidationTableTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='val-table-')
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / 'results' / 'experiments' / 'exp8_distill_vitb'
        final = {1: 88.4, 2: 87.1, 4: 87.9, 8: 87.5, 16: 89.0}
        for t, f in final.items():
            val3 = {e: 120.0 - e for e in range(1, 30)}
            val3[30] = f
            if t == 2:
                val3[27] = 86.6                              # the best epoch is not the last one
            valh = {e: 170.0 - e * 1.5 - t * 0.1 for e in range(1, 31)}
            vl = {e: (dict(total=9.0 - e * 0.01 + t, kd=3.0, kd_out=1.0), dict(total=12.0 - e * 0.02, kd=4.0, kd_out=2.0)) for e in range(1, 31)}
            write_log(self.root / f'{PREFIX}_kdT{t}_sweep', val3, valh, vl, kd_t=float(t))
        # T 16: trained, the post-steps died: train.log and settings only (as on the cluster); nothing else is needed
        write_log(self.root / PREFIX, {e: 105.0 - e * 0.58 for e in range(5, 31, 5)}, {e: 172.0 - e * 1.64 for e in range(5, 31, 5)}, settings=False)
        write_log(self.root / PREFIX.replace('_e30', '_e60'), {e: 100.0 - e * 0.234 for e in range(5, 61, 5)}, {e: 150.0 - e * 0.616 for e in range(5, 61, 5)})

    def test_collect_reads_final_best_and_loss(self):
        r = valtab.collect(str(self.root / f'{PREFIX}_kdT2_sweep'), 'log')
        self.assertEqual((r['group'], r['kd_t'], r['epochs'], r['n_evals'], r['final_eval_epoch']), ('sweep', 2.0, 30, 30, 30))
        self.assertEqual((r['3dpwval_pve'], r['3dpwval_pa_pve'], r['3dpwval_mpjpe'], r['3dpwval_recall']), (87.1, 57.1, 89.1, 89.0))
        self.assertEqual((r['holdout_pve'], r['holdout_recall']), (124.8, 99.0))
        self.assertEqual((r['best_3dpwval_pve_epoch'], r['best_3dpwval_pve']), (27, 86.6))
        self.assertAlmostEqual(r['final_minus_best_3dpwval_pve'], 0.5)
        self.assertAlmostEqual(r['3dpwval_loss_total'], 9.0 - 0.3 + 2)
        self.assertAlmostEqual(r['holdout_loss_total'], 12.0 - 0.6)
        self.assertEqual(r['source'], 'train.log')
        old = valtab.collect(str(self.root / PREFIX), 'log')
        self.assertEqual((old['group'], old['kd_t'], old['epochs'], old['n_evals']), ('other', 4.0, 30, 6))
        self.assertEqual(old['3dpwval_pve'], 87.6)
        self.assertIsNone(old['3dpwval_loss_total'])
        self.assertEqual(old['best_3dpwval_pve_epoch'], 30)

    def test_main_table_selection_lines_and_missing_pieces(self):
        empty = self.root / f'{PREFIX}_kdT32_sweep'
        empty.mkdir()                                           # a folder without any log
        out = Path(self._tmp.name) / 'out'
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = valtab.main([str(self.root / f'{PREFIX}_kdT*_sweep'), str(self.root / PREFIX),
                              str(self.root / PREFIX.replace('_e30', '_e60')), str(self.root / 'no_such_run'),
                              '--out', str(out), '--source', 'log'])
        text = buf.getvalue()
        self.assertEqual(rc, 0, text)
        self.assertIn('sweep, 3DPW-val PVE (final epoch): lowest at T = 2 (87.1 mm), highest at T = 16 (89.0 mm); '
                      'spread across 5 temperatures (max - min) = 1.9 mm; no value for T = 32', text)
        self.assertIn('sweep, AnnyOne holdout PVE (final epoch): lowest at T = 16 (123.4 mm), highest at T = 1 (124.9 mm)', text)
        self.assertIn(f'last epoch vs best epoch, 3DPW-val PVE: final - best is at most +0.5 mm over the 7 run(s) with a curve '
                      f'(largest in {PREFIX}_kdT2_sweep: best 86.6 at epoch 27, final 87.1)', text)
        self.assertIn(f'training length, {PREFIX} (30 epochs, T = 4): final-epoch 3DPW-val PVE 87.6, AnnyOne holdout PVE 122.8 mm (6 in-training evaluations)', text)
        self.assertIn(f'training length, {PREFIX.replace("_e30", "_e60")} (60 epochs, T = 4): final-epoch 3DPW-val PVE 86.0, AnnyOne holdout PVE 113.0 mm (12 in-training evaluations)', text)
        self.assertIn(f'NO 3DPW-validation numbers (no train.log / events, or the run has not evaluated yet): {PREFIX}_kdT32_sweep', text)
        self.assertIn('NOT FOUND (no such folder): ' + str(self.root / 'no_such_run'), text)
        rows = list(csv.DictReader(open(out / 'sweep_validation_table.csv')))
        self.assertEqual([r['kd_t'] for r in rows], ['1.0', '2.0', '4.0', '8.0', '16.0', '32.0', '4.0', '4.0'])
        self.assertEqual([r['group'] for r in rows], ['sweep'] * 6 + ['other'] * 2)
        t16 = rows[4]
        self.assertEqual((t16['3dpwval_pve'], t16['epochs'], t16['best_3dpwval_pve_epoch']), ('89.0', '30', '30'))
        self.assertEqual(rows[5]['3dpwval_pve'], '')
        self.assertEqual(rows[6]['3dpwval_loss_total'], '')
        md = (out / 'sweep_validation_table.md').read_text()
        self.assertIn(f'| {PREFIX}_kdT2_sweep | 2 | 30 | 30 | 87.1 | 57.1 | 89.1 | 89.0 | 124.8 | 94.8 | 126.8 | 99.0 | 10.700 | 11.400 | 86.6 (27) | +0.5 | train.log |', md)
        self.assertIn(f'| {PREFIX}_kdT32_sweep | 32 | - | - | - |', md)

    def test_no_run_at_all_is_an_error(self):
        with redirect_stdout(io.StringIO()) as buf:
            rc = valtab.main([str(self.root / 'nothing*'), '--out', str(Path(self._tmp.name) / 'o')])
        self.assertEqual(rc, 1)
        self.assertIn('ERROR: no run folder matches', buf.getvalue())

    def test_tensorboard_source_agrees_with_the_log(self):
        from scripts.analysis import plot_loss_curves as curves
        if not curves.tensorboard_available():
            self.skipTest('tensorboard not importable here')
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        run = self.root / f'{PREFIX}_kdT2_sweep'
        (run / 'train').mkdir()
        w = EventFileWriter(str(run / 'train'))
        log = valtab.collect(str(run), 'log')
        from scripts.analysis.plot_loss_curves import parse_train_log
        data = parse_train_log((run / 'train.log').read_text())
        for e in range(1, 31):
            w.add_event(Event(step=e * 1000 - 50, summary=Summary(value=[Summary.Value(tag='loss/total', simple_value=1.0)])))
            for ds in data['val']:
                for k, v in data['val'][ds][e].items():
                    w.add_event(Event(step=e * 1000, summary=Summary(value=[Summary.Value(tag=f'{ds}/{k}', simple_value=v)])))
                w.add_event(Event(step=e * 1000, summary=Summary(value=[Summary.Value(tag=f'val_loss/{ds}/total', simple_value=data['val_loss'][ds][e]['total'])])))
        w.close()
        tb = valtab.collect(str(run), 'auto')
        self.assertEqual(tb['source'], 'tensorboard')
        for k in ('3dpwval_pve', 'holdout_pve', '3dpwval_recall', 'best_3dpwval_pve', '3dpwval_loss_total'):
            self.assertAlmostEqual(tb[k], log[k], places=3, msg=k)
        self.assertEqual((tb['best_3dpwval_pve_epoch'], tb['epochs'], tb['n_evals']), (27, 30, 30))


# =============================================================================================== ablation table
def add_post_steps(run, holdout, threedpw, teacher=None, kd_mode=None):
    run = Path(run)
    (run / 'holdout_student').mkdir()
    (run / 'holdout_student' / 'summary.json').write_text(json.dumps(dict(variants={'V1_eval_masked': holdout})))
    (run / 'eval_3dpw_test_student.log').write_text(
        '***EVAL METRICS - annyone-test-1***\n    - pve: 999.9\n\n***EVAL METRICS - threedpw_anny-test-1***\n'
        + ''.join(f'    - {k}: {v:.1f}\n' for k, v in threedpw.items()))
    if teacher:
        (run / 'holdout_teacher').mkdir()
        (run / 'holdout_teacher' / 'summary.json').write_text(json.dumps(dict(variants={'V1_eval_masked': teacher[0]})))
        (run / 'eval_3dpw_test_teacher.log').write_text(
            '***EVAL METRICS - threedpw_anny-test-1***\n' + ''.join(f'    - {k}: {v:.1f}\n' for k, v in teacher[1].items()))
    if kd_mode:
        (run / 'final_table.json').write_text(json.dumps(dict(inputs=dict(kd_mode=kd_mode), rows=[])))


class KdAblationTableTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='kd-ablation-')
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        m = lambda p: dict(pve=p, pa_pve=p - 40, mpjpe=p + 7, recall=85.0)
        rep = write_log(self.root / PREFIX, {e: 105.0 - e * 0.58 for e in range(5, 31, 5)}, {e: 172.0 - e * 1.64 for e in range(5, 31, 5)})
        add_post_steps(rep, m(119.6), m(110.1), teacher=(m(73.7), m(89.0)))
        full_loss = {30: (dict(total=14.0, kd=3.0, kd_out=1.0, rotmat=2.0), dict(total=20.0, kd=4.0, kd_out=2.0))}
        t4 = write_log(self.root / f'{PREFIX}_kdT4_sweep', {e: 118.0 - e for e in range(1, 31)}, {e: 150.0 - e for e in range(1, 31)}, full_loss)
        add_post_steps(t4, m(117.1), m(110.1))
        nokd_loss = {30: (dict(total=11.0, rotmat=2.5, kd=9.5, kd_out=3.5), dict(total=15.5, kd=10.0, kd_out=4.0))}
        self.nokd = write_log(self.root / f'{PREFIX}_nokd', {e: 121.0 - e for e in range(1, 31)}, {e: 158.0 - e for e in range(1, 31)}, nokd_loss,
                              train_terms=('total', 'rotmat'), extra='EVAL_FREQ=1 VAL_LOSS=1 VAL_3DPW_N=300 KD_MODE=nokd\n')
        (self.nokd / 'settings.txt').write_text((self.nokd / 'settings.txt').read_text().replace('LAMBDA_KD=1.0 LAMBDA_KD_OUT=1.0', 'LAMBDA_KD=0 LAMBDA_KD_OUT=0'))
        add_post_steps(self.nokd, m(123.4), m(113.0), kd_mode='nokd')
        self.sup = self.root / f'{PREFIX}_supervised'           # not there yet: pending

    def args(self, **kw):
        a = dict(reported=self.root / PREFIX, sweep_t4=self.root / f'{PREFIX}_kdT4_sweep', nokd=self.nokd, supervised=self.sup)
        a.update(kw)
        out = []
        for k, v in a.items():
            if v:
                out += [f'--{k}', str(v)]
        return out + ['--out', str(self.root / 'out'), '--source', 'log']

    def run_main(self, **kw):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = abl.main(self.args(**kw))
        return rc, buf.getvalue()

    def test_table_with_a_pending_arm(self):
        rc, text = self.run_main()
        self.assertEqual(rc, 0, text)
        self.assertIn('| Student, reported (T 4, job 3614191) | complete | 30 | 87.6 | 57.6 | 89.6 | 89.0 | 122.8 | 92.8 | 124.8 | 99.0 |', text)
        self.assertIn('| no distillation loss (teacher heads + bone fix kept) | complete | 30 | 91.0 | 61.0 | 93.0 | 89.0 | 128.0 | 98.0 | 130.0 | 99.0 |', text)
        self.assertIn('| no teacher in training (supervised only) | pending | - | pending | pending |', text)
        # validation loss: the task part removes lambda x kd terms in the distilled run; total in the nokd arm
        self.assertIn('| Student, sweep T 4 arm (same recipe, re-run) | 14.000 | 10.000 | 3.000 | 1.000 | 20.000 | 14.000 | 4.000 | 2.000 |', text)
        self.assertIn('| no distillation loss (teacher heads + bone fix kept) | 11.000 | 11.000 | 9.500 | 3.500 | 15.500 | 15.500 | 10.000 | 4.000 |', text)
        self.assertIn('| Student, reported (T 4, job 3614191) | - | - | - | - | - | - | - | - |', text)
        self.assertIn('| no distillation loss (teacher heads + bone fix kept) | 123.4 | 83.4 | 130.4 | 85.0 | 113.0 | 73.0 | 120.0 | 85.0 |', text)
        self.assertIn('| teacher (ViT-L, re-measured in the reported run) | 73.7 | 33.7 | 80.7 | 85.0 | 89.0 | 49.0 | 96.0 | 85.0 |', text)
        self.assertIn('- sweep_t4 - reported (run-to-run noise (same recipe trained twice)): val_3dpwval_pve +0.4, val_holdout_pve -2.8, '
                      'test_holdout_pve -2.5, test_holdout_mpjpe -2.5, test_3dpw_pve +0.0, test_3dpw_mpjpe +0.0', text)
        self.assertIn('- nokd - sweep_t4 (removing the two distillation losses (head copy, head freeze, teacher bone fix kept)): '
                      'val_3dpwval_pve +3.0, val_holdout_pve +8.0, test_holdout_pve +6.3, test_holdout_mpjpe +6.3, test_3dpw_pve +2.9, test_3dpw_mpjpe +2.9', text)
        self.assertIn('- supervised - nokd (also removing the copied teacher heads and the teacher bone fix): pending', text)
        self.assertIn('NOT COMPLETE (no final test numbers yet): supervised [pending]', text)
        rows = {r['arm']: r for r in csv.DictReader(open(self.root / 'out' / 'kd_ablation_table.csv'))}
        self.assertEqual(list(rows), ['reported', 'sweep_t4', 'nokd', 'supervised', 'teacher'])
        self.assertEqual((rows['nokd']['kd_mode'], rows['nokd']['lambda_kd'], rows['nokd']['status']), ('nokd', '0.0', 'complete'))
        self.assertEqual((rows['reported']['kd_mode'], rows['sweep_t4']['lambda_kd_out']), ('full', '1.0'))
        self.assertEqual((rows['supervised']['status'], rows['supervised']['test_3dpw_pve'], rows['supervised']['val_3dpwval_pve']), ('pending', '', ''))
        self.assertEqual(rows['teacher']['test_3dpw_pve'], '89.0')
        self.assertTrue((self.root / 'out' / 'kd_ablation_table.md').read_text().startswith('Distillation ablation'))

    def test_an_arm_that_is_still_training_and_a_supervised_arm_without_kd_terms(self):
        sup_loss = {e: (dict(total=13.0 - e * 0.01, rotmat=3.0), dict(total=18.0, rotmat=3.5)) for e in range(1, 13)}
        write_log(self.sup, {e: 140.0 - e for e in range(1, 13)}, {e: 190.0 - e for e in range(1, 13)}, sup_loss,
                  train_terms=('total', 'rotmat'), extra='EVAL_FREQ=1 VAL_LOSS=1 VAL_3DPW_N=300 KD_MODE=supervised\n')
        rc, text = self.run_main()
        self.assertEqual(rc, 0, text)
        self.assertIn('| no teacher in training (supervised only) | training or post-steps pending | 12 | 128.0 | 98.0 | 130.0 | 89.0 | 178.0 |', text)
        self.assertIn('| no teacher in training (supervised only) | 12.880 | 12.880 | - | - | 18.000 | 18.000 | - | - |', text)
        self.assertIn('| no teacher in training (supervised only) | - | - | - | - | - | - | - | - |', text)
        self.assertIn('- supervised - nokd (also removing the copied teacher heads and the teacher bone fix): val_3dpwval_pve +37.0, val_holdout_pve +50.0', text)
        self.assertIn('NOT COMPLETE (no final test numbers yet): supervised [training or post-steps pending]', text)

    def test_a_folder_of_the_wrong_mode_is_flagged(self):
        rc, text = self.run_main(supervised=self.nokd)
        self.assertEqual(rc, 0)
        self.assertIn(f'NOTE: supervised: {PREFIX}_nokd records KD_MODE=nokd, expected supervised: wrong folder for this row?', text)
        with redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(abl.main(['--out', str(self.root / 'o2')]), 1)
        self.assertIn('ERROR: give at least one of', buf.getvalue())


# =============================================================================================== Relative Human
class RhRecallByCrowdTests(unittest.TestCase):
    DATA = ROOT / 'docs' / 'thesis' / 'data'

    def test_bins(self):
        bins = crowd.parse_bins('1,2,3-4,5-8,9+')
        self.assertEqual([crowd.bin_of(n, bins) for n in (1, 2, 3, 4, 5, 8, 9, 40)], ['1', '2', '3-4', '3-4', '5-8', '5-8', '9+', '9+'])
        self.assertIsNone(crowd.bin_of(3, crowd.parse_bins('1,2')))
        for bad in ('', '0', '3-2', '1,1', '2+,5', 'a'):
            with self.assertRaises(ValueError):
                crowd.parse_bins(bad)

    def test_model_names(self):
        self.assertEqual(crowd.model_name('docs/thesis/data/rh_per_person_final.csv'), 'final')
        self.assertEqual(crowd.model_name('results/diagnostics/relative_human_val_student_vitb_e30_00029/per_person.csv'), 'student_vitb_e30_00029')

    def test_final_and_base_reproduce_the_thesis_numbers(self):
        if not (self.DATA / 'rh_per_person_final.csv').exists():
            self.skipTest('docs/thesis/data/rh_per_person_final.csv not in this checkout')
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = crowd.main([str(self.DATA / 'rh_per_person_final.csv'), 'base=' + str(self.DATA / 'rh_per_person_base.csv'), '--out', tmp])
            self.assertEqual(rc, 0, buf.getvalue())
            text = buf.getvalue()
            self.assertIn('| people per image | images | people | recall final | recall base |', text)
            for line in ('| 1 | 21 | 21 | 0.762 | 0.333 |', '| 2 | 314 | 628 | 0.785 | 0.510 |', '| 3-4 | 190 | 630 | 0.678 | 0.476 |',
                         '| 5-8 | 82 | 509 | 0.566 | 0.444 |', '| 9+ | 28 | 342 | 0.453 | 0.439 |', '| All | 635 | 2130 | 0.647 | 0.471 |'):
                self.assertIn(line, text)
            rows = list(csv.DictReader(open(Path(tmp) / 'rh_recall_by_crowd.csv')))
            self.assertEqual(len(rows), 12)
            ref = self.DATA / 'rh_recall_by_crowd.csv'       # the table the thesis macros were made from (12 Nov)
            if ref.exists():
                mine = {(r['model'], r['bin']): (r['images'], r['people'], r['recall']) for r in rows if r['bin'] != 'All'}
                theirs = {(r['model'], r['bin']): (r['images'], r['people'], r['recall']) for r in csv.DictReader(open(ref))}
                self.assertEqual(mine, theirs)

    def test_synthetic_files_with_different_people_and_bad_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, bad = Path(tmp) / 'a.csv', Path(tmp) / 'b.csv', Path(tmp) / 'bad.csv'
            a.write_text('image,matched,iou\nx.jpg,1,0.5\ny.jpg,1,0.4\ny.jpg,0,0.0\nz.jpg,1,0.9\nz.jpg,1,0.8\nz.jpg,0,0\n')
            b.write_text('image,matched\nx.jpg,0\n')
            bad.write_text('image,iou\nx.jpg,0.5\n')
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = crowd.main([f'A={a}', f'B={b}', '--out', str(Path(tmp) / 'o')])
            self.assertEqual(rc, 0)
            self.assertIn('one table per model', buf.getvalue())
            self.assertIn('| 3-4 | 1 | 3 | 2 | 0.667 |', buf.getvalue())
            self.assertIn('| All | 3 | 6 | 4 | 0.667 |', buf.getvalue())
            self.assertIn('| 9+ | 0 | 0 | 0 | - |', buf.getvalue())
            with redirect_stdout(io.StringIO()) as buf:
                self.assertEqual(crowd.main([str(bad), '--out', str(Path(tmp) / 'o')]), 1)
            self.assertIn("needs the columns 'image' and 'matched'", buf.getvalue())
            with redirect_stdout(io.StringIO()) as buf:
                self.assertEqual(crowd.main([str(Path(tmp) / 'none.csv'), '--out', str(Path(tmp) / 'o')]), 1)


class RelativeHumanStudentCommandTests(unittest.TestCase):
    RUN = f'{CLUSTER}/results/experiments/exp8_distill_vitb/{PREFIX}'

    def line(self, **env):
        r = run_bash(EVAL_RH, dict(env, PRINT_ONLY='1'))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout.strip()

    def test_tag_of_the_exp8_layout_names_the_run_not_train(self):
        out = self.line(MODEL=f'{self.RUN}/train/checkpoints/00029.pt', BONE_FIX=f'{self.RUN}/bone_fix.json')
        self.assertIn(f'TAG={PREFIX}_00029 ', out)
        self.assertIn(f"BONE_FIX='{self.RUN}/bone_fix.json'", out)
        self.assertTrue(out.endswith(f'OUT=results/diagnostics/relative_human_val_{PREFIX}_00029'))
        out = self.line(MODEL=f'{self.RUN}/train/checkpoints/00029.pt', BONE_FIX=f'{self.RUN}/bone_fix.json', TAG='student_vitb_e30_00029')
        self.assertTrue(out.endswith('OUT=results/diagnostics/relative_human_val_student_vitb_e30_00029'))

    def test_the_older_layouts_keep_their_tags(self):
        self.assertIn('TAG=anny_s2_shape_v5_00099 ', self.line())
        self.assertIn('TAG=exp7_v5_final_00009 ', self.line(MODEL=f'{CLUSTER}/models/multiHMR/exp7_v5_final_00009.pt'))
        self.assertIn('TAG=anny_s2_shape_v5_00099 ', self.line(MODEL=f'{CLUSTER}/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt'))

    def test_the_python_side_takes_a_json_bone_fix_and_refuses_the_original_class(self):
        src = (ROOT / 'scripts' / 'diagnostics' / 'eval_relative_human.py').read_text()
        self.assertIn('dc.load_model(ckpt, device, bone_fix=args.bone_fix)', src)
        common = (ROOT / 'scripts' / 'diagnostics' / 'diag_common.py').read_text()
        self.assertIn("bone_fix_lib.from_arg(bone_fix, raw_args.get('bone_fix_spec'))", common)
        self.assertIn('checkpoint is not a model.py Model (looks like the original Multi_HMR class)', common)
        self.assertIn("feat_proj={'yes' if getattr(model, 'feat_proj', None) is not None else 'no'}", common)

    def test_bone_fix_from_arg_prefers_the_json_file_over_the_checkpoint_spec(self):
        import bone_fix
        flip_x = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        flip_z = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'bone_fix.json')
            bone_fix.save(path, {'neck02': flip_x, 'head': flip_z}, note='the student\'s own estimate')
            ckpt_spec = bone_fix.load(path)
            ckpt_spec = dict(ckpt_spec, bones={'neck02': flip_z})           # what the checkpoint args carry (the teacher's)
            spec, origin = bone_fix.from_arg(path, ckpt_spec)
            self.assertEqual(origin, path)
            self.assertEqual(sorted(spec['bones']), ['head', 'neck02'])
            self.assertEqual([[round(v) for v in row] for row in spec['bones']['neck02']], [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            spec, origin = bone_fix.from_arg('', ckpt_spec)
            self.assertEqual(sorted(spec['bones']), ['neck02'])
            self.assertIn('checkpoint args', origin)
            self.assertEqual(bone_fix.from_arg('none', ckpt_spec)[0], None)


if __name__ == '__main__':
    unittest.main()
