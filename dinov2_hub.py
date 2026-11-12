"""
Network-robust construction of the DINOv2 ViT (used by blocks/dinov2.py and multi_hmr_anny/encoder.py).

WHY. torch.hub.load('facebookresearch/dinov2', name) without a branch asks GitHub for the default branch on EVERY call
(torch.hub._parse_repo_info: urlopen('https://github.com/facebookresearch/dinov2/tree/main/')), even when the code is
already in the hub cache, and re-raises any HTTP error other than 404. Every model build does it: train.py at start-up
and every post-step that rebuilds a model (diag_holdout_trace.py, model_cost.py, the --eval_only evaluations). One
"urllib.error.HTTPError: HTTP Error 504: Gateway Timeout" from that probe killed the post-steps of a fully trained run
(exp9, KD_T 16, job 3717598), although nothing had to be downloaded.

WHAT. load_dinov2(name, pretrained):
  1. the hub cache first: <torch.hub.get_dir()>/facebookresearch_dinov2_main (or _master) with a hubconf.py is loaded
     with source='local' = the same hubconf entry point and the same code as before, with no request to GitHub. The
     ImageNet weights of pretrained=True come from <hub dir>/checkpoints/<name>_pretrain.pth when that file exists
     (torch.hub.load_state_dict_from_url reads the cached file without a request), else they are downloaded;
  2. no cached code: the original call torch.hub.load('facebookresearch/dinov2', name) (downloads the code once);
  3. a network error in 1 or 2 (URLError incl. HTTPError, timeouts, connection resets) is retried with a growing
     wait (5, 10, 20, 40, 60 s; DINOV2_HUB_RETRIES attempts, default 6), re-checking the cache before every attempt.
  Environment: DINOV2_HUB_SOURCE = auto (default, as above) | github (always the original call, still retried) |
  local (never ask GitHub for the code); DINOV2_HUB_RETRIES; DINOV2_HUB_WAIT (first wait in seconds, default 5).
One '[dinov2_hub]' line per build says which path was taken.
"""
import http.client
import os
import socket
import time
import urllib.error

REPO = 'facebookresearch/dinov2'
NETWORK_ERRORS = (urllib.error.URLError, http.client.HTTPException, socket.timeout, TimeoutError, ConnectionError)


def cached_repo_dir(hub_dir):
    """The DINOv2 code in a torch.hub cache folder (the folder holding hubconf.py), or None."""
    for ref in ('main', 'master'):
        d = os.path.join(hub_dir, 'facebookresearch_dinov2_%s' % ref)
        if os.path.isfile(os.path.join(d, 'hubconf.py')):
            return d
    return None


def load_dinov2(name, pretrained=False, hub=None, sleep=time.sleep, verbose=True):
    """The DINOv2 ViT `name` (e.g. dinov2_vitb14) as torch.hub builds it; see the module docstring.
    hub / sleep are injection points for the tests (default: torch.hub, time.sleep)."""
    if hub is None:
        import torch
        hub = torch.hub
    mode = os.environ.get('DINOV2_HUB_SOURCE', 'auto').strip().lower() or 'auto'
    if mode not in ('auto', 'github', 'local'):
        raise ValueError('DINOV2_HUB_SOURCE=%s (auto | github | local)' % mode)
    tries = max(1, int(os.environ.get('DINOV2_HUB_RETRIES', '6')))
    wait0 = max(0.0, float(os.environ.get('DINOV2_HUB_WAIT', '5')))
    last = None
    for attempt in range(tries):
        local = None if mode == 'github' else cached_repo_dir(hub.get_dir())
        try:
            if local is not None:
                model = hub.load(local, name, source='local', pretrained=pretrained)
                if verbose:
                    print('[dinov2_hub] %s built from the local hub cache %s (no request to GitHub)' % (name, local),
                          flush=True)
                return model
            if mode == 'local':
                raise FileNotFoundError('DINOV2_HUB_SOURCE=local but no facebookresearch_dinov2_main/hubconf.py under %s'
                                        % hub.get_dir())
            model = hub.load(REPO, name, pretrained=pretrained)
            if verbose:
                print('[dinov2_hub] %s built through torch.hub.load(%r) (%s)' % (
                    name, REPO, 'DINOV2_HUB_SOURCE=github' if mode == 'github' else 'no local hub cache yet'), flush=True)
            return model
        except NETWORK_ERRORS as e:
            last = e
            if attempt + 1 >= tries:
                break
            wait = min(60.0, wait0 * (2 ** attempt))
            print('[dinov2_hub] %s: network error on attempt %d/%d (%s: %s); retrying in %.0f s'
                  % (name, attempt + 1, tries, type(e).__name__, e, wait), flush=True)
            sleep(wait)
    raise RuntimeError('[dinov2_hub] %s could not be built after %d attempt(s); last network error: %s: %s. Nothing of '
                       'the run folder is lost: rerun the step (exp8: POST_ONLY=1 with the run\'s settings).'
                       % (name, tries, type(last).__name__, last)) from last
