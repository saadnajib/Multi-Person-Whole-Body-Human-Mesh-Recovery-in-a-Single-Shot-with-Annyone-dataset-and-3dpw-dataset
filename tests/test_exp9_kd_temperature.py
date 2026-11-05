"""CPU-only checks of the exp9 pieces: exp8's KD_T run suffix and --val_loss plumbing (bash, LOCAL_CHECK=1), the
exp9 sweep launcher (LOCAL_CHECK=1), the train.log parser and verdict of plot_loss_curves.py on synthetic logs that
mimic train.py's line formats, the sweep collector on fake run folders, and train.py's --val_loss argument.

train.py itself is never imported (it needs torch, smplx, anny and the GPU stack): its argparse is read with ast.
"""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis import kd_temperature_sweep as sweep   # noqa: E402
from scripts.analysis import plot_loss_curves as curves      # noqa: E402

EXP8 = ROOT / 'slurm' / 'experiments' / 'exp8_distill_vitb.sh'
EXP9 = ROOT / 'slurm' / 'experiments' / 'exp9_kd_temperature.sh'
TAG = 'v5_pc_dist5_bonefix_v23_e10_00009'
PREFIX = f'student_vitb_from_{TAG}_e30'


def run_bash(script, env, cwd=ROOT):
    full = {k: v for k, v in os.environ.items() if k not in ('RUN', 'KD_T', 'TEMPS', 'EPOCHS', 'TIME')}
    full.update(env)
    return subprocess.run(['bash', str(script)], cwd=str(cwd), env=full, capture_output=True, text=True)


def have_matplotlib():
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


# ----------------------------------------------------------------------------------------------- exp8 / exp9 (bash)
class Exp8RunSuffixTests(unittest.TestCase):
    def run_line(self, **env):
        env.setdefault('LOCAL_CHECK', '1')
        r = run_bash(EXP8, env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        first = r.stdout.splitlines()[0]
        self.assertTrue(first.startswith('=== exp8_distill_vitb '), first)
        return first.split()[-1], r.stdout

    def test_default_and_every_spelling_of_four_keep_the_reported_name(self):
        for kd_t in (None, '4', '4.0', '4.00', '4e0'):
            env = {} if kd_t is None else {'KD_T': kd_t}
            run, _ = self.run_line(**env)
            self.assertEqual(run, PREFIX, kd_t)

    def test_other_temperatures_get_a_suffix_after_the_epoch_count(self):
        for kd_t, suffix in (('8', '_T8'), ('0.5', '_T0.5'), ('16', '_T16'), ('1', '_T1')):
            run, out = self.run_line(KD_T=kd_t)
            self.assertEqual(run, PREFIX + suffix)
            self.assertIn(f'--kd_temperature {kd_t} ', out.replace('\\\n', ' '))

    def test_suffix_order_and_explicit_run_override(self):
        run, _ = self.run_line(KD_T='8', SMOKE='1')
        self.assertEqual(run, f'student_vitb_from_{TAG}_e1_it20_T8_smoke')
        run, _ = self.run_line(KD_T='8', RUN='my_run')
        self.assertEqual(run, 'my_run')

    def test_val_loss_and_eval_freq_reach_the_train_command_and_the_settings(self):
        _, out = self.run_line()
        flat = out.replace('\\\n', ' ')
        self.assertIn('--eval_freq 5 ', flat)
        self.assertIn('--val_loss 0 ', flat)
        self.assertIn('validation loss (--val_loss) 0', out)
        _, out = self.run_line(VAL_LOSS='1', EVAL_FREQ='1', KD_T='2')
        flat = out.replace('\\\n', ' ')
        self.assertIn('--eval_freq 1 ', flat)
        self.assertIn('--val_loss 1 ', flat)
        self.assertIn('in-training evaluation every 1 epoch(s)', out)
        r = run_bash(EXP8, {'LOCAL_CHECK': '1', 'VAL_LOSS': '2'})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('VAL_LOSS=2', r.stdout)


class Exp9LauncherTests(unittest.TestCase):
    def test_local_check_names_one_sweep_run_per_temperature(self):
        r = run_bash(EXP9, {'LOCAL_CHECK': '1'})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = r.stdout
        for t in ('1', '2', '4', '8', '16'):
            run = f'{PREFIX}_kdT{t}_sweep'
            self.assertIn(f'KD_T={t}  ->  results/experiments/exp8_distill_vitb/{run}', out)
            self.assertIn(f'=== exp8_distill_vitb {run}', out)
            self.assertIn(f'--kd_temperature {t} ', out.replace('\\\n', ' '))
        self.assertNotIn(f'=== exp8_distill_vitb {PREFIX}\n', out)          # the reported folder is never a target
        flat = out.replace('\\\n', ' ')
        self.assertEqual(flat.count('--val_loss 1 '), 5)
        self.assertEqual(flat.count('--eval_freq 1 '), 5)
        self.assertEqual(flat.count('--time=16:00:00'), 5)
        self.assertIn('kd_temperature_sweep.py --runs "results/experiments/exp8_distill_vitb/%s_kdT*_sweep" --reported results/experiments/exp8_distill_vitb/%s' % (PREFIX, PREFIX), out)
        self.assertIn('plot_loss_curves.py results/experiments/exp8_distill_vitb/%s_kdT*_sweep' % PREFIX, out)
        self.assertIn('LOCAL_CHECK=1: nothing checked against the cluster, nothing submitted', out)

    def test_custom_temperatures_and_refusals(self):
        r = run_bash(EXP9, {'LOCAL_CHECK': '1', 'TEMPS': '0.5 32', 'SWEEP_TAG': 'sweep2'})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(f'=== exp8_distill_vitb {PREFIX}_kdT0.5_sweep2', r.stdout)
        self.assertIn(f'=== exp8_distill_vitb {PREFIX}_kdT32_sweep2', r.stdout)
        self.assertNotIn('_kdT4_', r.stdout)
        for bad in ({'KD_T': '8'}, {'RUN': 'x'}, {'TEMPS': 'abc'}, {'TEMPS': '4 -1'}):
            env = {'LOCAL_CHECK': '1'}
            env.update(bad)
            r = run_bash(EXP9, env)
            self.assertNotEqual(r.returncode, 0, bad)
            self.assertIn('ERROR', r.stdout, bad)


# ----------------------------------------------------------------------------------------------- loss curves
TERMS = ['total', 'bce', 'offset', 'rotmat', 'shape', 'dist', 'transl', 'j3d', 'v3d', 'j2d', 'v2d', 'kd', 'kd_out']


def synthetic_log(n_epochs, eval_freq, val_loss, n_iters=1000, steps_skipped=2):
    """train.py stdout as exp8 captures it (tqdm \\r fragments included), with decreasing train losses, the eval
    blocks every eval_freq epochs and, with val_loss, the [val_loss] lines. Returns (text, expected)."""
    lines = ['[distill] teacher loaded from /x/00009.pt: 500/500 tensors (100.0%)',
             '[distill] lambda_kd=1.0 T=4.0 softmax_dim=channel projector=no (same width)', '']
    exp = dict(train={}, val={}, val_loss={}, steps={})
    for e in range(n_epochs):
        lines.append('')
        lines.append('TRAIN: ')
        lines.append('  0%|          | 0/145000 [00:00<?, ?it/s]\r  0%|          | 1/145000 [00:01<40:00:00,  1.01s/it]')
        last = None
        for i in range(0, n_iters, 50):
            vals = {t: round((20.0 - e * 0.5) * (1 + k) / 10 - i * 1e-4, 4) for k, t in enumerate(TERMS)}
            last = vals
            parts = ' | '.join(f'{t}: {vals[t]:.4f}' for t in TERMS)
            lines.append(f'EPOCH={e:03d} - i={i:05d}/145000 -> {parts}')
            lines.append(f'  {i // 1450}%|{"#" * (i // 100)}| {i}/145000 [00:35<28:00:00,  1.43it/s]\r')
        exp['train'][e + 1] = {t: float(f'{last[t]:.4f}') for t in TERMS}
        n_steps = n_iters - steps_skipped
        exp['steps'][e + 1] = n_steps
        lines.append(f'[throughput] epoch {e}: {n_steps} iters x batch 8 in 690s = {n_steps / 690:.2f} it/s, {n_steps * 8 / 690:.2f} images/s')
        lines.append(f'[lr] epoch {e}: lr = 3.00e-05')
        if (e + 1) % eval_freq == 0:
            for ds, base in (('annyone-holdout-1', 170.0), ('threedpw_anny-validation-1', 105.0)):
                pve = round(base - (e + 1) * 1.5, 1)
                lines.append('')
                lines.append('EVAL: ')
                lines.append('  0%|          | 0/100 [00:00<?, ?it/s]')
                lines.append(f'i=0 - Recall=100.0 - PVE={pve:.1f} - PA-PVE=80.0 - MPJPE=0.0 - PA-MPJPE=0.0')
                lines.append(f'***EVAL METRICS - {ds}***')
                for k, v in (('pve', pve), ('pa_pve', 80.1), ('precision', 95.0), ('recall', 92.0),
                             ('f1_score', 93.5), ('mpjpe', pve + 5), ('pa_mpjpe', 70.2)):
                    lines.append(f'    - {k}: {v:.1f}')
                exp['val'].setdefault(ds, {})[e + 1] = dict(pve=pve, pa_pve=80.1, recall=92.0, mpjpe=round(pve + 5, 1))
                if val_loss:
                    vl = {t: round((22.0 - e * 0.45) * (1 + k) / 10, 4) for k, t in enumerate(TERMS)}
                    body = ' '.join(f'{t}={vl[t]:.4f}' for t in TERMS)
                    lines.append(f'[val_loss] epoch={e:03d} ds={ds} n=100 {body}')
                    lines.append(f'[val_loss] epoch={e:03d} ds={ds} note has_hand_labels: not in the batch: set to 1 for every person')
                    exp['val_loss'].setdefault(ds, {})[e + 1] = {t: float(f'{vl[t]:.4f}') for t in TERMS}
    return '\n'.join(lines) + '\n', exp


class LossCurveParserTests(unittest.TestCase):
    def test_old_run_log_without_val_loss(self):
        text, exp = synthetic_log(30, 5, val_loss=False)
        data = curves.parse_train_log(text)
        self.assertEqual(data['train'], exp['train'])
        self.assertEqual(data['steps'], exp['steps'])
        self.assertEqual(data['val'], exp['val'])
        self.assertEqual(data['val_loss'], {})
        self.assertEqual(sorted(data['val']['annyone-holdout-1']), [5, 10, 15, 20, 25, 30])

    def test_new_run_log_with_val_loss_every_epoch(self):
        text, exp = synthetic_log(6, 1, val_loss=True)
        data = curves.parse_train_log(text)
        self.assertEqual(data['val_loss'], exp['val_loss'])
        self.assertEqual(sorted(data['val_loss']), ['annyone-holdout-1', 'threedpw_anny-validation-1'])
        self.assertEqual(data['val'], exp['val'])

    def test_slopes_and_verdicts(self):
        falling = {e: 100.0 - e for e in range(1, 31)}
        flat = {e: 50.0 + (0.01 if e % 2 else -0.01) for e in range(1, 31)}
        rising = {e: 50.0 + e * 0.5 for e in range(1, 31)}
        s = curves.slope_last_fraction(falling, 0.2, 30)
        self.assertEqual(s[2], [25, 26, 27, 28, 29, 30])
        self.assertAlmostEqual(s[0], -1.0)
        self.assertEqual(curves.trend(s[1], 0.25), 'falling')
        self.assertEqual(curves.trend(curves.slope_last_fraction(flat, 0.2, 30)[1], 0.25), 'flat')
        self.assertEqual(curves.trend(curves.slope_last_fraction(rising, 0.2, 30)[1], 0.25), 'rising')
        # a validation curve measured every 5 epochs: the last-20% window of 30 epochs holds epochs 25 and 30
        sparse = {5: 165.1, 10: 170.0, 15: 139.5, 20: 133.0, 25: 130.5, 30: 122.8}
        s = curves.slope_last_fraction(sparse, 0.2, 30)
        self.assertEqual(s[2], [25, 30])
        self.assertAlmostEqual(s[0], (122.8 - 130.5) / 5)
        self.assertIsNone(curves.slope_last_fraction({30: 1.0}, 0.2, 30))
        self.assertIn('longer training may help', curves.verdict('falling', 'falling'))
        self.assertIn('over-fitting, stop', curves.verdict('falling', 'flat'))
        self.assertIn('over-fitting, stop', curves.verdict('falling', 'rising'))
        self.assertIn('converged', curves.verdict('flat', 'flat'))
        self.assertIn('no verdict', curves.verdict('unknown', 'flat'))

    def test_main_on_an_old_style_run_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / f'{PREFIX}'
            run.mkdir()
            text, exp = synthetic_log(30, 5, val_loss=False)
            (run / 'train.log').write_text(text)
            out = Path(tmp) / 'out'
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = curves.main([str(run), '--out', str(out), '--source', 'log', '--no_plot'])
            self.assertEqual(rc, 0, buf.getvalue())
            text_out = buf.getvalue()
            self.assertIn('the validation PVE stands in for it', text_out)
            self.assertIn('VERDICT (val PVE annyone-holdout-1): both still falling -> longer training may help', text_out)
            self.assertIn('source: train.log', text_out)
            rows = list(__import__('csv').DictReader(open(out / f'{PREFIX}_loss_curves.csv')))
            self.assertEqual(len(rows), 30)
            self.assertEqual(float(rows[0]['train_total']), exp['train'][1]['total'])
            self.assertEqual(rows[0]['val_annyone_holdout_1_pve'], '')
            self.assertEqual(float(rows[4]['val_annyone_holdout_1_pve']), exp['val']['annyone-holdout-1'][5]['pve'])
            self.assertEqual(float(rows[29]['val_threedpw_anny_validation_1_pve']), exp['val']['threedpw_anny-validation-1'][30]['pve'])
            self.assertEqual(float(rows[29]['train_kd']), exp['train'][30]['kd'])
            self.assertFalse(list(run.iterdir()) != [run / 'train.log'], 'the run folder must stay untouched')
            if have_matplotlib():
                with redirect_stdout(io.StringIO()):
                    rc = curves.main([str(run), '--out', str(out), '--source', 'log'])
                self.assertEqual(rc, 0)
                self.assertTrue((out / f'{PREFIX}_loss_curves.png').stat().st_size > 1000)
                self.assertTrue((out / f'{PREFIX}_loss_curves.pdf').exists())

    def test_tensorboard_source_agrees_with_the_log(self):
        if not curves.tensorboard_available():
            self.skipTest('tensorboard not importable here')
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        text, exp = synthetic_log(4, 2, val_loss=True, n_iters=200, steps_skipped=1)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run_e4'
            (run / 'train').mkdir(parents=True)
            (run / 'train.log').write_text(text)
            w = EventFileWriter(str(run / 'train'))

            def scalar(tag, value, step):
                w.add_event(Event(step=step, summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)])))
            step = 0
            for e in range(1, 5):
                for i in range(0, 200, 50):              # running means: only the last one per epoch counts
                    scalar('loss/total', exp['train'][e]['total'] + (10.0 if i < 150 else 0.0), step + i)
                    scalar('loss/kd', exp['train'][e]['kd'], step + i)
                step += exp['steps'][e]                   # current_iter after the epoch
                if e in exp['val']['annyone-holdout-1']:
                    for ds in exp['val']:
                        scalar(f'{ds}/pve', exp['val'][ds][e]['pve'], step)
                        scalar(f'val_loss/{ds}/total', exp['val_loss'][ds][e]['total'], step)
                scalar('workload/evaluate', 1.0, e - 1)
            w.close()
            data, src = curves.load_run(str(run), 'auto')
            self.assertEqual(src, 'tensorboard')
            r4 = lambda d, k: {e: round(v[k], 4) for e, v in d.items()}     # event files hold float32
            self.assertEqual(r4(data['train'], 'total'), r4(exp['train'], 'total'))
            self.assertEqual(r4(data['val']['annyone-holdout-1'], 'pve'), r4(exp['val']['annyone-holdout-1'], 'pve'))
            self.assertEqual(r4(data['val_loss']['threedpw_anny-validation-1'], 'total'),
                             r4(exp['val_loss']['threedpw_anny-validation-1'], 'total'))
            self.assertNotIn('workload', data['val'])
            data_log, src = curves.load_run(str(run), 'log')
            self.assertEqual(src, 'train.log')
            self.assertEqual(r4(data_log['train'], 'total'), r4(data['train'], 'total'))


# ----------------------------------------------------------------------------------------------- sweep collector
def fake_run(root, name, kd_t, holdout, threedpw, settings=True, table_only=False, teacher=None):
    run = Path(root) / name
    run.mkdir(parents=True)
    if settings and not table_only:
        (run / 'settings.txt').write_text(f'exp8_distill_vitb {name}, job 1\nLAMBDA_KD=1.0 LAMBDA_KD_OUT=1.0 KD_T={kd_t} FREEZE_HEADS_EPOCHS=5\n')
    if table_only:
        rows = []
        for lab, k in (('PVE', 'pve'), ('PA-PVE', 'pa_pve'), ('MPJPE', 'mpjpe'), ('recall', 'recall')):
            rows.append(dict(section='AnnyOne holdout (100 img, V1 trace)', metric=lab, teacher=1.0, student=holdout[k]))
            rows.append(dict(section='3DPW test (evaluate(), threedpw_anny_v23)', metric=lab, teacher=1.0, student=threedpw[k]))
        (run / 'final_table.json').write_text(json.dumps(dict(rows=rows)))
        return run
    (run / 'holdout_student').mkdir()
    (run / 'holdout_student' / 'summary.json').write_text(json.dumps(dict(
        ckpt='x', variants={'V1_eval_masked': dict(holdout, placement=dict(root_err_mm=800.0)), 'V0_eval': dict(pve=999.0)})))
    log = ['task 0: exp8_x  ->  /x.pt', 'labels: /labels ', 'bone fix: none', '',
           '***EVAL METRICS - annyone-test-1***', '    - pve: 999.9', '    - recall: 1.0', '',
           '***EVAL METRICS - threedpw_anny-test-1***'] + [f'    - {k}: {v:.1f}' for k, v in threedpw.items()] + ['', 'done']
    (run / 'eval_3dpw_test_student.log').write_text('\n'.join(log) + '\n')
    if teacher:
        (run / 'holdout_teacher').mkdir()
        (run / 'holdout_teacher' / 'summary.json').write_text(json.dumps(dict(variants={'V1_eval_masked': dict(pve=teacher[0])})))
        (run / 'eval_3dpw_test_teacher.log').write_text(f'***EVAL METRICS - threedpw_anny-test-1***\n    - pve: {teacher[1]:.1f}\n')
    return run


class SweepCollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='kd-sweep-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'results' / 'experiments' / 'exp8_distill_vitb'
        self.rows = {}
        for t, h, p in ((1, 125.0, 112.0), (2, 121.0, 110.5), (4, 118.9, 109.0), (8, 117.0, 111.5), (16, 123.0, 115.0)):
            hold = dict(pve=h, pa_pve=h - 40, mpjpe=h + 7, recall=85.0)
            dpw = dict(pve=p, pa_pve=p - 38, mpjpe=p + 2, recall=99.0)
            fake_run(self.root, f'{PREFIX}_kdT{t}_sweep', float(t), hold, dpw, table_only=(t == 16),
                     teacher=(73.9, 89.2) if t == 4 else None)
            self.rows[t] = (hold, dpw)
        hold = dict(pve=119.6, pa_pve=80.8, mpjpe=127.2, recall=85.4)
        dpw = dict(pve=110.1, pa_pve=71.8, mpjpe=112.0, recall=99.0)
        fake_run(self.root, PREFIX, 4.0, hold, dpw, settings=False)   # an old folder: KD_T only in train.log
        (self.root / PREFIX / 'train.log').write_text('[distill] lambda_kd=1.0 T=4.0 softmax_dim=channel projector=no\n')

    def test_collect_run_reads_summary_eval_log_and_final_table(self):
        r = sweep.collect_run(str(self.root / f'{PREFIX}_kdT2_sweep'))
        self.assertEqual(r['temperature'], 2.0)
        self.assertEqual((r['holdout_pve'], r['holdout_pa_pve'], r['holdout_mpjpe'], r['holdout_recall']), (121.0, 81.0, 128.0, 85.0))
        self.assertEqual((r['3dpw_pve'], r['3dpw_pa_pve'], r['3dpw_mpjpe'], r['3dpw_recall']), (110.5, 72.5, 112.5, 99.0))
        r = sweep.collect_run(str(self.root / f'{PREFIX}_kdT16_sweep'))        # final_table.json only
        self.assertEqual(r['temperature'], 16.0)
        self.assertEqual((r['holdout_pve'], r['3dpw_mpjpe']), (123.0, 117.0))
        r = sweep.collect_run(str(self.root / PREFIX), 'reported')
        self.assertEqual(r['temperature'], 4.0)
        self.assertEqual(r['holdout_pve'], 119.6)
        empty = Path(self.tmp.name) / 'empty_kdT3_sweep'
        empty.mkdir()
        r = sweep.collect_run(str(empty))
        self.assertEqual(r['temperature'], 3.0)
        self.assertIsNone(r['holdout_pve'])

    def test_main_writes_csv_prints_pick_and_noise(self):
        out = Path(self.tmp.name) / 'exp9'
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = sweep.main(['--runs', str(self.root / f'{PREFIX}_kdT*_sweep'), '--reported', str(self.root / PREFIX),
                             '--out', str(out), '--no_plot'])
        self.assertEqual(rc, 0, buf.getvalue())
        text = buf.getvalue()
        self.assertIn('AnnyOne holdout PVE: T = 8 (117.0 mm)', text)
        self.assertIn('3DPW test PVE: T = 4 (109.0 mm)', text)
        self.assertIn('PICK: T = 4', text)
        self.assertIn('holdout_pve -0.7', text)      # sweep T4 118.9 minus reported 119.6
        self.assertIn('3dpw_pve -1.1', text)
        self.assertIn('teacher re-measured in %s_kdT4_sweep: holdout PVE 73.9, 3DPW PVE 89.2' % PREFIX, text)
        import csv
        rows = list(csv.DictReader(open(out / 'kd_temperature_sweep.csv')))
        self.assertEqual([r['temperature'] for r in rows], ['1.0', '2.0', '4.0', '8.0', '16.0', '4.0'])
        self.assertEqual([r['source'] for r in rows][-1], 'reported')
        self.assertEqual(rows[2]['holdout_pve'], '118.9')
        if have_matplotlib():
            with redirect_stdout(io.StringIO()):
                rc = sweep.main(['--runs', str(self.root / f'{PREFIX}_kdT*_sweep'), '--reported', str(self.root / PREFIX), '--out', str(out)])
            self.assertEqual(rc, 0)
            self.assertTrue((out / 'kd_temperature_sweep.png').stat().st_size > 1000)
            self.assertTrue((out / 'kd_temperature_sweep.pdf').exists())


# ----------------------------------------------------------------------------------------------- train.py (ast)
class TrainPyValLossFlagTests(unittest.TestCase):
    def test_argparse_accepts_val_loss_with_default_0(self):
        tree = ast.parse((ROOT / 'train.py').read_text())
        found = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'add_argument' \
                    and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--val_loss':
                found = {k.arg: k.value for k in node.keywords}
        self.assertIsNotNone(found, 'train.py has no --val_loss')
        self.assertEqual(found['default'].value, 0)
        self.assertEqual(found['type'].id, 'int')
        self.assertEqual([c.value for c in found['choices'].elts], [0, 1])

    def test_evaluate_calls_the_hook_only_behind_the_flag(self):
        src = (ROOT / 'train.py').read_text()
        self.assertIn("val_loss_meters = {} if int(getattr(self.args, 'val_loss', 0)) else None", src)
        self.assertIn('if val_loss_meters is not None:\n                self.val_loss_step(', src)
        self.assertIn('def val_loss_step(self, x, gt, ds_name, meters):', src)
        self.assertIn('def val_loss_report(self, ds_tag, meters):', src)
        self.assertIn('f"val_loss/{ds_tag}/{k}"', src)
        self.assertIn('print(f"[val_loss] epoch={self.current_epoch:03d} ds={ds_tag} n={n} {vals}"', src)


if __name__ == '__main__':
    unittest.main()
