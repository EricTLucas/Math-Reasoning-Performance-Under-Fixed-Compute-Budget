#!/usr/bin/env python3
"""Prepare, run and export a resumable three-attempt Argon baseline (stdlib CLI)."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time

ATTEMPTS = 3

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(str(tmp), str(path))

def rows_from(path):
    rows, seen = [], set()
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = json.dumps(r['id'], sort_keys=True)
            if key in seen or not isinstance(r['problem'], str) or not r['problem'].strip():
                raise ValueError('Duplicate ID or empty problem')
            seen.add(key)
            r['answer'] = int(r['answer'])
            if not 0 <= r['answer'] <= 99999:
                raise ValueError('Reference outside supported integer range')
            rows.append({'id': r['id'], 'problem': r['problem'], 'answer': r['answer']})
    if not rows:
        raise ValueError('Empty dataset')
    return rows

def prepare(args):
    rows = rows_from(args.data)
    root = Path(args.experiment).resolve()
    root.mkdir(parents=True, exist_ok=False)
    # Capture model identity without hashing all 60 GiB for each job.
    model = Path(args.model).resolve()
    revision = (model / 'DOWNLOAD_REVISION.txt').read_text().strip()
    cfg = {'schema': 1, 'count': len(rows), 'chunk_size': args.chunk_size,
           'tasks': math.ceil(len(rows) / args.chunk_size), 'attempts': ATTEMPTS,
           'context': args.context, 'seconds_per_attempt': args.seconds_per_attempt,
           'seeds': [1764, 1849, 1936], 'temperature': 1.0, 'min_p': 0.02,
           'dtype': 'bfloat16', 'kv_cache_dtype': 'auto', 'gpu_memory_utilization': .95,
           'model_revision': revision, 'model_config_sha256': digest(model / 'config.json'),
           'core_sha256': digest(Path(__file__).with_name('solver_core.py')),
           'runner_sha256': digest(__file__), 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with (root / 'dataset.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    cfg['dataset_sha256'] = digest(root / 'dataset.jsonl')
    atomic_json(root / 'manifest.json', cfg)
    print('Prepared {} questions, {} attempts, {} tasks'.format(len(rows), len(rows)*3, cfg['tasks']))
    print('Experiment:', root)

def load_experiment(root):
    cfg = json.loads((root / 'manifest.json').read_text())
    if digest(root / 'dataset.jsonl') != cfg['dataset_sha256']:
        raise ValueError('Dataset changed; create a new experiment')
    return cfg, rows_from(root / 'dataset.jsonl')

def record_path(root, index, attempt):
    return root / 'records' / ('%06d-%d.json' % (index, attempt))

def read_record(root, index, attempt, row):
    p = record_path(root, index, attempt)
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    if r['row_index'] != index or r['attempt'] != attempt or r['problem_id'] != row['id'] or r['reference_answer'] != row['answer']:
        raise ValueError('Inconsistent checkpoint: ' + str(p))
    return r

class JobInterrupted(BaseException):
    pass

def run(args):
    import fcntl
    root = Path(args.experiment).resolve()
    cfg, rows = load_experiment(root)
    if digest(__file__) != cfg['runner_sha256'] or digest(Path(__file__).with_name('solver_core.py')) != cfg['core_sha256']:
        raise ValueError('Code differs from experiment manifest; use original code or a new experiment')
    if not 1 <= args.task <= cfg['tasks']:
        raise ValueError('Task outside dataset range')
    start = (args.task - 1) * cfg['chunk_size']
    stop = min(start + cfg['chunk_size'], len(rows))
    (root / 'locks').mkdir(exist_ok=True)
    lock = (root / 'locks' / ('task-%d.lock' % args.task)).open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('This task is already running')
    pending = [(i, a) for i in range(start, stop) for a in range(1, 4)
               if read_record(root, i, a, rows[i]) is None]
    if not pending:
        print('Task already complete; no model loading needed.'); return
    model = Path(args.model).resolve()
    if (model / 'DOWNLOAD_REVISION.txt').read_text().strip() != cfg['model_revision'] or digest(model / 'config.json') != cfg['model_config_sha256']:
        raise ValueError('Model identity mismatch')
    run_dir = root / 'job_logs' / ('task-%03d-%s-%d' % (args.task, time.strftime('%Y%m%d-%H%M%S'), os.getpid()))
    run_dir.mkdir(parents=True)
    os.chdir(run_dir)
    print('Logs:', run_dir, flush=True)
    import solver_core as core
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one scheduler-allocated GPU')
    if torch.cuda.get_device_capability(0) != (8, 0):
        raise RuntimeError('Baseline requires an A100-class compute capability 8.0 GPU')
    C = core.CFG
    C.model_path = str(model); C.context_tokens = cfg['context']
    C.workers = 1; C.batch_size = 1; C.attempts = 3; C.early_stop = 4
    C.dtype = cfg['dtype']; C.kv_cache_dtype = cfg['kv_cache_dtype']
    C.gpu_memory_utilization = cfg['gpu_memory_utilization']
    C.server_timeout = 2400; C.jupyter_timeout = 30; C.problem_count = len(rows)
    C.stream_interval = 1  # Responsive deadline checking; no change to prompts/sampling.
    # Allocate an available loopback port; binding happens shortly afterward.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    def interrupt(signum, frame):
        raise JobInterrupted('Job interrupted; current attempt left uncommitted')
    def deadline(signum, frame):
        raise core.AttemptDeadline('Per-attempt time limit reached')
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGALRM, deadline)
    solver = core.AIMO3Solver.__new__(core.AIMO3Solver)
    job_started = time.monotonic()
    kernel = None
    try:
        solver.__init__(C, port=port)
        solver.client = solver.client.with_options(max_retries=0, timeout=60)
        startup = time.monotonic() - job_started
        atomic_json(run_dir / 'startup.json', {'startup_seconds': startup,
                    'gpu': torch.cuda.get_device_name(0), 'job_id': os.getenv('JOB_ID'),
                    'python': sys.version, 'torch': torch.__version__,
                    'host': socket.gethostname()})
        for i, attempt in pending:
            if time.monotonic() - job_started + cfg['seconds_per_attempt'] + 120 > args.max_job_seconds:
                print('Stopping before walltime; resubmit task %d to continue.' % args.task, flush=True)
                return
            if solver.server_process.poll() is not None:
                raise RuntimeError('vLLM server exited; remaining attempts not marked failed')
            # Fresh tool state each attempt. Kernel startup is measured separately.
            kernel_started = time.monotonic()
            kernel = core.AIMO3Sandbox(timeout=C.jupyter_timeout)
            kernel_startup = time.monotonic() - kernel_started
            solver.sandbox_pool.put(kernel)
            begin = time.monotonic()
            print('Problem %s, attempt %d/3' % (rows[i]['id'], attempt), flush=True)
            signal.setitimer(signal.ITIMER_REAL, cfg['seconds_per_attempt'])
            try:
                result = solver._process_attempt(
                    rows[i]['problem'] + ' ' + C.preference_prompt,
                    C.system_prompt, attempt - 1, core.threading.Event(),
                    time.time() + cfg['seconds_per_attempt'])
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            elapsed = time.monotonic() - begin
            # A dead server is infrastructure failure: do not consume an attempt.
            if solver.server_process.poll() is not None:
                raise RuntimeError('Server died during attempt; resubmit to retry missing work')
            entropy = result['Entropy']
            rec = {'row_index': i, 'problem_id': rows[i]['id'], 'attempt': attempt,
                   'seed': cfg['seeds'][attempt-1], 'predicted_answer': result['Answer'],
                   'reference_answer': rows[i]['answer'],
                   'correct': result['Answer'] == rows[i]['answer'],
                   'status': result.get('Status', 'no_answer'), 'elapsed_seconds': elapsed,
                   'kernel_startup_seconds': kernel_startup,
                   'generated_tokens': result['Response Length'],
                   'python_calls': result['Python Calls'], 'python_errors': result['Python Errors'],
                   'entropy': entropy if math.isfinite(entropy) else None,
                   'error': result.get('Error'), 'job_id': os.getenv('JOB_ID')}
            atomic_json(record_path(root, i, attempt), rec)
            print('SAVED:', json.dumps(rec), flush=True)
            solver.sandbox_pool.get_nowait()
            kernel.close(); kernel = None
        print('Task %d complete' % args.task, flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if kernel is not None:
            kernel.close()
        solver.close()
        lock.close()

def export(args):
    root = Path(args.experiment).resolve()
    cfg, rows = load_experiment(root)
    attempts, summaries, missing_tasks = [], [], set()
    for i, row in enumerate(rows):
        records = [read_record(root, i, a, row) for a in range(1, 4)]
        found = [r for r in records if r is not None]
        attempts.extend(found)
        complete = len(found) == 3
        if not complete:
            missing_tasks.add(i // cfg['chunk_size'] + 1)
        weights = {}
        for r in found:
            ans = r['predicted_answer']
            if ans is not None:
                weights.setdefault(ans, 0.0)
                if r['entropy'] is not None:
                    weights[ans] += 1.0 / max(r['entropy'], 1e-9)
        voted = max(weights, key=weights.get) if weights and complete else None
        s = {'problem_id': row['id'], 'reference_answer': row['answer'],
             'complete': complete, 'completed_attempts': len(found),
             'correct_attempts': sum(r['correct'] for r in found),
             'wrong_answers': json.dumps([r['predicted_answer'] for r in found
                                          if r['predicted_answer'] is not None and not r['correct']]),
             'failed_attempts': sum(r['predicted_answer'] is None for r in found),
             'total_attempt_seconds': sum(r['elapsed_seconds'] for r in found),
             'voted_answer': voted, 'voted_correct': voted == row['answer'] if voted is not None else None}
        for a, r in enumerate(records, 1):
            s['attempt_%d_answer' % a] = r['predicted_answer'] if r else None
            s['attempt_%d_seconds' % a] = r['elapsed_seconds'] if r else None
            s['attempt_%d_status' % a] = r['status'] if r else 'missing'
        summaries.append(s)
    out = root / 'exports'; out.mkdir(exist_ok=True)
    for filename, data in [('baseline_attempts.csv', attempts), ('baseline_summary.csv', summaries)]:
        tmp = out / (filename + '.tmp')
        with tmp.open('w', newline='', encoding='utf-8-sig') as f:
            fields = list(data[0]) if data else ['problem_id', 'attempt']
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(data)
        os.replace(str(tmp), str(out / filename))
    (out / 'missing_tasks.txt').write_text(''.join('%d\n' % n for n in sorted(missing_tasks)))
    report = {'questions': len(rows), 'expected_attempts': len(rows)*3,
              'saved_attempts': len(attempts), 'complete_problems': sum(s['complete'] for s in summaries),
              'missing_tasks': sorted(missing_tasks)}
    atomic_json(out / 'completion.json', report)
    print(json.dumps(report, indent=2)); print('CSV files:', out)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--data', required=True)
    a.add_argument('--experiment', required=True); a.add_argument('--model', required=True)
    a.add_argument('--chunk-size', type=int, default=12)
    a.add_argument('--context', type=int, default=4096)
    a.add_argument('--seconds-per-attempt', type=int, default=300)
    a = sub.add_parser('run'); a.add_argument('--experiment', required=True)
    a.add_argument('--model', default='/models/gpt-oss-120b')
    a.add_argument('--task', type=int, required=True)
    a.add_argument('--max-job-seconds', type=int, default=13800)
    a = sub.add_parser('export'); a.add_argument('--experiment', required=True)
    args = p.parse_args()
    if args.command == 'prepare' and min(args.chunk_size, args.context, args.seconds_per_attempt) < 1:
        p.error('Numeric settings must be positive')
    globals()[args.command](args)

if __name__ == '__main__':
    main()
