# Resumable Argon math baseline

Based on the user's tested `aimo_argon.py`, itself adapted from the supplied Kaggle notebook. Preserve upstream attribution when publishing. No weights or container are included.

## Evaluation protocol

Three sequential attempts per problem, seeds 1764 / 1849 / 1936 (same seed scheme as the supplied solver), temperature 1.0, min_p .02, HIGH Harmony reasoning, Python tools, 4096-token context, BF16 KV cache, eager execution, one A100 80GB, 300 seconds per attempt. No agreement-based early stopping. All three attempts are scheduled even if the first answer is correct. Reference answers are used by scoring, never inserted into the prompt.

This is a compute-constrained baseline, not the full competition configuration. The supplied prompts, answer extraction and inverse top-logprob-entropy voting are retained. In particular, the original extractor can accept a boxed answer in streamed text before the final channel. Entropy uses only returned top-5 token probabilities; it is not calibrated correctness confidence. Each attempt gets a fresh Jupyter kernel. Stream interval is reduced from 200 to 1 for responsive deadlines. Python tool execution timeout stays 30 seconds. Generated code runs under your account; this is not a security sandbox, and mounted input files are accessible to tools.

Wall-clock time is measured from just before generation to its return, including tool calls, but excluding model startup, kernel startup, and shutdown. Kernel startup and model startup are recorded separately. SIGALRM interrupts ordinary Python/network waits at the attempt deadline; OS/native-code stalls can delay delivery, so scheduler termination and resume remain the fallback. Abruptly interrupted attempts have no committed record and are rerun. Completed timeouts/errors count as failed attempts. A detected dead server aborts the task without consuming that attempt.

## Files

- `aimo_baseline.py`: standard-library prepare/run/export CLI (Python >=3.7; use your container Python).
- `solver_core.py`: model/server/tool code derived from your uploaded working script.
- `run_baseline.sh`: SGE array job, four-hour allocation, up to 12 problems/task.
- `test_baseline.py`: CPU-only checkpoint/export tests with synthetic data; does not test GPU inference.

## Install on Argon

Extract this ZIP into `~/aimo3`, producing `~/aimo3/aimo_baseline_bundle/`. No container rebuild required. Keep the old `aimo_argon.py`.

Run from an Argon login session:

```bash
cd ~/aimo3
mkdir -p logs baselines
apptainer exec --cleanenv --pwd / --bind "$HOME/aimo3:/work" \
  containers/aimo3-env-v2.sif /opt/aimo-venv/bin/python \
  /work/aimo_baseline_bundle/aimo_baseline.py prepare \
  --data /work/data/problems.jsonl \
  --model /work/models/gpt-oss-120b \
  --experiment /work/baselines/a100-4k-3tries-v1
```

Prepare once. It snapshots all input questions, freezes settings, hashes dataset and code, and records the model revision/config hash. It refuses to overwrite an experiment. It does not hash all model weights. Keep weights unchanged. Any deliberate protocol/code/dataset change requires a new experiment name (and corresponding job path). Code hashes are checked on every task.

## Pilot, then full array

Run only the first task initially (12 questions, 36 attempts):

```bash
qsub -t 1 -o "$HOME/aimo3/logs" ~/aimo3/aimo_baseline_bundle/run_baseline.sh
qstat -u "$USER"
```

A `qw` job is queued; `r` is running. You can disconnect after submission. The four-hour request includes loading, kernel startup, solving and shutdown. Each task stops starting attempts at a conservative 230-minute job budget. At the worst-case deadline, 12 problems require 3 hours of generation, leaving about an hour for startup and overhead. If insufficient, resubmit that task.

Inspect `~/aimo3/baselines/a100-4k-3tries-v1/job_logs/` for vLLM logs and startup.json. Scheduler output is in `~/aimo3/logs/`. Attempt records are in the experiment's `records/` directory. Each task holds an advisory filesystem lock to reject duplicate concurrent writers. Different tasks write different checkpoint files. If filesystem locking is unsupported, the run fails rather than bypassing locking.

After pilot completion, export as below and inspect statuses/outputs. For exactly 2580 questions and chunk size 12, there are 215 tasks. Submit the full array:

```bash
qsub -t 1-215 -tc 1 -o "$HOME/aimo3/logs" ~/aimo3/aimo_baseline_bundle/run_baseline.sh
```

`-tc 1` caps this array at one running task (one GPU). It does not limit other arrays you submit; avoid overlapping full submissions. Task 1 skips already-completed attempts. Use the task count printed by prepare if your dataset size differs. Campus scheduling/queue access governs when tasks run. The supplied one-hour interactive tests do not establish all batch queue limits; report any scheduler rejection rather than altering resource assignments manually.

## Export / completeness / resume

Safe on the login node using container Python; no GPU needed. Run one exporter at a time.

```bash
cd ~/aimo3
apptainer exec --cleanenv --pwd / --bind "$HOME/aimo3:/work" \
  containers/aimo3-env-v2.sif /opt/aimo-venv/bin/python \
  /work/aimo_baseline_bundle/aimo_baseline.py export \
  --experiment /work/baselines/a100-4k-3tries-v1
```

Outputs in `exports/`:

- `baseline_attempts.csv`: one row per saved attempt; elapsed seconds, answer, reference, correctness, status, seed, tokens, Python counts, entropy, error, job ID.
- `baseline_summary.csv`: one row per problem; answers/statuses/times for all three attempts, number correct, wrong numerical answers (duplicates retained), no-answer count, total attempt seconds, weighted-vote answer/correctness.
- `completion.json`: expected/saved attempts and complete problem counts.
- `missing_tasks.txt`: task indices still needing work.

Incomplete problems have `complete=False`; correct_attempts is the count so far, not a final out-of-three score. Missing answers remain blank, never zero. A legitimate answer of zero is valid. Weighted-vote fields remain blank until all three attempts are recorded. Export is a snapshot if jobs are still running; export again after completion.

After all scheduled tasks have stopped, resubmit a missing task, e.g. 7:

```bash
qsub -t 7 -o "$HOME/aimo3/logs" ~/aimo3/aimo_baseline_bundle/run_baseline.sh
```

Or resubmit the full capped array once; completed tasks exit before loading the model. Completed timeout/error attempts are not rerun automatically. Review error prevalence before accepting a final accuracy result. Do not selectively rerun incorrect answers as part of this baseline.

Full completion for 2580 questions means 2580 complete problems, 7740 saved attempts and an empty missing_tasks list.

## Download to Windows

In local PowerShell, not inside SSH:

```powershell
scp elucas1@argon.hpc.uiowa.edu:~/aimo3/baselines/a100-4k-3tries-v1/exports/baseline_summary.csv "$env:USERPROFILE\Downloads\baseline_summary.csv"
scp elucas1@argon.hpc.uiowa.edu:~/aimo3/baselines/a100-4k-3tries-v1/exports/baseline_attempts.csv "$env:USERPROFILE\Downloads\baseline_attempts.csv"
```

Use `scp -P 40` if needed. CSVs use UTF-8 BOM for Excel. Keep manifest.json with published results for provenance; keep a copy of the source code and container package lock as well.

## Verification limits

Static syntax checks and CPU-only synthetic tests cover checkpoint atomicity, resume lookup, missing outcomes, duplicate IDs, zero-valued answers, summary aggregation and data-change detection. GPU inference and SGE execution must be validated by the first Argon task. The script cannot submit to or observe your Argon account from this chat.
