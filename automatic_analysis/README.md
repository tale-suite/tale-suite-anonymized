# Failure Detector and Step-Budget Analysis Code

This directory contains the analysis code used for the TALES failure-detector
and step-budget experiments. It intentionally does not include generated
detection outputs or transcript data.

## Directory Layout

- `rule_learning/detect_rule_learning.py`: detects Rule Learning failures from
  transcript JSON by flagging repeated errored actions and repeated
  action-response cycles.
- `situated_awareness/detect_failures.py`: runs the Situated Awareness detector
  over transcripts after replaying them through the underlying environments.
- `situated_awareness/replay_*.py`: framework-specific replay helpers used by
  the Situated Awareness detector to recover ground-truth state.
- `step_budget/cap_appropriateness.py`: analyzes 100-to-400-step runs to test
  whether extra budget produces additional score, and where.
- `step_budget/cap50_predict_100.py`: evaluates whether 50-step behavior
  predicts 100-step gains for mid-tier models.
- `step_budget/trajectory_extension_analysis.py`: ranks models for beyond-100
  evaluation using early trajectory signals.
- `step_budget/detect_brute_force.py`: computes parser-failure, repetition,
  stagnation, and late-gain signals used to characterize brute-force behavior.
- `step_budget/gpt54_actual_gains.py`: summarizes score gains in 100-step
  buckets for 400-step runs.

## Expected Inputs

The scripts default to paths relative to the repository root:

- `transcripts_slim/<framework>/*.json`
- `transcripts/<framework>/*.json` as a fallback for
  `trajectory_extension_analysis.py`
- `error_messages/master/<framework>/<task>.txt` for the Rule Learning detector

Most scripts accept command-line path overrides. For example:

```bash
python analysis/rule_learning/detect_rule_learning.py \
  --transcripts-dir /path/to/transcripts_slim \
  --error-dir /path/to/error_messages/master \
  --output-dir /path/to/rule_learning_outputs

python analysis/situated_awareness/detect_failures.py \
  --transcripts-dir /path/to/transcripts_slim \
  --output-dir /path/to/situated_awareness_outputs
```

The Situated Awareness detector requires the TALES environments and framework
dependencies to be installed, since it replays transcripts to recover
ground-truth state.
