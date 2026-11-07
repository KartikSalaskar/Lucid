# Lucid Trust Evaluation Plan

## Current status
- Prediction helpers are merged.
- Shared evaluation metric and setup checks are merged.
- All 18 evaluation metric tests passed.
- All 5,824 image paths exist.
- No overlapping filename groups were found across train, validation and test.
- Patient identity separation remains unverified because patient IDs are missing.

## Baseline agreement — pending
The following hashes identify Nikhil's current local files.
They are recorded for comparison and are NOT yet the agreed team baseline.

Model: models/classifier.pt
SHA256: ccf931e289ac37748699138c1c2bc6d4afffd36b6c825c53db4ef5ccda21b426

Splits: data/processed/splits.json
SHA256: 196dbf0407a6a3f6b61921ff5ffe25cfa4b4b2c3577cf0111ac0cf92d1fb3f49

Class order: NORMAL=0, PNEUMONIA=1

Before shared evaluation, record the team's agreed checkpoint and split hashes.
Calibration and OOD artifacts must match the checkpoint used for inference.
Ensemble members have separate checkpoints; record each member's hash.

## Shared error-detection evaluation
- Target: wrong=1 when the agreed baseline prediction is incorrect.
- Larger signal values must mean greater suspicion of an error.
- Metric: ROC-AUC with 500 bootstrap resamples, seed 42.
- Report the 95% percentile bootstrap interval.
- Report undefined AUC explicitly when all examples have the same target.
- Compare signals on the same examples in the same order.
- Include 1 minus maximum softmax probability and a seeded random baseline.

## Data-use rules
- Fit calibration parameters on validation data only.
- Fit OOD statistics on training data; select thresholds on validation data.
- Do not tune models, thresholds or artifact selection using test results.
- Keep existing splits unchanged while baseline agreement is pending.

## Ensemble handoff
- Disha: Seed 1 and Seed 2.
- Chithara: designated Seed 3 checkpoint.
- Share checkpoint files, training histories and SHA256 hashes.
- Preserve Disha's duplicate Seed 3 as a backup.
- Verify the designated Seed 3 before final ensemble evaluation.

## Outstanding work
- Agree on the shared baseline checkpoint.
- Verify artifact provenance and compatibility.
- Complete Chithara's Seed 3 handoff to Disha.
- Resolve outstanding calibration and OOD review findings.
- Run shared evaluation after the required inputs are verified.
