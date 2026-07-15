# Continuous Learning Loop

Once the SDK is live, this is how each release gets more accurate than the last.

```
Customer images (opt-in only — see note below)
        │
        ▼
Low-confidence / ASK_RETAKE / REVIEW_REQUIRED cases logged
        │
        ▼
Human review (confirm correct digit string)
        │
        ▼
Added to training set (dataset/real/, versioned)
        │
        ▼
Retrain detector and/or recognizer
        │
        ▼
Evaluate on a held-out real-world test set (never train on this set)
        │
        ▼
If accuracy improves: export new ONNX, version-bump, ship
```

## Practical notes

- **Only log images you have explicit permission to retain.** Treat odometer
  photos as potentially containing identifying information (license plates,
  reflections, interiors). Get opt-in consent and document retention/deletion
  policy before wiring this up in production — this is a legal/privacy
  decision your team should make deliberately, not a default this scaffold
  makes for you.
- **Prioritize logging exactly the cases the SDK was least sure about**
  (`ConfidenceAction.RETRY_PREPROCESSING`, `ASK_RETAKE`, and business-rule
  `REVIEW_REQUIRED` cases) — these are far more valuable per-image for
  retraining than routine high-confidence reads.
- **Keep a fixed, never-retrained-on held-out test set** from the start so
  each new model version can be compared apples-to-apples across releases,
  not just against its own training distribution.
- **Version your models** (`detector_v1.onnx`, `detector_v2.onnx`, ...) and
  keep the previous version deployable — treat model updates like any other
  production release with rollback capability.
