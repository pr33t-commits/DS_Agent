# Experiment artifacts

`runs/` holds timestamped agent runs, including input snapshots, manifests,
traces and reports. Existing runs were moved here from the checkout root and
the agent package. Historical manifest paths are preserved as recorded provenance.
New runs default to this location; `--output` still overrides it.

Future checkpoints and training metrics belong here in per-experiment folders.
Generated artifacts are ignored by version control.
