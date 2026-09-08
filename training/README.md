# Training layer

Training is planned, not implemented. Add `sft.py`, `dpo.py`, `grpo.py` and
`trainer_utils.py` as the algorithms are implemented. Use PyTorch/Transformers
for model access, Accelerate for device management, PEFT for adapters and TRL
as a reference implementation. Keep these dependencies optional for agent users.

Consume reviewed datasets from the data layer, save checkpoints and metrics
under experiments, and use the inference generation contract for rollouts.
Do not automatically train on unreviewed agent traces.
