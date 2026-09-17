export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../eval_policy.py "$@" \
    --exp_name=block_stacking \
    --checkpoint_path=first_run \
