export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=block_stacking \
    --checkpoint_path=first_run \
    --actor \
    --save_video \
    --human_classifier \
