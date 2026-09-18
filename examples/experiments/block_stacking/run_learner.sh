# Runs natively on the Ubuntu desktop (unlike flexiv_task/run_learner.sh,
# which wraps this in a WSL2 call for the Windows dev laptop) -- both actor
# and learner run on the same Linux box here, connected over --ip=localhost.
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.75 && \
python ../../train_rlpd.py "$@" \
    --exp_name=block_stacking \
    --checkpoint_path=pos1_first_run \
    --demo_path=../../demo_data/block_stacking_pos1.pkl \
    --learner
