# Learner runs inside WSL2 (Ubuntu-24.04) for GPU access -- the actor
# (run_actor.sh) stays on native Windows for robot/camera/gripper/SpaceMouse
# hardware access, and connects to this over --ip=localhost (WSL2 forwards it
# automatically). See hil-serl/examples/experiments/flexiv_task/wrapper.py and
# the WSL2 setup notes from this session for why the split is done this way.
"/c/Windows/System32/wsl.exe" -d Ubuntu-24.04 -- bash -c "
source ~/hilserl-venv/bin/activate && \
cd /mnt/c/Users/gloria/Documents/robotscripts/hil-serl/examples/experiments/flexiv_task && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.75 && \
python ../../train_rlpd.py $* \
    --exp_name=flexiv_task \
    --checkpoint_path=third_run \
    --demo_path=../../demo_data/peg_insertion_no_chamfer_25.pkl \
    --learner
"
