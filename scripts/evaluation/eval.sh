export CUDA_VISIBLE_DEVICES=0
# export CKPT=/mnt/nas25/wenxin.shao/workspace/DiffusionDrive/download/diffusiondrive_navsim_88p1_PDMS
# export CKPT=/mnt/nas25/wenxin.shao/workspace/DiffusionDrive/exp/training_diffusiondrive_agent/2025.07.24.10.08.48/lightning_logs/version_0/checkpoints/epoch_99-step_16700.ckpt
export CKPT=/mnt/nas25/wenxin.shao/workspace/DiffusionDrive/exp/training_fiery_agent/2025.08.13.21.35.25/lightning_logs/version_0/checkpoints/epoch_99-step_16700.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=fiery_agent \
        agent.checkpoint_path=$CKPT \
        worker.max_workers=1 \
        experiment_name=test_fix
        # experiment_name=diffusiondrive_agent_eval_ori
        # agent=diffusiondrive_agent \