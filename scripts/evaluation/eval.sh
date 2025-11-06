export CUDA_VISIBLE_DEVICES=0
# export CKPT=/mnt/nas25/wenxin.shao/workspace/DiffusionDrive/download/diffusiondrive_navsim_88p1_PDMS
# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/download/diffusiondrive_navsim_88p1_PDMS
# export CKPT=/mnt/nas25/wenxin.shao/workspace/DiffusionDrive/exp/training_diffusiondrive_agent/2025.07.24.10.08.48/lightning_logs/version_0/checkpoints/epoch_99-step_16700.ckpt
# export CKPT=/mnt/nas25/wenxin.shao/workspace/DiffusionDrive/exp/training_fiery_agent/2025.08.13.21.35.25/lightning_logs/version_0/checkpoints/epoch_99-step_16700.ckpt
# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_diffusiondrive_agent/exp001/2025.10.23.09.06.43/lightning_logs/version_0/checkpoints/epoch-99-step_16700.ckpt
# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_fiery_agent/exp001/2025.10.23.14.02.41/lightning_logs/version_0/checkpoints/epoch-99-step-22200.ckpt
# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_fiery_agent_flow_matching_no_anchor_normed_input/2025.10.25.06.50.38/lightning_logs/version_0/checkpoints/epoch99-step22200.ckpt
# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_fiery_agent_ddim_no_anchor_normed_input/2025.10.25.08.55.39/lightning_logs/version_0/checkpoints/epoch99-step22200.ckpt
# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_transfuer_flow_matching_no_anchor_discrete_50steps_ema/2025.10.27.13.50.03/lightning_logs/version_0/checkpoints/epoch99-step16700.ckpt
# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_transfuer_flow_matching_no_anchor_discrete_20steps/2025.10.27.12.56.17/lightning_logs/version_0/checkpoints/epoch99-step16700.ckpt

# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_fiery_agent_flow_matching_no_anchor_continuous/2025.10.27.00.36.24/lightning_logs/version_0/checkpoints/epoch99-step22200.ckpt
# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_fiery_agent_flow_matching_no_anchor_100steps/2025.10.27.00.30.27/lightning_logs/version_0/checkpoints/epoch99-step22200.ckpt
# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_fiery_agent_flow_matching_bev_crossAttn/2025.10.28.03.31.18/lightning_logs/version_0/checkpoints/epoch99-step22200.ckpt

# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_fiery_agent_flow_matching_efficient_b0/2025.10.28.12.36.24/lightning_logs/version_0/checkpoints/epoch99-step16700.ckpt
# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_fiery_agent_flow_matching_ema/2025.10.28.12.24.14/lightning_logs/version_0/checkpoints/epoch98-step21978.ckpt
# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_fiery_agent_flow_matching_bev_pool/2025.10.28.10.12.38/lightning_logs/version_0/checkpoints/epoch99-step22200.ckpt

# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_fiery_agent_flow_matching_ema/2025.11.04.06.34.47/lightning_logs/version_0/checkpoints/last.ckpt
# export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_fiery_agent_flow_matching_ema_nft/2025.11.05.08.35.09/lightning_logs/version_0/checkpoints/last.ckpt
export CKPT=/mnt/cwai/hpfs0/kun.li/DiffusionDrive/exp/training_fiery_agent_flow_matching_no_anchor_normed_input/2025.11.05.15.10.11/lightning_logs/version_0/checkpoints/last.ckpt


python ./navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=fiery_agent_no_anchor \
        agent.checkpoint_path=$CKPT \
        worker.max_workers=64 \
        experiment_name=training_fiery_agent_flow_matching_ema
        # experiment_name=diffusiondrive_agent_eval_ori \ fiery_agent_flow_matching_no_anchor_normed_input
        # agent=diffusiondrive_agent \ fiery_agent \ diffusiondrive_agent_no_anchor