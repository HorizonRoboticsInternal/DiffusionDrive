
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

export NCCL_BLOCKING_WAIT=0

export HF_ENDPOINT=https://hf-mirror.com

# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
experiment_name=training_fiery_agent_flow_matching_no_anchor_normed_input
agent=fiery_agent_no_anchor


python ./navsim/planning/script/run_training.py \
        agent=$agent \
        experiment_name=$experiment_name \
        cache_path="/mnt/cwai/hpfs0/navsim/fiery_training_cache" \
        train_test_split=navtrain  \
        split=trainval \
        trainer.params.max_epochs=100 \
        use_cache_without_dataset=True  \
        force_cache_computation=False


        # use following setting to train Fiery+DiffusionDrive head
        # agent=fiery_agent \ fiery_agent_no_anchor
        # experiment_name=training_fiery_agent \ #training_fiery_agent_ddim_no_anchor \ training_fiery_agent_flow_matching_no_anchor
        # cache_path="/mnt/cwai/hpfs0/navsim/fiery_training_cache" \
        # training_fiery_agent_flow_matching_no_anchor_100steps


        # use following setting to train original DiffusionDrive
        # agent=diffusiondrive_agent \ diffusiondrive_agent_no_anchor
        # experiment_name=training_diffusiondrive_agent  \
        # cache_path="/mnt/cwai/hpfs0/navsim/training_cache" \

cd ${NAVSIM_EXP_ROOT}/${$experiment_name}
for file in epoch=*-step=*.ckpt; do
    epoch=$(echo $file | sed -n 's/.*epoch=\([0-9][0-9]\).*/\1/p')
    new_filename="epoch${epoch}.ckpt"
    mv "$file" "$new_filename"
done

# select latest checkpoint
CKPT=$(ls -t epoch*.ckpt | head -n 1)
echo "Using checkpoint: $CKPT"

cd -
python ./navsim/planning/script/run_pdm_score.py \
        train_test_split=navtest \
        agent=$agent \
        agent.checkpoint_path=$CKPT \
        worker.max_workers=64 \
        experiment_name=$experiment_name

