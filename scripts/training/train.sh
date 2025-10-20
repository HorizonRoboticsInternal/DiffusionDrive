# export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
python ./navsim/planning/script/run_training.py \
        agent=fiery_agent \
        experiment_name=test  \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
        train_test_split=navtrain  \
        split=trainval   \
        trainer.params.max_epochs=100 \
        use_cache_without_dataset=True  \
        force_cache_computation=False


        # use following setting to train Fiery+DiffusionDrive head
        # agent=fiery_agent \
        # experiment_name=training_fiery_agent  \
        # cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \


        # use following setting to train original DiffusionDrive
        # agent=diffusiondrive_agent \
        # experiment_name=training_diffusiondrive_agent  \
        # cache_path="${NAVSIM_EXP_ROOT}/training_cache_dd/" \
