TRAIN_TEST_SPLIT=navtrain_part_1
CACHE_PATH=/mnt/cwai/hpfs0/navsim/trainval_metric_cache_parts/part_1

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# 220
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain_part_1 cache.cache_path=/mnt/cwai/hpfs0/navsim/trainval_metric_cach_parts/part_1

e
# 62
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain_part_2 cache.cache_path=/mnt/cwai/hpfs0/navsim/trainval_metric_cache_parts/part_2

# 38
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain_part_3 cache.cache_path=/mnt/cwai/hpfs0/navsim/trainval_metric_cache_parts/part_3

# 28
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain_part_4 cache.cache_path=/mnt/cwai/hpfs0/navsim/trainval_metric_cache_parts/part_4


# 19
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain_part_5 cache.cache_path=/mnt/cwai/hpfs0/navsim/trainval_metric_cache_parts/part_5

# 67
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain_part_6 cache.cache_path=/mnt/cwai/hpfs0/navsim/trainval_metric_cache_parts/part_6

# 85
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain_part_7 cache.cache_path=/mnt/cwai/hpfs0/navsim/trainval_metric_cache_parts/part_7_v2

# 51
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain_missing_0 cache.cache_path=/mnt/cwai/hpfs0/navsim/navtrain_metric_cache

# 85
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain_missing_1 cache.cache_path=/mnt/cwai/hpfs0/navsim/navtrain_metric_cache

# 67
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain_missing_2 cache.cache_path=/mnt/cwai/hpfs0/navsim/navtrain_metric_cache

# 19
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain_missing_3 cache.cache_path=/mnt/cwai/hpfs0/navsim/navtrain_metric_cache

# 19
python navsim/planning/script/run_metric_caching.py train_test_split=navtrain_missing_4 cache.cache_path=/mnt/cwai/hpfs0/navsim/navtrain_metric_cache
