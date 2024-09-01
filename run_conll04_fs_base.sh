for i in {1..1}
do
    echo "Running $i"
    python tot2_fs_base.py --dataset conll04 --suffix conll04_5shot_base --test_k -1 --split $i --n_shot 5
done