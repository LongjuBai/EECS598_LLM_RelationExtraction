# for i in {1..1}
# do
#     echo "Running $i"
#     python tot2_fs.py --dataset conll04 --suffix conll04_5shot_compact_nolower --test_k -1 --split $i --compact --n_shot 5
# done

for i in {1..1}
do
    echo "Running $i"
    python tot2_fs.py --dataset scierc1 --suffix scierc1_5shot_test --test_k -1 --split $i --compact --n_shot 5
done

# for i in {1..1}
# do
#     echo "Running $i"
#     python tot2_fs.py --dataset nyt --suffix nyt_5shot_test_tt --test_k 10 --split $i --compact --n_shot 5 --neg 0
# done