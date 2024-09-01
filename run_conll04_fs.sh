# python tot2_fs.py --dataset scierc --seed 0 --suffix scierc_5shot_2 --model umgpt --test_k 100 --relation_type_extraction

# for i in {1..1}
# do
#     echo "Running $i"
#     python tot2_fs.py --dataset conll04 --suffix conll04_5shot_compact_nolower --test_k -1 --split $i --compact --n_shot 5
# done

# for i in {1..1}
# do
#     echo "Running $i"
#     python tot2_fs.py --dataset scierc --suffix scierc_5shot_test --test_k -1 --split $i --compact --n_shot 5
# done

for i in {1..1}
do
    echo "Running $i"
    python tot2_fs.py --dataset nyt --suffix nyt_5shot_test_neg3 --test_k 100 --split $i --compact --n_shot 5 --relation_type_extraction
done