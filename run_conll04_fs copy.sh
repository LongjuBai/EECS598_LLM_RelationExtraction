# python tot2_fs.py --dataset scierc --seed 0 --suffix scierc_5shot_2 --part test --model umgpt --test_k 100 --relation_type_extraction

# for i in {1..5}
# do
#     echo "Running $i"
#     python tot2_fs.py --dataset conll04 --seed 0 --suffix conll04_1shot --part test --model umgpt --test_k -1 --relation_type_extraction --split $i --n_shot 1
# done

for i in {6..10}
do
    echo "Running $i"
    python tot2_fs.py --dataset conll04 --seed 0 --suffix conll04_10shot --part test --model umgpt --test_k -1 --relation_type_extraction --split $i --n_shot 10
done