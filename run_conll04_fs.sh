# python tot2_fs.py --dataset scierc --seed 0 --suffix scierc_5shot_2 --part test --model umgpt --test_k 100 --relation_type_extraction

for i in {2..10}
do
    echo "Running $i"
    python tot2_fs.py --dataset conll04 --seed 0 --suffix conll04_5shot --part test --model umgpt --test_k -1 --relation_type_extraction --split $i
done