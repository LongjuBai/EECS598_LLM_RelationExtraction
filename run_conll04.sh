for i in {42..42}
do
    echo "Running $i"
    # python tot2_multi.py --dataset conll04 --seed $i --suffix umgpttesting2_async --part test --model umgpt --is_async --test_k -1
    python tot2_multi.py --dataset conll04 --seed $i --suffix umgpttesting2 --part test --model umgpt --test_k -1
done