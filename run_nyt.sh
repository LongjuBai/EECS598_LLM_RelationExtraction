for i in {42..42}
do
    echo "Running $i"
    python tot2_multi.py --dataset nyt --seed $i --suffix 'umgpt-testing' --model 'umgpt' --test_k 10
done