for i in {42..42}
do
    echo "Running $i"
    python tot2_multi.py --dataset nyt --seed $i --suffix nyt_test --part test --model umgpt --do_paraphrase --test_k 5
done