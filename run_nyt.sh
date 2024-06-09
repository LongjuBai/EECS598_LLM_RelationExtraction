for i in {42..42}
do
    echo "Running $i"
<<<<<<< Updated upstream
    python tot2_multi.py --dataset nyt --seed $i --suffix 'umgpt-testing' --model 'umgpt' --test_k 10
=======
    python main.py --dataset nyt --seed $i --test_k 100
>>>>>>> Stashed changes
done