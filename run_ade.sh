for i in {0..0}
do
    echo "Running $i"
    python tot2_multi.py --dataset ade --split $i --suffix umgpttesting --model umgpt --do_paraphrase --test_k -1
done