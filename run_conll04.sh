for i in {42..42}
do
    echo "Running $i"
    # python tot2_multi.py --dataset conll04 --seed $i --suffix umgpttesting2_async --part test --model umgpt --is_async --test_k -1
    python tot2_multi.py --dataset conll04 --seed $i --suffix conll04_test_noPrior_icl_5entities4priorpara41sentences --part test --model umgpt --do_paraphrase --test_k -1 --number_of_shots_for_entity_extraction 5 --number_of_shots_for_relation 41
done