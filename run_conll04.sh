for i in {42..42}
do
    echo "Running $i"
    # python tot2_multi.py --dataset conll04 --seed $i --suffix umgpttesting2_async --part test --model umgpt --is_async --test_k -1
    python tot2_multi.py --dataset conll04 --seed $i --suffix conll04_test_noPrior_icl_10entities4priorpara10sentences --part test --model umgpt --do_paraphrase --test_k 20 --use_icl --number_of_shots_for_entity_extraction 3 --number_of_shots_for_relation 3
done