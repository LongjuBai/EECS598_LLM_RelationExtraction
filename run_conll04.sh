for i in {42..42}
do
    echo "Running $i"
    python tot2_multi.py --dataset conll04 --seed $i --suffix CONLL04_train --do_paraphrase --part train
done