for i in {42..42}
do
    echo "Running $i"
    python main.py --dataset conll04 --seed $i 
done