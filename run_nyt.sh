for i in {42..42}
do
    echo "Running $i"
    python main.py --dataset nyt --seed $i 
done