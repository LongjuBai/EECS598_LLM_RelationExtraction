for i in {0..9}
do
    echo "Running $i"
    python main.py --dataset ade --split $i 
done