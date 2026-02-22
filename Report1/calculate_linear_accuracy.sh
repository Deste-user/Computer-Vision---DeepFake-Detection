SET='stylegan1 stablediffusion'

echo " Model: linear"
for set in $SET; do

    echo "   [1/3] Training with the $set Dataset"
    python3 main.py --classificator_model linear --dataset $set --mode train --num_epochs 20
    
    echo " Testing with Cross validation ..."
    python3 main.py --classificator_model linear --dataset $set --mode test --cross_validate
done