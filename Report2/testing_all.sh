#!/bin/bash



SET='stylegan1 stablediffusion'


echo "=========================================="
echo "AVVIO ESPERIMENTO COMPLETO DEEPFAKE DETECTION"
echo "=========================================="


for set in $SET; do
    echo "   Testing with trained model on the dataset $set ..."
    python3 main_v2.py --dataset $set --mode test 

    echo " Drawing the ACC graph of models trained on dataset $set ..."
    python3 main_v2.py --dataset $set --graphs 
done    
