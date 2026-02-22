#!/bin/bash



# Modelli da addestrare
MODELS="mlp svm"


SET='stylegan1 stablediffusion'


echo "=========================================="
echo "AVVIO CALCOLO METRICHE DEI RISULTATI"
echo "=========================================="

for model in $MODELS; do
    echo " Model: $model "
    for set in $SET; do

        echo "   [1/3] Calculate metrics of the $set Dataset with No Crossvalidation."
        python3 main.py --metrics --dataset $set --classificator_mode $model

        echo "   [1/3] Calculate metrics of the $set Dataset with  Crossvalidation."
        python3 main.py --metrics --dataset $set --cross_validate --classificator_mode $model
        
    done    
done

echo "All result are in the directory metrics_results"