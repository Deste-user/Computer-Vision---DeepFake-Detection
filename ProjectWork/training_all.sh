
#!/bin/bash

# Training models first (Global CLS) for basic datasets (if available) with MLP and SVM
echo "=========================================="
echo "   Training MODELS (GLOBAL CLS)           "
echo "=========================================="

MODELS='mlp svm'
DATASETS='stylegan1 sdv1_4'

for MODEL in $MODELS; do
    for ds in $DATASETS; do
        if [ -d "dataset_embeddings/${ds}_CLS" ]; then
            echo "   Training model $MODEL for $ds (CLS) ..."
            python3 main.py --classificator_model $MODEL --device cuda --batch_size 32 --token_mode CLS --mode train --dataset $ds
        fi
    done
done



# Training models for ALL datasets and ALL token modes with linear
TOKEN_MODES='CLS PATCHES'
DATASETS='stylegan1 sdv1_4 stylegan3 styleganxl stylegan2 sdv2_1'

echo "=========================================="
echo "   Training MODELS FOR ALL DATASETS       "
echo "=========================================="

for MODE in $TOKEN_MODES; do
    for DATASET_TARGET in $DATASETS; do
        if [ ! -d "dataset_embeddings/${DATASET_TARGET}_${MODE}" ]; then
            echo "   Skipping dataset $DATASET_TARGET for mode $MODE (embeddings non trovati) ..."
            continue
        fi
        echo "   Training linear model with dataset $DATASET_TARGET for mode $MODE ..."
        python3 main.py --classificator_model linear --device cuda --batch_size 32 --token_mode $MODE --mode train --dataset $DATASET_TARGET
    done
done