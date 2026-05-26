
#!/bin/bash

# Training models first (Global CLS) for basic datasets (if available)
echo "=========================================="
echo "   Training MODELS (GLOBAL CLS)           "
echo "=========================================="

MODELS='linear mlp svm'
for MODEL in $MODELS; do
    if [ -d "dataset_embeddings/stylegan1_CLS" ]; then
        echo "   Training model $MODEL for stylegan1 (CLS) ..."
        python3 main.py --classificator_model $MODEL --device cuda --batch_size 32 --token_mode CLS --mode train --dataset stylegan1
    fi
done

# Training models for ALL datasets and ALL token modes
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

