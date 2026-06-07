#!/bin/bash



TOKEN_MODES='CLS PATCHES'

echo "=========================================="
echo "   CREAZIONE EMBEDDINGS PER TUTTI I MODI   "
echo "=========================================="

for MODE in $TOKEN_MODES; do
    echo "   Creating embeddings for token mode $MODE ..."
    python3 main.py --create_embeddings --token_mode $MODE
done



 