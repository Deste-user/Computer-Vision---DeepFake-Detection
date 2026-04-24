#!/bin/bash



EXPERIMENT_VERSION='v1 v2 v3'

echo "=========================================="
echo "   CREAZIONE EMBEDDINGS PER ESPERIMENTI   "
echo "=========================================="

for EXPERIMENT in $EXPERIMENT_VERSION; do
    echo "   Creating embeddings for experiment $EXPERIMENT ..."
    python3 main.py --create_embeddings --experiment_version $EXPERIMENT
done



 