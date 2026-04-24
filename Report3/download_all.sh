#!/bin/bash



EXPERIMENT_VERSION='v1 v2 v3'

echo "=========================================="
echo "   Scarico EMBEDDINGS PER ESPERIMENTI   "
echo "=========================================="

for EXPERIMENT in $EXPERIMENT_VERSION; do
    echo "   Downloading embeddings for experiment $EXPERIMENT ..."
    scp -r mdestefano@targaryen.micc.unifi.it:/equilibrium/students/mdestefano/Computer-Vision---DeepFake-Detection/Report3/dataset_embeddings_$EXPERIMENT "C:\Users\deste\Projects\Computer Vision Project\Report3"
done