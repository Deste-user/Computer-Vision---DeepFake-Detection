#!/bin/bash



echo "=========================================="
echo "   Scaricamento Dati dal Server           "
echo "=========================================="

echo "   Downloading dataset_embeddings ..."
scp -r mdestefano@targaryen.micc.unifi.it:/equilibrium/students/mdestefano/Computer-Vision---DeepFake-Detection/FinalReport/dataset_embeddings .

echo "   Downloading classificators ..."
scp -r mdestefano@targaryen.micc.unifi.it:/equilibrium/students/mdestefano/Computer-Vision---DeepFake-Detection/FinalReport/classificators .