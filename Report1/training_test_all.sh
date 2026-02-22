#!/bin/bash



# Modelli da addestrare
MODELS="mlp svm"


SET='stylegan1 stablediffusion'


echo "=========================================="
echo "AVVIO ESPERIMENTO COMPLETO DEEPFAKE DETECTION"
echo "=========================================="

for model in $MODELS; do
    echo " Model: $model "
    for set in $SET; do

        echo "   [1/3] Training with the $set Dataset"
        python3 main.py --classificator_model $model --dataset $set --mode train
        
        # 2. TEST (Same Domain - StyleGAN with no crossvalidation)
        echo "   [2/3] Testing with trained model on the dataset $set ..."
        python3 main.py --classificator_model $model --dataset $set --mode test 
        
        # 3. TEST (Generalization - Stable Diffusion)
        echo " Testing with Cross validation ..."
        python3 main.py --classificator_model $model --dataset $set --mode test --cross_validate
    done    
done


#echo ""
#echo "=========================================="
#echo "GENERAZIONE REPORT FINALE"
#echo "=========================================="

# 4. METRICHE E PLOT
# Questo script deve leggere i file salvati nei passaggi prima e creare l'Excel/Grafici
#python create_report.py 

#echo "FATTO! Controlla i risultati."