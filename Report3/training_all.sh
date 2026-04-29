EXPERIMENT_VERSION='v3'
DATASET_FAKE_VERSION='fake_1 fake_2'

echo "=========================================="
echo "   Training MODELS FOR EXPERIMENTS   "
echo "=========================================="

for EXPERIMENT in $EXPERIMENT_VERSION; do
    for DATASET_FAKE in $DATASET_FAKE_VERSION; do
        echo "   Training model with dataset $DATASET_FAKE for experiment $EXPERIMENT ..."
        python3 main.py --classificator_model linear --device cuda --batch_size 32 --experiment_version $EXPERIMENT --mode train --dataset $DATASET_FAKE
done
done
