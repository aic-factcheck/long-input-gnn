#!/bin/bash
#SBATCH --partition amdgpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 16G
#SBATCH --gres gpu:1
#SBATCH --time 1-0:00:00
#SBATCH --job-name multisource-00-bert-baseline-%J
#SBATCH --output logs/multisource-00-bert-baseline-%J.log

# get tunneling info
# XDG_RUNTIME_DIR=""
# port=$(shuf -i8000-9999 -n1)
# node=$(hostname -s)
# user=$(whoami)

# print tunneling instructions jupyter-log
# echo -e "
# MacOS or linux terminal command to create your ssh tunnel:
# ssh -N -L ${port}:${node}:${port} ${user}@login.rci.cvut.cz
# 
# Here is the MobaXterm info:
# 
# Forwarded port:same as remote port
# Remote server: ${node}
# Remote port: ${port}
# SSH server: login.rci.cvut.cz
# SSH login: $user
# SSH port: 22
# 
# Use a Browser on your local machine to go to:
# localhost:${port}  (prefix w/ https:// if using password)
# "

DATASET_PATH=${1:-"/home/hlavava1/afc/datasets/thesis-datasets/multisource/multisource-40k-similar"}
MODEL_NAME=${2:-"bert-base-cased"}
MIN_EPOCHS=5
MAX_EPOCHS=5
SEED=123

echo "DATASET: $DATASET_PATH"
echo "MODEL NAME: $MODEL_NAME"
echo "SEED: $SEED"

source /home/hlavava1/scripts/load_env.sh
cd /home/hlavava1/afc

python experiments/thesis-experiments/multisource/00_bert_baseline.py \
    --dataset-path=$DATASET_PATH \
    --model-name=$MODEL_NAME \
    --min-epochs=$MIN_EPOCHS \
    --max-epochs=$MAX_EPOCHS \
    --seed=$SEED \
    --lr=0.00002 \
    --weight-decay=0.01 \
    --quiet


