#!/bin/bash
#SBATCH --partition amdgpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 16G
#SBATCH --gres gpu:1
#SBATCH --time 1-0:00:00
#SBATCH --job-name ctknews-01-model-hyperparams-%J
#SBATCH --output logs/ctknews-01-model-hyperparams-%J.log

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

MIN_EPOCHS=50
MAX_EPOCHS=50
N_LAYERS=${1:-"1"}
N_HEADS=${2:-"1"}
HIDDEN_DIM=${3:-"256"}
CONNECTIVITY=${4:-"full"}
MODEL_NAME=${5:-"fav-kky/FERNET-C5"}
EVIDENCE_COUNT=${6:-"10"}
SEED=123

echo "MODEL NAME: $MODEL_NAME"
echo "SEED: $SEED"
echo ""
echo "N_LAYERS: $N_LAYERS"
echo "N_HEADS: $N_HEADS"
echo "HIDDEN_DIM: $HIDDEN_DIM"
echo "CONNECTIVITY: $CONNECTIVITY"

source /home/hlavava1/scripts/load_env.sh
cd /home/hlavava1/afc

python experiments/thesis-experiments/ctknews/01_model_hyperparameters.py \
    --model-name=$MODEL_NAME \
    --min-epochs=$MIN_EPOCHS \
    --max-epochs=$MAX_EPOCHS \
    --seed=$SEED \
    --n-layers=$N_LAYERS \
    --n-heads=$N_HEADS \
    --hidden-dim=$HIDDEN_DIM \
    --connectivity=$CONNECTIVITY \
    --lr=0.00002 \
    --weight-decay=0.01 \
    --batch-size=1 \
    --accumulate-grad-batches=32 \
    --evidence-count=$EVIDENCE_COUNT \
    --quiet


