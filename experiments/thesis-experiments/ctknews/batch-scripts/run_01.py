import os

CONFIG = {
    "n_layers": [1, 2, 3, 4],
    "n_heads": [1, 2, 4],
    "hidden_dim": [256],
}

BATCH_SCRIPT = "/home/hlavava1/afc/experiments/thesis-experiments/ctknews/batch-scripts/01_model_hyperparameters.sh"

def main():
    for n_layers in CONFIG["n_layers"]:
        for n_heads in CONFIG["n_heads"]:
            for hidden_dim in CONFIG["hidden_dim"]:
                cmd = f"sbatch {BATCH_SCRIPT} {n_layers} {n_heads} {hidden_dim} local_plus_global"
                os.system(cmd)


if __name__ == "__main__":
    main()
