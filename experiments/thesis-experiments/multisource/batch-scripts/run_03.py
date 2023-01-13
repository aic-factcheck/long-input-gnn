import os

CONFIG = {
    "window_size": [1, 2, 3, 4],
    "connectivity": ["full", "local_plus_global", "local", "none"],
    "hidden_dim": [256]
}

BATCH_SCRIPT = "/home/hlavava1/afc/experiments/thesis-experiments/multisource/batch-scripts/03_ablation_sliding_window.sh"

def main():
    for window_size in CONFIG["window_size"]:
        for connectivity in CONFIG["connectivity"]:
            # This configuration has been already computed as part of Experiment 01
            # if window_size == 1 and connectivity == "full":
            #     continue

            cmd = f"sbatch {BATCH_SCRIPT} {window_size} {connectivity} 256"
            os.system(cmd)


if __name__ == "__main__":
    main()
