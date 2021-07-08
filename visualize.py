
import argparse
import os

import mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_cached_data_id", type=str, default="")
    args = parser.parse_args()

    mlflow_exp_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))
    upstream = os.path.join(
        "/tmp/mlruns",
        str(mlflow_exp_id),
        args.preprocess_cached_data_id,
        "artifacts/downstream"
    )
    