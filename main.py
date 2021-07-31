import argparse
import os

import mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_config", type=str, default="./config/preprocess/coco2014.yaml")
    parser.add_argument("--preprocess_cached_data_id", type=str, default="")
    parser.add_argument("--train_config", type=str, default="./config/train/resnet18.yaml")
    args = parser.parse_args()

    mlflow_exp_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))
    preprocess_cached_data_id = args.preprocess_cached_data_id
    preprocess_downstream = "/opt/data/preprocess"
    train_downstream = "/opt/data/model"
    
    with mlflow.start_run() as r:
        if preprocess_cached_data_id:
            train_upstream = os.path.join(
                "/tmp/mlruns",
                str(mlflow_exp_id),
                preprocess_cached_data_id,
                "artifacts/downstream"
            )
        else:
            preprocess_run = mlflow.run(
                uri="./preprocess",
                entry_point="preprocess",
                backend="local",
                parameters={
                    "config": args.preprocess_config,
                    "downstream": preprocess_downstream,
                }
            )
            train_upstream = os.path.join(
                "/tmp/mlruns",
                str(mlflow_exp_id),
                preprocess_run.run_id,
                "artifacts/downstream"
            )
        train_run = mlflow.run(
            uri="./train",
            entry_point="train",
            backend="local",
            docker_args={
                "gpus": "all"
            },
            parameters={
                "config": args.train_config,
                "upstream": train_upstream,
                "downstream": train_downstream
            }
        )

if __name__ == '__main__':
    main()