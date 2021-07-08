import argparse
from distutils.dir_util import copy_tree
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=str, default="./data/preprocess")
    args = parser.parse_args()

    UPSTREAM = Path(args.upstream)
    copy_tree(
        str(UPSTREAM / "fiftyone_db"),
        "/root/.fiftyone/var/lib/mongo"
    )

    import fiftyone as fo
    dataset = fo.load_dataset(
        'person-segm-train'
    )
    print(dataset.info['licenses'])
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == '__main__':
    main()
