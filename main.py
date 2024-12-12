import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from pipeline import train, test
from src.utils.seed import set_seed
from src.utils.options import Tee
from src.sam.segment_anything.build_sam import *

@hydra.main(
        config_path="config", 
        config_name="main"
        )
def main(
    cfg: DictConfig
    ) -> None:
    
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    log_path = f"{script_dir}/log/{cfg.memo}.txt"
    log_file = open(log_path, 'a')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    print(f"Log 기록을 위한 Memo: {cfg}")  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.model)

    set_seed(cfg.seed)
    data = pd.read_csv(cfg.meta_csv)
    train_data, tmp_data = train_test_split(
        data, 
        test_size=(1-cfg.train_size), 
        random_state=cfg.seed
        )
    valid_data, test_data = train_test_split(
        tmp_data, 
        test_size=(cfg.test_size/(1-cfg.train_size)), 
        random_state=cfg.seed
        )

    if cfg.task == "train":
        train(cfg, model, device, train_data, valid_data)
    elif cfg.task == "test":
        test(cfg, model, device, test_data)
    else:
        raise ValueError(f"Invalid task mode: {cfg.task}")
    
    sys.stdout = original_stdout
    log_file.close()


if __name__ == '__main__':

    main()