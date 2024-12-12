import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.sam_dataset import SamDataset
from src.sam_train import sam_train
from src.sam_test import sam_test

def train(
        cfg: DictConfig,
        model: nn.Module,
        device: torch.device,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame
        ) -> None:
    
    train_dataset = SamDataset(cfg, model, train_data)
    valid_dataset = SamDataset(cfg, model, valid_data)

    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        shuffle=True
        )
    valid_dataloader = DataLoader(
        dataset=valid_dataset, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        shuffle=False
        )

    optimizer = instantiate(cfg.optimizer, model.mask_decoder.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer)
    sam_train(cfg, model, device, optimizer, scheduler, train_dataloader, valid_dataloader)

def test(
        cfg: DictConfig, 
        model: nn.Module, 
        device: torch.device,
        test_data: pd.DataFrame,
        ) -> None:
    
    test_dataset = SamDataset(cfg, model, test_data)
    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        shuffle=False
        )

    model.load_state_dict(torch.load(cfg.weight, weights_only=True))
    sam_test(cfg, model, device, test_dataloader)