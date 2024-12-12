from typing import Tuple
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import warnings
warnings.filterwarnings(action='ignore')

from src.utils.loss import mean_iou, compute_loss

def sam_validation(
        cfg: DictConfig, 
        model: nn.Module, 
        device: torch.device,
        scheduler: lr_scheduler, 
        val_loader: DataLoader
        ) -> Tuple[float, float]:
    
    model.to(device)
    model.eval()
    scheduler.step()
    val_miou = []
    val_loss = []

    with torch.no_grad():
        
        for input, target in tqdm(val_loader):

            input = input.to(device)
            target = target.to(device, dtype=torch.float32)

            val_encode_feature = model.image_encoder(input)
            val_sparse_embeddings, val_dense_embeddings = model.prompt_encoder(points = None, 
                                                                               boxes = None, 
                                                                               masks = None
                                                                               )
            pred, pred_iou = model.mask_decoder(image_embeddings = val_encode_feature,
                                                image_pe = model.prompt_encoder.get_dense_pe(),
                                                sparse_prompt_embeddings = val_sparse_embeddings,
                                                dense_prompt_embeddings = val_dense_embeddings,
                                                multimask_output = False
                                                )
            
            true_iou = mean_iou(pred, target, eps=1e-6)
            loss = compute_loss(pred, target, pred_iou, true_iou)

            val_miou += true_iou.tolist()
            val_loss.append(loss.item())

        _val_miou = np.mean(val_miou)
        _val_loss = np.mean(val_loss)

    return _val_miou, _val_loss
    
def sam_train(
        cfg: DictConfig, 
        model: nn.Module, 
        device: torch.device, 
        optimizer: optim, 
        scheduler: lr_scheduler, 
        train_loader: DataLoader, 
        val_loader: DataLoader
        ) -> None:
    
    best_miou = 0
    best_model = None

    train_miou_list = []
    train_loss_list = []
    val_loss_list = []
    val_miou_list = []

    try:
        for epoch in range(cfg.epochs):
        
            model.to(device)
            model.train()
            train_miou = []
            train_loss = []
            
            for input, target in tqdm(train_loader):

                input = input.to(device)
                target = target.to(device, dtype=torch.float32)

                with torch.no_grad():

                    train_encode_feature = model.image_encoder(input)
                    train_sparse_embeddings, train_dense_embeddings = model.prompt_encoder(points=None,
                                                                                           boxes=None,
                                                                                           masks=None
                                                                                           )
                pred, pred_iou = model.mask_decoder(image_embeddings = train_encode_feature,
                                                    image_pe = model.prompt_encoder.get_dense_pe(),
                                                    sparse_prompt_embeddings = train_sparse_embeddings,
                                                    dense_prompt_embeddings = train_dense_embeddings,
                                                    multimask_output = False
                                                    )
                
                true_iou = mean_iou(pred, target, eps=1e-6)
                loss = compute_loss(pred, target, pred_iou, true_iou)

                train_miou += true_iou.tolist()
                train_loss.append(loss.item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            _train_miou = np.mean(train_miou)
            _train_loss = np.mean(train_loss)
            _val_miou, _val_loss = sam_validation(cfg, model, device, scheduler, val_loader)
            
            print(f"Epoch[{epoch+1}]")
            print(f"Train_miou: [{_train_miou:.4f}]")   
            print(f"Train_loss: [{_train_loss:.4f}]")   
            print(f"Val_miou: [{_val_miou:.4f}]")
            print(f"Val_loss: [{_val_loss:.4f}]")
            print('#' * 200)

            # torch.cuda.empty_cache()
            train_miou_list.append(_train_miou)
            train_loss_list.append(_train_loss)
            val_miou_list.append(_val_miou)
            val_loss_list.append(_val_loss)

            if best_miou < _val_miou:

                best_miou = _val_miou
                best_model = model
                model_save_path = f"./weight/{cfg.memo}.pth"
                torch.save(best_model.to('cpu').state_dict(), model_save_path)
                best_model.to(device)

                print(f"{epoch+1} ::::::::::::::: update the best model best miou {best_miou} :::::::::::::::")
                print('#' * 200)
    
    except KeyboardInterrupt as e:
        print(f"{e} 그래프를 그립니다.")

    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./history/{cfg.memo}_loss.png")

    plt.figure(figsize=(12, 6))
    plt.plot(train_miou_list, label='Train mIOU')
    plt.plot(val_miou_list, label='Validation mIOU')
    plt.title("mIOU History")
    plt.xlabel('Epochs')
    plt.ylabel('miou')
    plt.legend()
    plt.savefig(f"./history/{cfg.memo}_miou.png")