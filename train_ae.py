import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import os
from config import cfg
from utils import setup_global_seed, save_loss
from Models import VQVAE, Loss
from dataset import CustomDataset


def train_epoch(model, loss_fn, train_loader, optimizer, device, epoch):
    model.train()
    avg_loss = 0
    with tqdm(train_loader, desc=f'epoch {epoch + 1}', dynamic_ncols=True) as tqdmTrainLoader:
        for y, real_s, init_s, seedvox in tqdmTrainLoader:
            real_s = real_s.to(device)
            optimizer.zero_grad()
            pred_s, vq_loss = model(real_s)
            pred_loss = loss_fn(real_s, pred_s) + vq_loss
            pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
           
            optimizer.step()
            avg_loss += pred_loss.item()

    return avg_loss / len(train_loader)


def eval_epoch(model, loss_fn,  eval_loader, device, epoch):
    model.eval()
    avg_loss = 0
    with tqdm(eval_loader, desc=f'epoch {epoch + 1}', dynamic_ncols=True) as tqdmEvalLoader:
        with torch.no_grad():
            for y, real_s, init_s, seedvox in tqdmEvalLoader:
                real_s = real_s.to(device)
                pred_s, vq_loss = model(real_s)
                pred_loss = loss_fn(real_s, pred_s) + vq_loss
                avg_loss += pred_loss.item()
    return avg_loss / len(eval_loader)


if __name__ == "__main__":
    setup_global_seed()
    train_path = os.path.join("Datasets", "train.mat")
    val_path = os.path.join("Datasets", "val.mat")

    train_dataset = CustomDataset(train_path)
    val_dataset = CustomDataset(val_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True)

    best_loss = float('inf')
    loss = Loss()

    model = VQVAE(6004, 128).to(cfg['device'])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3, betas=(0.9, 0.999))
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=1e-7)

    
    os.makedirs("Logs", exist_ok=True)
    os.makedirs("Weights", exist_ok=True)

    for epoch in range(cfg['epochs']):
        train_loss_val = train_epoch(
            model, loss, train_loader, optimizer, cfg['device'], epoch)
        eval_loss_val = eval_epoch(
            model, loss, val_loader, cfg['device'], epoch)
        if best_loss > eval_loss_val:
            best_loss = eval_loss_val
            torch.save(model.state_dict(), 'Weights/auto.pth')

        logFile = os.path.join("Logs", "train_ae.txt")
        with open(logFile, 'a') as file:
            current_lr = optimizer.param_groups[0]['lr']
            file.write(f"Epoch {epoch + 1}: rate = {current_lr:.1e}\n")
            file.write(
                f'epoch:{epoch + 1} train_loss={train_loss_val :.4f}\n')
            file.write(
                f'epoch:{epoch + 1} eval_loss={eval_loss_val :.4f}\n')
        scheduler.step()
    