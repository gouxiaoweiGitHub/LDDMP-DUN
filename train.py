import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import os
import copy
from tqdm import tqdm
from config import cfg
from utils import setup_global_seed, save_loss, load_model_del_fix, filter_optimizer
from Models import VQVAE, UNetModel
from dataset import CustomDataset
from algorithm import AMP, Loss


def train_epoch(model, loss, train_loader, optimizer, device, epoch, global_scale):
    model.train()
    avg_loss = 0
    with tqdm(train_loader, desc=f'epoch {epoch + 1}', dynamic_ncols=True) as tqdmTrainLoader:
        for y, real_s, _, _ in tqdmTrainLoader:
            y, real_s = y.to(device), real_s.to(device)
            optimizer.zero_grad()
            pred_s = model(y, global_scale)
            pred_loss = loss(real_s, pred_s)
            pred_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            avg_loss += pred_loss.item()

    return avg_loss / len(train_loader)


def eval_epoch(model, loss, eval_loader, device, epoch, global_scale):
    model.eval()
    avg_loss = 0
    with tqdm(eval_loader, desc=f'epoch {epoch + 1}', dynamic_ncols=True) as tqdmEvalLoader:
        with torch.no_grad():
            for y, real_s, _, _ in tqdmEvalLoader:
                y, real_s = y.to(device), real_s.to(device)
                pred_s = model(y, global_scale)
                pred_loss = loss(real_s, pred_s)
                avg_loss += pred_loss.item()
    return avg_loss / len(eval_loader)


if __name__ == "__main__":
    setup_global_seed()
    train_path = os.path.join("Datasets", "train.mat")
    val_path = os.path.join("Datasets", "val.mat")

    ae = VQVAE(6004, 128)
    base_model = UNetModel(1, 64, 1, 4, 1)
    try:
        ae.load_state_dict(
            torch.load("./Weights/auto.pth",  map_location='cpu', weights_only=True))
        loaded_stats = torch.load(
            "./Weights/diffusion.pth", map_location='cpu', weights_only=True)
        base_model.load_state_dict(
            load_model_del_fix(loaded_stats['state_dict']))
        global_scale = loaded_stats['global_scale']
    except Exception as e:
        print(f"[错误] 加载失败: {type(e).__name__}: {e}")

    train_dataset = CustomDataset(train_path)
    val_dataset = CustomDataset(val_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True)

    model = AMP(base_model, ae, cfg['diffusion']).to(cfg['device'])

    best_loss = float('inf')
    best_weights = None
    loss = Loss()
    os.makedirs("Logs", exist_ok=True)
    os.makedirs("Weights", exist_ok=True)

    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=1e-3,
    #     betas=(0.9, 0.999),
    #     weight_decay=0.01)

    decay_params, no_decay_params = filter_optimizer(model)
    optimizer_params = [
        {
            'params': decay_params,
            'weight_decay': 0.01
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0
        }
    ]

    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=1e-3,
        betas=(0.9, 0.999),
    )

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=1e-7)

    train_loss = []
    eval_loss = []
    for epoch in range(cfg['epochs']):
        train = train_epoch(
            model, loss, train_loader, optimizer, cfg['device'], epoch, global_scale)
        eval = eval_epoch(
            model, loss, val_loader, cfg['device'], epoch, global_scale)

        if best_loss > eval:
            best_loss = eval
            best_weights = copy.deepcopy(model.state_dict())
            checkpoint = {
                'global_scale': global_scale,
                'state_dict': best_weights
            }
            torch.save(checkpoint, 'Weights/weight.pth')

        logFile = os.path.join("Logs", "trainFile.txt")
        with open(logFile, 'a') as file:
            current_lr = optimizer.param_groups[0]['lr']
            file.write(f"Epoch {epoch + 1}: rate = {current_lr:.1e}\n")
            file.write(f'epoch:{epoch + 1} train_loss={train :.4f}\n')
            file.write(f'epoch:{epoch + 1} eval_loss={eval :.4f}\n')
        scheduler.step()
