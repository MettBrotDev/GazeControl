import os
import time
import datetime
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from models import PretrainAutoencoder
from config import Config


def main():
    parser = argparse.ArgumentParser(description="Pretrain decoder as autoencoder")
    parser.add_argument("--single", action="store_true", help="Overfit a single random image sampled from the selected dataset")
    parser.add_argument("--steps", type=int, default=None, help="Override Config.PRETRAIN_STEPS")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"pretrain_decoder_{Config.DATA_SOURCE}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    # Pretraining: train on [-1,1] with Tanh output
    transform_img = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])

    if Config.DATA_SOURCE == "mnist":
        transform_mnist = transforms.Compose([
            transforms.Resize(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])
        dataset = datasets.MNIST(root=Config.MNIST_DATA_DIR, train=True, transform=transform_mnist, download=True)
    elif Config.DATA_SOURCE == "cifar100":
        dataset = datasets.CIFAR100(root=Config.CIFAR100_DATA_DIR, train=True, transform=transform_img, download=True)
    else:
        dataset = datasets.CIFAR100(root=Config.CIFAR100_DATA_DIR, train=True, transform=transform_img, download=True)
    if args.single:
        # sample a single random item from dataset
        idx = torch.randint(0, len(dataset), (1,)).item()
        img_t, _ = dataset[idx]
        if img_t.dim() == 3:
            img_t = img_t.unsqueeze(0)
        dataset = TensorDataset(img_t, torch.zeros(1, dtype=torch.long))
        batch_size = 1
        shuffle = False
        print(f"Pretraining on a single random dataset image (idx={idx})")
    else:
        batch_size = max(8, getattr(Config, "PRETRAIN_BATCH_SIZE", 32))
        shuffle = True

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    autoenc = PretrainAutoencoder(state_size=Config.HIDDEN_SIZE,
                                  img_size=Config.IMG_SIZE,
                                  decoder_latent_ch=Config.DECODER_LATENT_CH,
                                  out_activation="tanh").to(Config.DEVICE)

    # Optimizer + scheduler + loss (L1-dominant)
    lr = getattr(Config, "PRETRAIN_LR", 3e-3)
    opt = torch.optim.Adam(autoenc.parameters(), lr=lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1000, Config.PRETRAIN_STEPS))
    mse = nn.MSELoss(reduction="mean")
    l1 = nn.L1Loss(reduction="mean")
    w_l1 = getattr(Config, "PRETRAIN_L1_WEIGHT", 0.8)
    w_mse = getattr(Config, "PRETRAIN_MSE_WEIGHT", 0.2)
    use_amp = bool(getattr(Config, "PRETRAIN_USE_AMP", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    steps = 0
    max_steps = args.steps if args.steps is not None else Config.PRETRAIN_STEPS
    autoenc.train()
    best_loss = float("inf")
    best_path = os.path.join(run_dir, "pretrained_decoder_best.pth")
    last_log_t = time.time()
    while steps < max_steps:
        for images, _ in loader:
            images = images.to(Config.DEVICE)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = autoenc(images)
                # losses in [-1,1]
                mse_val = mse(pred, images)
                l1_val = l1(pred, images)
                loss = w_mse * mse_val + w_l1 * l1_val
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            steps += 1
            if steps % 50 == 0:
                writer.add_scalar("PretrainAE/loss", loss.item(), steps)
                writer.add_scalar("PretrainAE/mse", mse_val.item(), steps)
                writer.add_scalar("PretrainAE/l1", l1_val.item(), steps)
                writer.add_scalar("PretrainAE/lr", sched.get_last_lr()[0], steps)
                # log last sample only, like main training (orig vs recon)
                with torch.no_grad():
                    img_last = (images[-1:].detach() + 1.0) / 2.0
                    pred_last = (pred[-1:].detach().clamp(-1, 1) + 1.0) / 2.0
                    grid = vutils.make_grid(torch.cat([img_last, pred_last], dim=0), nrow=2, normalize=True)
                writer.add_image("PretrainAE/original_vs_reconstruction", grid, steps)
                now = time.time()
                img_per_s = (50 * images.size(0)) / max(1e-6, (now - last_log_t))
                last_log_t = now
                print(f"step {steps:6d} | loss {loss.item():.4f} | l1 {l1_val.item():.4f} | mse {mse_val.item():.4f} | lr {sched.get_last_lr()[0]:.2e} | {img_per_s:.1f} img/s")
            # Track best checkpoint
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(autoenc.decoder.state_dict(), best_path)
            if steps >= max_steps:
                break

    dec_path = os.path.join(run_dir, "pretrained_decoder_last.pth")
    torch.save(autoenc.decoder.state_dict(), dec_path)
    print(f"Saved pretrained decoder (last) to {dec_path}")
    if os.path.exists(best_path):
        print(f"Best checkpoint saved to {best_path} (loss={best_loss:.4f})")
        # Also export best to configured pretrained path for easy loading in train.py
        target = getattr(Config, "PRETRAINED_DECODER_PATH", "pretrained_components/pretrained_decoder.pth")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        torch.save(autoenc.decoder.state_dict(), target)
        print(f"Exported best decoder to {target}")
    writer.close()


if __name__ == "__main__":
    main()
