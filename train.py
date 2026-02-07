import os
import torch
import wandb
import argparse
from torch.utils.data import DataLoader
from src.models.hubert_beats_model import AudioClassificationModel
from src.datasets.audio_dataset import AudioEmotionDataset
from torch.cuda.amp import GradScaler, autocast
import logging
import time
import gc
import pdb
import random
import multiprocessing as mp
os.environ["TOKENIZERS_PARALLELISM"] = "false"
CUDA_VISIBLE_DEVICES = 0

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training3.log"),  # 日志输出到文件
        logging.StreamHandler()               # 日志输出到终端
    ]
)

# 设置随机种子
seed = 42
torch.manual_seed(seed)            # CPU 随机数种子
torch.cuda.manual_seed(seed)       # GPU 随机数种子
torch.cuda.manual_seed_all(seed)   # 多 GPU 随机数种子
random.seed(seed)                  # Python 的随机数种子
# np.random.seed(seed)               # NumPy 随机数种子


# -------------------- Argument Parser --------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", default=False, help="Run testing on the test dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for DataLoader")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint file for resuming/testing")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size for projection layers")
    parser.add_argument("--accum_grad_iters", type=int, default=2, help="Number of iterations for gradient accumulation")
    args = parser.parse_args()
    return args

# -------------------- Dataset and Dataloader --------------------
def get_dataloaders(batch_size):
    train_annotation = "data/train_annotation_emotion.json"
    dev_annotation = "data/dev_annotation_adjusted.json"
    test_annotation = "data/test_annotation_adjusted.json"

    train_dataset = AudioEmotionDataset(annotation_file=train_annotation)
    dev_dataset = AudioEmotionDataset(annotation_file=dev_annotation)
    test_dataset = AudioEmotionDataset(annotation_file=test_annotation)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                              collate_fn=train_dataset.collater, pin_memory=True)
    val_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                            collate_fn=dev_dataset.collater, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                             collate_fn=test_dataset.collater, pin_memory=True)

    return train_loader, val_loader, test_loader

# -------------------- Model Initialization --------------------
def initialize_model(hidden_size):
    hubert_path = "src/models/hubert"
    beats_path = "src/models/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    llama_model_path = "src/models/llama"
    model = AudioClassificationModel(
        hubert_model_path=hubert_path,
        beats_model_path=beats_path,
        llama_model_path=llama_model_path,
        hidden_size=hidden_size,
        num_classes=7,
        freeze_beats=True,
        freeze_hubert=True
    )
    return model

# -------------------- Optimizer and Scheduler --------------------
def get_optimizer(model, config):
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
    optim_params = [
        {"params": p_wd, "weight_decay": config.weight_decay},
        {"params": p_non_wd, "weight_decay": 0},
    ]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=config.init_lr,
        weight_decay=config.weight_decay,
        betas=(0.9, config.beta2),
    )
    return optimizer

class LinearWarmupCosineLRScheduler:
    def __init__(self, optimizer, max_epoch, iters_per_epoch, min_lr, init_lr, warmup_steps=0, warmup_start_lr=-1):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        total_cur_step = cur_epoch * self.iters_per_epoch + cur_step
        if total_cur_step < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=total_cur_step,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch * self.iters_per_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )

def warmup_lr_schedule(step, optimizer, max_step, init_lr, max_lr):
    lr = init_lr + step * (max_lr - init_lr) / max_step
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def cosine_lr_schedule(epoch, optimizer, max_epoch, init_lr, min_lr):
    epoch = torch.tensor(float(epoch), dtype=torch.float32)
    max_epoch = torch.tensor(float(max_epoch), dtype=torch.float32)  # 确保是 Tensor
    lr = min_lr + 0.5 * (init_lr - min_lr) * (1 + torch.cos(torch.pi * epoch / max_epoch))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr.item()


# -------------------- Training Function --------------------
def train(model, train_loader, val_loader, optimizer, scheduler, device, args):
    best_val_accuracy = 0.0
    best_gen_acc =0.0
    patience, early_stop_counter = 20, 0
    output_dir = "./checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_model.pt")

    wandb.init(project="audio_emotion_recognition", name="simple-layout", config=args)
    # wandb.watch(model, log="parameters", log_freq=10)

    scaler = GradScaler()

    for epoch in range(args.num_epochs):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for batch_idx, batch in enumerate(train_loader):
            raw_wav = batch["raw_wav"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)

            # Forward pass
            outputs = model(samples=batch, speech_input=raw_wav, acoustic_input=raw_wav, padding_mask=padding_mask)
            loss = outputs["loss"] / args.accum_grad_iters  # Gradient accumulation
            correct = outputs.get("correct", 0)
            total = outputs.get("total", 0)

            # Accumulate loss
            # total_loss += outputs["loss"].item()
            total_loss += loss.item()  # 记录最终混合损失

            # Backward pass
            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.accum_grad_iters == 0 or (batch_idx + 1) == len(train_loader):
                # 先解除缩放，恢复真实梯度
                scaler.unscale_(optimizer)

                # 进行梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 进行优化器更新
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # 清空梯度，防止影响下一次更新
                scheduler.step(epoch, batch_idx)

            # Update metrics
            total_correct += correct
            total_samples += total
            # del raw_wav, padding_mask, outputs, batch

            #
        torch.cuda.empty_cache()
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples

        # Validation
        val_loss, val_accuracy, gen_acc = evaluate_generation_token(model, val_loader, device)
        wandb.log({"Train Loss": avg_train_loss, "Train Accuracy": train_accuracy,
                   "Val Loss": val_loss, "Val Accuracy": val_accuracy})

        logging.info(f"Epoch [{epoch + 1}/{args.num_epochs}] | Train Loss: {avg_train_loss:.4f} | "
                     f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | "
                     f"Gen Accuracy: {gen_acc:.4f}")

        # Save best model
        if gen_acc >= best_gen_acc:
            best_gen_acc = gen_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Epoch {epoch + 1}: Best model saved with gen Accuracy: {gen_acc:.4f}")
        else:
            early_stop_counter += 1
            logging.info(f"Epoch {epoch + 1}: Early stop counter incremented to {early_stop_counter}")
            if early_stop_counter >= patience:
                logging.info("Early stopping triggered.")
                break

    wandb.finish()

# -------------------- Evaluation Function --------------------
def evaluate_generation_token(model, val_loader, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    total_gen_correct = 0  # 用于统计生成的类别正确数量
    total_gen_samples = 0  # 用于统计生成的样本数量

    results = []

    with torch.no_grad():
        for batch in val_loader:
            raw_wav = batch["raw_wav"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            outputs = model(samples=batch, speech_input=raw_wav, acoustic_input=raw_wav, padding_mask=padding_mask)
            # loss = outputs["loss"]
            loss = outputs["loss"]
            correct = outputs.get("correct", 0)
            total = outputs.get("total", 0)

            total_loss += loss.item()
            total_correct += correct
            total_samples += total

            # model.eval()
            # Generation-level评估
            gen_results = model.evaluate_model(
                samples=batch,
                speech_input=raw_wav,
                acoustic_input=raw_wav,
                padding_mask=padding_mask,
                save_path="generated_results.json"
            )
            total_gen_correct += sum(1 for result in gen_results if result["match"])
            total_gen_samples += len(gen_results)

    # Token-level
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples

    # Generation-level
    gen_accuracy = total_gen_correct / total_gen_samples

    return avg_loss, accuracy, gen_accuracy


# -------------------- Main Function --------------------
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size)
    model = initialize_model(args.hidden_size).to(device)

    config = argparse.Namespace(
        weight_decay=0.03,
        init_lr=5e-6,
        beta2=0.999,
        min_lr=1e-7,
        warmup_steps=1500,
        warmup_start_lr=1e-6,
        max_epoch=args.num_epochs,
    )
    optimizer = get_optimizer(model, config)
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=args.num_epochs,
        iters_per_epoch=len(train_loader),
        min_lr=config.min_lr,
        init_lr=config.init_lr,
        warmup_steps=config.warmup_steps,
        warmup_start_lr=config.warmup_start_lr,
    )

    if not args.test:
        train(model, train_loader, val_loader, optimizer, scheduler, device, args)
    else:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        test_loss, test_accuracy, gen_accuracy = evaluate_generation_token(model, test_loader, device)
        logging.info(f"Validation: Avg Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Gen Accuracy: {gen_accuracy:.4f}")


if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    main()
