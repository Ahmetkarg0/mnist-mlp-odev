import copy
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import torch
from torch import nn

from data import get_dataloaders
from model import MLP, count_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    # Toplamlar GPU'da tutulur; her batch'te .item() çağırmak GPU'yu bekletir
    total_loss = torch.zeros((), device=DEVICE)
    correct = torch.zeros((), device=DEVICE)
    n = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum()
        n += labels.size(0)
    return total_loss.item() / n, correct.item() / n


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


def run_experiment(name, hidden_sizes=(256, 128), dropout=0.0, optimizer_name="adam",
                   lr=1e-3, momentum=0.9, batch_size=64, epochs=20, seed=42, verbose=True):
    set_seed(seed)
    train_loader, val_loader, test_loader, (mean, std) = get_dataloaders(batch_size, DEVICE)

    model = MLP(hidden_sizes, dropout).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        raise ValueError(optimizer_name)

    config = dict(hidden_sizes=list(hidden_sizes), dropout=dropout, optimizer=optimizer_name,
                  lr=lr, momentum=momentum if optimizer_name == "sgd" else None,
                  batch_size=batch_size, epochs=epochs, seed=seed,
                  params=count_parameters(model), device=str(DEVICE))
    if verbose:
        print(f"\n=== {name} === {config}")
        init_loss, init_acc = evaluate(model, val_loader, loss_fn)
        print(f"Eğitim öncesi val loss: {init_loss:.4f} (beklenen ~ln(10)=2.3026), val acc: {init_acc:.4f}")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch, best_state = 0.0, 0, None

    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn)
        for key, val in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[key].append(val)
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"Epoch {epoch:2d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
                  f"| val loss {va_loss:.4f} acc {va_acc:.4f}")
    train_time = time.perf_counter() - start

    # Model seçimi validation'a göre yapıldı; test sadece burada, bir kez kullanılır
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)

    results = dict(
        name=name, config=config, history=history, train_time_sec=train_time,
        best_epoch=best_epoch, best_val_acc=best_val_acc,
        final_train_loss=history["train_loss"][-1], final_val_loss=history["val_loss"][-1],
        final_train_acc=history["train_acc"][-1], final_val_acc=history["val_acc"][-1],
        test_loss=test_loss, test_acc=test_acc,
    )
    if verbose:
        print(f"Eğitim süresi: {train_time:.1f} sn | En iyi epoch: {best_epoch} "
              f"(val acc {best_val_acc:.4f}) | TEST acc: {test_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save({"state_dict": best_state, "config": config, "mean": mean, "std": std},
               f"models/{name}.pt")
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{name}_sonuclar.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def plot_history(results, path):
    h = results["history"]
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, h["train_loss"], "o-", label="Train loss")
    ax1.plot(epochs, h["val_loss"], "s-", label="Validation loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss (Cross-Entropy)")
    ax1.set_title("Loss"); ax1.grid(alpha=0.3); ax1.legend()

    ax2.plot(epochs, h["train_acc"], "o-", label="Train accuracy")
    ax2.plot(epochs, h["val_acc"], "s-", label="Validation accuracy")
    ax2.axvline(results["best_epoch"], color="gray", ls="--", label=f"En iyi epoch ({results['best_epoch']})")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy"); ax2.grid(alpha=0.3); ax2.legend()

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f"{results['name']}  |  Test accuracy: {results['test_acc']:.4f}")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    results = run_experiment("deney1_temel", hidden_sizes=(256, 128), optimizer_name="adam",
                             lr=1e-3, batch_size=64, epochs=20)
    plot_history(results, "outputs/deney1_temel_grafikler.png")
    print("Grafik: outputs/deney1_temel_grafikler.png")
