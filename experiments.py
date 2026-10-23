import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from train import plot_history, run_experiment

BASE = dict(hidden_sizes=(256, 128), dropout=0.0, optimizer_name="adam", lr=1e-3,
            batch_size=64, epochs=20)

# Her deneyde temel ayarlardan sadece bir tanesi değişir
RUNS = {
    "deney1_temel":         {},
    "deney2_lr_0.1":        dict(lr=0.1),
    "deney2_lr_0.01":       dict(lr=0.01),
    "deney2_lr_0.0001":     dict(lr=1e-4),
    "deney3_sgd_momentum":  dict(optimizer_name="sgd", lr=0.01, momentum=0.9),
    "deney3_sgd_saf":       dict(optimizer_name="sgd", lr=0.01, momentum=0.0),
    "deney4_kucuk_64_32":   dict(hidden_sizes=(64, 32)),
    "deney4_genis_1024_512": dict(hidden_sizes=(1024, 512)),
    "deney4_derin_4katman": dict(hidden_sizes=(512, 256, 128, 64)),
    "deney5_dropout_0.2":   dict(dropout=0.2),
    "deney5_dropout_0.5":   dict(dropout=0.5),
    "deney6_seed_1":        dict(seed=1),
    "deney6_seed_2":        dict(seed=2),
    "deney6_seed_3":        dict(seed=3),
    "deney6_seed_4":        dict(seed=4),
}

GROUPS = {
    "deney2_learning_rate": ("Deney 2 - Learning rate (Adam)", [
        ("lr=0.1", "deney2_lr_0.1"), ("lr=0.01", "deney2_lr_0.01"),
        ("lr=0.001 (temel)", "deney1_temel"), ("lr=0.0001", "deney2_lr_0.0001")]),
    "deney3_optimizer": ("Deney 3 - Optimizer (lr=0.01 SGD / lr=0.001 Adam)", [
        ("Adam lr=0.001 (temel)", "deney1_temel"),
        ("SGD + momentum 0.9", "deney3_sgd_momentum"), ("SGD (momentum yok)", "deney3_sgd_saf")]),
    "deney4_mimari": ("Deney 4 - Mimari", [
        ("64-32", "deney4_kucuk_64_32"), ("256-128 (temel)", "deney1_temel"),
        ("1024-512", "deney4_genis_1024_512"), ("512-256-128-64", "deney4_derin_4katman")]),
    "deney5_dropout": ("Deney 5 - Dropout", [
        ("dropout=0 (temel)", "deney1_temel"), ("dropout=0.2", "deney5_dropout_0.2"),
        ("dropout=0.5", "deney5_dropout_0.5")]),
    "deney6_seed": ("Deney 6 - Temel model, farklı seed'ler", [
        ("seed=42 (temel)", "deney1_temel"), ("seed=1", "deney6_seed_1"), ("seed=2", "deney6_seed_2"),
        ("seed=3", "deney6_seed_3"), ("seed=4", "deney6_seed_4")]),
}


def load(name):
    with open(f"outputs/{name}_sonuclar.json", encoding="utf-8") as f:
        return json.load(f)


def plot_group(title, members, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for label, name in members:
        h = load(name)["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], "o-", ms=3, label=label)
        axes[1].plot(epochs, h["val_loss"], "o-", ms=3, label=label)
        axes[2].plot(epochs, h["val_acc"], "o-", ms=3, label=label)
    for ax, t in zip(axes, ["Train loss", "Validation loss", "Validation accuracy"]):
        ax.set_title(t); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_yscale("log"); axes[1].set_yscale("log")
    axes[2].set_ylim(0.9, 1.0)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def summary_table():
    lines = ["| Deney | Gizli katmanlar | Optimizer | lr | Dropout | Parametre | Train loss | Val loss "
             "| Train acc | En iyi val acc (epoch) | Test acc | Süre (sn) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in RUNS:
        r = load(name)
        c = r["config"]
        opt = c["optimizer"] if c["optimizer"] == "adam" else f"sgd (m={c['momentum']})"
        lines.append(
            f"| {name} | {'-'.join(map(str, c['hidden_sizes']))} | {opt} | {c['lr']} | {c['dropout']} "
            f"| {c['params']:,} | {r['final_train_loss']:.4f} | {r['final_val_loss']:.4f} "
            f"| {r['final_train_acc']:.4f} | {r['best_val_acc']:.4f} ({r['best_epoch']}) "
            f"| {r['test_acc']:.4f} | {r['train_time_sec']:.1f} |")
    seed_runs = ["deney1_temel"] + [n for n in RUNS if n.startswith("deney6_seed")]
    for key, title in [("best_val_acc", "En iyi val acc"), ("test_acc", "Test acc")]:
        vals = np.array([load(n)[key] for n in seed_runs])
        lines.append(f"\n{title} ({len(vals)} seed): ortalama {vals.mean():.4f} ± {vals.std(ddof=1):.4f} (std)")
    table = "\n".join(lines)
    with open("outputs/ozet_tablo.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    return table


if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(RUNS)
    if "--plot" not in sys.argv:
        for name in only:
            changes = RUNS[name]
            results = run_experiment(name, **{**BASE, **changes})
            plot_history(results, f"outputs/{name}_grafikler.png")

    for file_name, (title, members) in GROUPS.items():
        plot_group(title, members, f"outputs/{file_name}_karsilastirma.png")
    print("\n" + summary_table())
    print("\nTablo: outputs/ozet_tablo.md")
