"""
Adım 1 - Veri hazırlama

MNIST'in resmi hali 60.000 eğitim + 10.000 test görüntüsüdür.
Biz resmi eğitim kümesini ikiye bölüyoruz:
    - 50.000 train      -> modelin ağırlıkları bununla güncellenir
    - 10.000 validation -> hiperparametre seçimi / overfitting takibi
Resmi 10.000'lik test kümesine ise en sona kadar DOKUNMUYORUZ.

Hız notu: MNIST küçük (~200 MB float32) olduğu için tüm veriyi bir kez tensöre
çevirip (mümkünse GPU belleğine) koyuyoruz. Her epoch'ta görüntüleri tek tek
PIL -> tensör dönüştürmek epoch başına ~23 sn sürüyordu.
"""

import torch
from torch.utils.data import TensorDataset
from torchvision import datasets

DATA_DIR = "./mnist_data"
SEED = 42
TRAIN_SIZE = 50_000
VAL_SIZE = 10_000


def _to_float(raw):
    """uint8 [N, 28, 28] (0-255) -> float [N, 1, 28, 28] (0-1)"""
    return raw.float().div(255.0).unsqueeze(1)


def get_datasets(device="cpu"):
    full_train = datasets.MNIST(DATA_DIR, train=True, download=True)
    test_raw = datasets.MNIST(DATA_DIR, train=False, download=True)

    x_full, y_full = _to_float(full_train.data), full_train.targets
    x_test, y_test = _to_float(test_raw.data), test_raw.targets

    # Sabit seed ile karıştırıp böl -> her çalıştırmada aynı train/val ayrımı
    generator = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(x_full), generator=generator)
    train_idx, val_idx = perm[:TRAIN_SIZE], perm[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]

    # Normalizasyon istatistikleri SADECE train kümesinden hesaplanır.
    # Validation/test verisinden istatistik çıkarmak veri sızıntısı (data leakage) olur.
    mean = x_full[train_idx].mean().item()
    std = x_full[train_idx].std().item()

    def normalize(x):
        return ((x - mean) / std).to(device)

    train_set = TensorDataset(normalize(x_full[train_idx]), y_full[train_idx].to(device))
    val_set = TensorDataset(normalize(x_full[val_idx]), y_full[val_idx].to(device))
    test_set = TensorDataset(normalize(x_test), y_test.to(device))
    return train_set, val_set, test_set, (mean, std)


class TensorLoader:
    """DataLoader'ın sade hali. Veri zaten bellekte (GPU'da) olduğu için her batch'i
    örnekleri tek tek okumak yerine karıştırılmış indekslerle tek hamlede dilimler."""

    def __init__(self, dataset, batch_size, shuffle):
        self.x, self.y = dataset.tensors
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        n = len(self.x)
        idx = torch.randperm(n, device=self.x.device) if self.shuffle else torch.arange(n, device=self.x.device)
        for i in range(0, n, self.batch_size):
            batch = idx[i:i + self.batch_size]
            yield self.x[batch], self.y[batch]

    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size


def get_dataloaders(batch_size=64, device="cpu"):
    train_set, val_set, test_set, stats = get_datasets(device)
    train_loader = TensorLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = TensorLoader(val_set, batch_size=1000, shuffle=False)
    test_loader = TensorLoader(test_set, batch_size=1000, shuffle=False)
    return train_loader, val_loader, test_loader, stats


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    train_set, val_set, test_set, (mean, std) = get_datasets()
    print(f"Train: {len(train_set)} | Validation: {len(val_set)} | Test: {len(test_set)}")
    print(f"Train kümesinden hesaplanan mean={mean:.4f}, std={std:.4f}")

    img, label = train_set[0]
    print(f"Tek görüntünün boyutu: {tuple(img.shape)}, etiketi: {label.item()}")
    print(f"Normalize sonrası piksel aralığı: [{img.min():.2f}, {img.max():.2f}]")

    os.makedirs("outputs", exist_ok=True)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.8))
    for ax, i in zip(axes.flat, range(16)):
        img, label = train_set[i]
        ax.imshow(img.squeeze() * std + mean, cmap="gray")  # görselleştirmek için normalizasyonu geri al
        ax.set_title(str(label.item()))
        ax.axis("off")
    plt.suptitle("MNIST train kümesinden örnekler (siyah arka plan, beyaz rakam)")
    plt.tight_layout()
    plt.savefig("outputs/mnist_ornekler.png", dpi=120)
    print("Örnek görüntüler outputs/mnist_ornekler.png dosyasına kaydedildi.")
