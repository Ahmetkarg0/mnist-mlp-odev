"""
Adım 5 - Kendi el yazısı rakamlarla test

Bir kağıda yazılmış okul numarasının fotoğrafı MNIST formatına dönüştürülür:
    1. Gri tonlama (+ telefonun EXIF döndürme bilgisinin uygulanması)
    2. Aydınlatma düzeltme: kağıdın arka planı tahmin edilip görüntü ona bölünür
    3. Eşikleme (Otsu) + renk ters çevirme -> siyah arka plan, beyaz rakam (MNIST gibi)
    4. Bağlantılı bileşenlerle rakamları bulma, soldan sağa sıralama
    5. Her rakamı: kırp -> çizgiyi kalınlaştır -> en-boy oranını koruyarak 20x20 kutuya sığdır
       -> 28x28 içine ağırlık merkezi ortada olacak şekilde yerleştir (MNIST'in kendi yöntemi)
    6. Train kümesinin mean/std değeriyle normalizasyon

Kullanım:
    python predict_own.py el_yazisi/numara.jpg --label 2021123456
    python predict_own.py el_yazisi/numara.jpg --label 2021123456 --all-models
"""

import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from model import MLP


def load_gray(path, max_side=1200):
    # cv2.imread Windows'ta Türkçe karakterli yollarda ("Masaüstü") çalışmadığı için PIL ile okuyoruz
    img = ImageOps.exif_transpose(Image.open(path))  # telefon fotoğraflarının yönünü düzelt
    gray = np.array(img.convert("L"))
    scale = max_side / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray


def binarize(gray):
    """Beyaz kağıt üstü koyu mürekkep -> siyah zemin üstü beyaz rakam maskesi (0/255)."""
    # Büyük çekirdekli "closing", koyu ince çizgileri silip sadece kağıdın parlaklığını bırakır.
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    flat = cv2.divide(gray, background, scale=255)  # gölge / ışık farkı giderilir
    flat = cv2.GaussianBlur(flat, (5, 5), 0)
    _, mask = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return flat, mask


def find_digits(mask):
    """Rakam maskelerini soldan sağa sıralı olarak döndürür: [(maske, (x, y, w, h)), ...]"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = mask.shape
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches_border = x == 0 or y == 0 or x + w >= W or y + h >= H
        if not touches_border:
            comps.append([x, x + w, [i]])
    if not comps:
        return []
    max_area = max(stats[c[2][0], cv2.CC_STAT_AREA] for c in comps)
    comps = [c for c in comps if stats[c[2][0], cv2.CC_STAT_AREA] > 0.01 * max_area]  # gürültü noktaları

    # Yatayda üst üste binen parçaları birleştir (ör. 5'in şapkası, kopuk çizilmiş 4)
    comps.sort(key=lambda c: c[0])
    groups = [comps[0]]
    for x0, x1, ids in comps[1:]:
        gx0, gx1, gids = groups[-1]
        overlap = min(x1, gx1) - max(x0, gx0)
        if overlap > 0.5 * min(x1 - x0, gx1 - gx0):
            groups[-1] = [gx0, max(x1, gx1), gids + ids]
        else:
            groups.append([x0, x1, ids])

    digits = []
    for _, _, ids in groups:
        digit_mask = np.isin(labels, ids).astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(digit_mask)
        digits.append((digit_mask[y:y + h, x:x + w], (x, y, w, h)))
    max_h = max(box[3] for _, box in digits)
    return [d for d in digits if d[1][3] > 0.4 * max_h]  # rakamdan çok küçük lekeleri at


def to_mnist(crop, thicken=True):
    """Kırpılmış rakam maskesini MNIST tarzı 28x28, 0-1 aralığında görüntüye çevirir."""
    h, w = crop.shape
    if thicken:
        # MNIST'te çizgi kalınlığı rakam boyunun ~%12'si; kalem çizgisi genelde çok daha ince.
        stroke = 2 * cv2.distanceTransform(crop, cv2.DIST_L2, 3).max()
        k = int(round(0.12 * max(h, w) - stroke))
        if k >= 2:
            crop = cv2.copyMakeBorder(crop, k, k, k, k, cv2.BORDER_CONSTANT, value=0)
            crop = cv2.dilate(crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            h, w = crop.shape

    scale = 20.0 / max(h, w)  # uzun kenar 20 piksel, en-boy oranı korunur
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(crop.astype(np.float32) / 255.0, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img = np.zeros((28, 28), np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    img[top:top + new_h, left:left + new_w] = small

    # Ağırlık merkezini (14, 14)'e kaydır. MLP konum değişimine karşı dayanıksızdır
    # (piksel 300'ün anlamı her zaman aynıdır), bu yüzden ortalama kritik.
    m = cv2.moments(img)
    if m["m00"] > 0:
        dx, dy = 14 - m["m10"] / m["m00"], 14 - m["m01"] / m["m00"]
        img = cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28))
    return np.clip(img, 0, 1)


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = MLP(cfg["hidden_sizes"], cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"]


@torch.no_grad()
def predict(model, mean, std, images, device):
    x = torch.tensor(np.stack(images)).unsqueeze(1).to(device)  # [N, 1, 28, 28]
    x = (x - mean) / std  # eğitimdeki normalizasyonun AYNISI
    probs = torch.softmax(model(x), dim=1).cpu().numpy()
    return probs.argmax(1), probs.max(1)


def save_figures(gray, flat, mask, digits, images, preds, confs, label, out_prefix):
    # 1) Ön işleme adımları
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for _, (x, y, w, h) in digits:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), max(2, gray.shape[1] // 300))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, im, t in zip(axes, [vis, flat, mask],
                         ["1) Orijinal (gri) + bulunan rakamlar", "2) Aydınlatma düzeltilmiş",
                          "3) Eşiklenmiş + ters çevrilmiş (MNIST renkleri)"]):
        ax.imshow(im, cmap="gray"); ax.set_title(t); ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_on_isleme.png", dpi=110)
    plt.close(fig)

    # 2) 28x28 girdiler ve tahminler
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 2.6))
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        true = label[i] if label and i < len(label) else None
        color = "black" if true is None else ("green" if str(preds[i]) == true else "red")
        ax.set_title(f"tahmin {preds[i]}\n%{100 * confs[i]:.0f}" + (f"\n(gerçek {true})" if true else ""),
                     color=color, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_tahminler.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="okul numarasının fotoğrafı")
    parser.add_argument("--label", help="gerçek numara (doğruluk hesabı için), ör. 2021123456")
    parser.add_argument("--model", default="models/deney1_temel.pt")
    parser.add_argument("--all-models", action="store_true", help="models/ içindeki tüm modelleri karşılaştır")
    parser.add_argument("--no-thicken", action="store_true", help="çizgi kalınlaştırmayı kapat")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gray = load_gray(args.image)
    flat, mask = binarize(gray)
    digits = find_digits(mask)
    images = [to_mnist(crop, thicken=not args.no_thicken) for crop, _ in digits]
    print(f"{len(images)} rakam bulundu.")
    if args.label and len(args.label) != len(images):
        print(f"UYARI: numarada {len(args.label)} hane var ama {len(images)} rakam bulundu. "
              "Rakamlar birbirine değiyor ya da kağıt dışında leke olabilir.")

    base = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = os.path.join(os.path.dirname(args.image) or ".", "islenmis")
    os.makedirs(out_dir, exist_ok=True)
    for i, im in enumerate(images):  # modele verilen 28x28 görüntüler
        Image.fromarray((im * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{base}_rakam{i + 1}.png"))

    model, mean, std = load_model(args.model, device)
    preds, confs = predict(model, mean, std, images, device)
    print(f"Tahmin:  {''.join(map(str, preds))}")
    if args.label:
        print(f"Gerçek:  {args.label}")
        n = min(len(args.label), len(preds))
        correct = sum(str(p) == t for p, t in zip(preds[:n], args.label[:n]))
        print(f"Doğruluk: {correct}/{len(args.label)} = %{100 * correct / len(args.label):.1f}")

    os.makedirs("outputs", exist_ok=True)
    save_figures(gray, flat, mask, digits, images, preds, confs, args.label, f"outputs/el_yazisi_{base}")
    print(f"Grafikler: outputs/el_yazisi_{base}_on_isleme.png, outputs/el_yazisi_{base}_tahminler.png")

    if args.all_models:
        print(f"\n{'Model':28s} {'Tahmin':>14s}  Doğru")
        for path in sorted(glob.glob("models/*.pt")):
            m, mu, sd = load_model(path, device)
            p, _ = predict(m, mu, sd, images, device)
            pred_str = "".join(map(str, p))
            score = sum(a == b for a, b in zip(pred_str, args.label)) if args.label else "-"
            print(f"{os.path.basename(path)[:-3]:28s} {pred_str:>14s}  {score}")


if __name__ == "__main__":
    main()
