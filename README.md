# MNIST ile MLP Tabanlı El Yazısı Rakam Tanıma

**Öğrenci:** Ahmet KARGI — 24120205078 · **Ders:** Derin Öğrenme

MNIST üzerinde PyTorch ile **sıfırdan** eğitilen bir MLP (CNN ve hazır eğitilmiş model kullanılmadı). Temel modelden sonra learning rate, optimizer, mimari ve dropout karşılaştırıldı, son olarak model kendi el yazımla yazdığım okul numaram üzerinde test edildi.

## Çalıştırma

```bash
pip install -r requirements.txt
python train.py         # Deney 1 (~30 sn, GPU)
python experiments.py   # Deney 1-6, 15 eğitim (~8 dk); --plot: sadece grafik/tablo üretir
python predict_own.py el_yazisi/numara.jpg --label 24120205078 --all-models --ablation
```

`data.py` veri hazırlama · `model.py` MLP · `train.py` eğitim döngüsü · `experiments.py` karşılaştırmalar · `predict_own.py` el yazısı ön işleme ve tahmin. Tüm grafikler ve epoch bazlı sonuçlar `outputs/` klasöründe. Sonuçlar sabit seed ile birebir tekrarlanabilir.

## Veri ayrımı, model ve hiperparametreler

| Küme | Kaynak | Örnek | Kullanım |
|---|---|---|---|
| Train | MNIST eğitim kümesi | 50.000 | Ağırlıkların güncellenmesi |
| Validation | MNIST eğitim kümesi | 10.000 | En iyi epoch ve ayar seçimi |
| Test | MNIST test kümesi | 10.000 | Sadece eğitim bittikten sonra, bir kez |

- Bölme sabit seed (42) ile yapıldı. **Test verisi hiçbir seçimde kullanılmadı.**
- Normalizasyon: `(x − 0.1307) / 0.3082`. Mean/std **yalnızca train kümesinden** hesaplandı (veri sızıntısı olmaması için).
- Model: `784 → 256 → ReLU → 128 → ReLU → 10` (235.146 parametre), loss: `CrossEntropyLoss`.
- **Temel hiperparametreler:** Adam, lr = 0.001, batch 64, 20 epoch, dropout 0, seed 42, PyTorch varsayılan ağırlık başlatma; augmentation ve weight decay yok. Diğer deneylerde **yalnızca bir ayar** değiştirildi.

## Deney 1 – Temel model

| Train loss | Val loss | Train acc | Val acc (en iyi, epoch 20) | **Test acc** | Eğitim süresi |
|---|---|---|---|---|---|
| 0.0101 | 0.1305 | %99.68 | %97.96 | **%98.27** | 29.9 sn (GPU) |

![Deney 1](outputs/deney1_temel_grafikler.png)

Eğitim öncesi val loss 2.304 ≈ ln(10), yani model gerçekten rastgele başlıyor. Train loss sürekli düşerken val loss ~7. epoch'ta minimuma (0.086) inip yükseliyor: **overfitting**. Val accuracy ise sabit kalıyor, çünkü model yanlış bildiği örneklerde giderek **aşırı emin** hale geliyor. Bu loss'u artırıyor ama accuracy'yi değiştirmiyor.

## Karşılaştırma deneyleri

Temel model 5 farklı seed ile eğitildiğinde val acc **%97.89 ± 0.10**, test acc **%98.02 ± 0.17** çıktı. Bu yüzden ~%0.2'nin altındaki farklar anlamlı sayılmadı.

| Deney | Ayar | Train loss | Val loss | En iyi val acc | Test acc |
|---|---|---|---|---|---|
| 1 Temel | Adam, lr 0.001, 256-128 | 0.0101 | 0.1305 | 0.9796 | 0.9827 |
| 2 Learning rate | lr = 0.1 | 2.3105 | 2.3151 | 0.1129 | 0.1135 |
| | lr = 0.01 | 0.1091 | 0.2825 | 0.9674 | 0.9688 |
| | lr = 0.0001 | 0.0155 | 0.0938 | 0.9734 | 0.9778 |
| 3 Optimizer | SGD + momentum 0.9 (lr 0.01) | 0.0005 | 0.0981 | 0.9784 | 0.9822 |
| | SGD, momentum yok (lr 0.01) | 0.0577 | 0.1065 | 0.9678 | 0.9720 |
| 4 Mimari | 64-32 (52 bin parametre) | 0.0172 | 0.1476 | 0.9722 | 0.9734 |
| | 1024-512 (1.33 milyon) | 0.0187 | 0.1642 | 0.9807 | 0.9797 |
| | 512-256-128-64 (575 bin) | 0.0140 | 0.1316 | 0.9807 | 0.9805 |
| 5 Dropout | 0.2 | 0.0346 | 0.0948 | 0.9797 | 0.9811 |
| | 0.5 | 0.1022 | 0.1043 | 0.9747 | 0.9787 |

Süreler dahil tam tablo: [outputs/ozet_tablo.md](outputs/ozet_tablo.md). Karşılaştırma grafikleri: [learning rate](outputs/deney2_learning_rate_karsilastirma.png) · [optimizer](outputs/deney3_optimizer_karsilastirma.png) · [mimari](outputs/deney4_mimari_karsilastirma.png) · [dropout](outputs/deney5_dropout_karsilastirma.png) · [seed](outputs/deney6_seed_karsilastirma.png)

- **Learning rate en kritik ayar.** lr = 0.1'de loss ln(10)'da takılı kalıyor ve model her girdiye aynı cevabı veriyor (%11). lr = 0.01 gürültülü, lr = 0.0001 düzgün ama yavaş (20 epoch yetmiyor).
- **Optimizer:** SGD + momentum, Adam ile aynı seviyeye ulaşıyor. Momentum olmadan SGD belirgin şekilde yavaş.
- **Mimari:** Küçük model %0.7 geride. 2.4–5.7 kat büyük modeller yalnızca ~%0.1–0.2 kazandırıyor (azalan getiri).
- **Dropout 0.2**, doğruluğu değiştirmeden son epoch val loss'unu 0.13'ten 0.095'e indiriyor (daha az overfitting). 0.5 ise modeli hafif underfit ediyor.

![Learning rate karşılaştırması](outputs/deney2_learning_rate_karsilastirma.png)

## Kendi el yazımla test

![Okul numaram](el_yazisi/numara.jpg)

Ön işleme ([predict_own.py](predict_own.py)): gri tonlama → aydınlatma düzeltme → Otsu eşikleme ve **renk ters çevirme** (MNIST'te siyah zemin, beyaz rakam) → rakamları ayırma → çizgi kalınlaştırma → en-boy oranı korunarak 20×20'ye sığdırma → 28×28 içinde ağırlık merkezine göre ortalama → train mean/std ile normalizasyon. Modele verilen görüntüler: [el_yazisi/islenmis/](el_yazisi/islenmis/)

![Tahminler](outputs/el_yazisi_numara_tahminler.png)

Temel model `24120205078` için `26120205078` tahmin etti: **10/11 (%90.9)** (MNIST testinde %98.27). Tek hata `4`: model %99.97 güvenle `6` diyor.

| Model / ön işleme | Doğru (11 hane) |
|---|---|
| Temel model | 10 |
| Dropout 0.2 | **11** (`4` için %69 olasılık; kararsızlığını belli ediyor) |
| lr = 0.1 | 1 (her rakama `1` diyor) |
| Diğer 12 model | 9–10 (12'si de aynı `4 → 6` hatasını yapıyor) |
| Temel, kalınlaştırma yok | 9 |
| Temel, ağırlık merkezi ortalama yok | 8 |
| Temel, renk ters çevirme yok | **0** (her rakama %100 güvenle `3` diyor) |

## Genel değerlendirme

- Basit bir MLP, MNIST testinde %98'in üzerine çıkıyor, ancak birkaç epoch sonra ezberlemeye başlıyor. En iyi epoch'un validation'a göre seçilmesi, test verisine dokunmadan bunu sınırlıyor.
- Doğru learning rate her şeyden önemli. Mimariyi büyütmenin getirisi az; dropout ise overfitting'i azaltan etkili bir yöntem.
- **Distribution shift:** Kendi el yazımda doğruluk %98'den %91'e düştü. 11 örnek az, ama hatanın nedeni net: üstü açık yazdığım `4`, MNIST'te nadir görülen bir stil ve 15 modelin 12'si aynı hatayı yaptı. MLP pikselleri konumlarıyla öğrendiği için ortalama, ölçek, kalınlık ve renk MNIST'e benzetilmeden doğru çalışmıyor.
- **En önemli bulgu:** Model dağılım dışı girdilerde hata yaparken bile %100 emin, yani softmax güveni tek başına güvenilir değil. Genellemeyi artırmak için veri artırma, regularization veya uzamsal yapıyı kullanan CNN'ler (ödev kapsamı dışında) kullanılabilir.
