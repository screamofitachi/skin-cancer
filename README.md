# 🔬 Skin Cancer Classification — HAM10000

EfficientNet-B0 tabanlı, **ROI segmentasyon + Optuna hiperparametre optimizasyonu + iki aşamalı fine-tuning** içeren cilt lezyonu sınıflandırma projesi. HAM10000 veri seti üzerinde 7 sınıflı sınıflandırma yapılmıştır.

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![Optuna](https://img.shields.io/badge/Optuna-HPO-blueviolet)
![Kaggle](https://img.shields.io/badge/GPU-Tesla%20T4-76B900)

---

## 📋 İçindekiler
- [Proje Özeti](#-proje-özeti)
- [Pipeline](#-pipeline)
- [Veri Seti](#-veri-seti)
- [Sonuçlar](#-sonuçlar)
- [Optuna Hiperparametre Sonuçları](#-optuna-hiperparametre-sonuçları)
- [Sınıf Bazlı Performans](#-sınıf-bazlı-performans)
- [Kullanım](#-kullanım)
- [Dosya Yapısı](#-dosya-yapısı)

---

## 🎯 Proje Özeti

| Özellik | Detay |
|---|---|
| **Model** | EfficientNet-B0 (ImageNet pretrained) |
| **Veri Seti** | HAM10000 (10,015 dermatoskopik görüntü) |
| **Sınıf Sayısı** | 7 (akiec, bcc, bkl, df, mel, nv, vasc) |
| **Görüntü Boyutu** | 224 × 224 |
| **Eğitim Cihazı** | NVIDIA Tesla T4 (14.6 GB VRAM) |
| **Toplam Eğitim Süresi** | ~52 dk Optuna + ~30 dk final eğitim |
| **Test Accuracy** | **76.82%** |
| **Macro F1-Score** | **0.5869** |

---

## 🔄 Pipeline

```
HAM10000 metadata
        ↓
Train/Val/Test split (lesion_id bazlı, stratified — data leakage önleme)
        ↓
ROI Segmentasyon  (Otsu + morfoloji + contour bounding box)
        ↓
ROI Cache (.npy) — Optuna trial'larında tekrar hesaplanmasın diye
        ↓
Optuna HPO (15 trial × 6 epoch, TPESampler + MedianPruner)
        ↓
Final Eğitim — Aşama 1: Classifier head (backbone donuk, 50 epoch + early stopping)
        ↓
Final Eğitim — Aşama 2: Fine-tuning (full unfreeze, düşük LR, 30 epoch)
        ↓
Test + Classification Report + Confusion Matrix
```

### Önemli Tasarım Kararları

- **Lesion-ID bazlı split:** Aynı lezyonun farklı görüntülerinin train/test'e dağılması engellendi (data leakage önleme).
- **ROI cache mekanizması:** ~10K görüntü için ROI bir kez hesaplanıp `.npy` olarak cache'lendi (toplam 1.4 GB). Optuna trial'larının diskten okuması ile büyük hız kazancı.
- **Sqrt class weights (max=3):** İlk denemede ekstrem class weight + WeightedRandomSampler kombinasyonu çoğunluk sınıfını (`nv`) öğrenmeyi bozmuştu. v2'de yumuşatıldı, sampler kaldırıldı.
- **Macro F1 optimize edildi:** Optuna objective'i accuracy değil macro F1 — azınlık sınıflarına da değer verilsin diye.
- **İki aşamalı eğitim:** Önce sadece classifier başlığı eğitildi, sonra tüm ağ düşük LR ile fine-tune edildi.

---

## 📊 Veri Seti

**HAM10000** — 10,015 dermatoskopik görüntü, 7 sınıf

| Sınıf | Açıklama | Görüntü Sayısı | Oran |
|---|---|---:|---:|
| `nv` | Melanocytic Nevi | 6,705 | 66.9% |
| `mel` | Melanoma | 1,113 | 11.1% |
| `bkl` | Benign Keratosis | 1,099 | 11.0% |
| `bcc` | Basal Cell Carcinoma | 514 | 5.1% |
| `akiec` | Actinic Keratoses | 327 | 3.3% |
| `vasc` | Vascular Lesions | 142 | 1.4% |
| `df` | Dermatofibroma | 115 | 1.1% |

**Split:** Train 7,054 | Val 1,464 | Test 1,497

> ⚠️ Veri seti oldukça **dengesiz** — `nv` sınıfı `df`'in ~58 katı kadar görüntüye sahip. Class weights ile bu durum ele alındı.

---

## 🏆 Sonuçlar

### Final Test Performansı

| Metrik | Aşama 1 (Classifier) | **Aşama 2 (Fine-tuned)** |
|---|---:|---:|
| Test Accuracy | 0.7555 | **0.7682** |
| Test Loss | 0.8599 | **0.8255** |
| Precision (macro) | 0.5794 | 0.5742 |
| Recall (macro) | 0.5680 | **0.6034** |
| **F1 (macro)** | 0.5686 | **0.5869** |

**Fine-tuning iyileşmesi: +1.27 puan accuracy, +1.83 puan macro F1**

---

## 🔧 Optuna Hiperparametre Sonuçları

15 trial sonunda bulunan en iyi parametreler:

| Parametre | Değer |
|---|---:|
| Learning Rate | 4.69 × 10⁻⁴ |
| Dropout | 0.212 |
| Weight Decay | 3.49 × 10⁻⁵ |
| Batch Size | 64 |
| **Best Val F1** | **0.5568** |

Optuna toplam süresi: **51.9 dakika** (Tesla T4)

---

## 📈 Sınıf Bazlı Performans

| Sınıf | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| akiec | 0.434 | 0.451 | 0.442 | 51 |
| bcc | 0.578 | 0.623 | 0.600 | 77 |
| bkl | 0.526 | 0.580 | 0.552 | 157 |
| df | 0.296 | 0.364 | 0.327 | 22 |
| mel | 0.500 | 0.619 | 0.553 | 168 |
| **nv** | **0.923** | **0.860** | **0.890** | 1000 |
| vasc | 0.762 | 0.727 | 0.744 | 22 |
| **macro avg** | 0.574 | 0.603 | **0.587** | 1497 |
| weighted avg | 0.788 | 0.768 | 0.776 | 1497 |

### Yorum
- **`nv` (çoğunluk sınıfı):** F1 = 0.89 ile çok iyi öğrenilmiş.
- **`vasc` (azınlık):** Sadece 22 örnek olmasına rağmen F1 = 0.74 — class weights başarılı.
- **`df` (en zor sınıf):** Hem örnek sayısı az (22) hem de görsel olarak `bkl`/`nv` ile karışıyor → F1 = 0.33.
- **`mel` (klinik açıdan en kritik):** Recall = 0.62 — melanoma yakalama oranı geliştirilebilir.

---

## 🚀 Kullanım

### Gereksinimler
```bash
pip install torch torchvision optuna opencv-python pandas scikit-learn matplotlib
```

### Kaggle Üzerinde Çalıştırma
1. Notebook'u Kaggle'a yükleyin.
2. `kmader/skin-cancer-mnist-ham10000` dataset'ini ekleyin.
3. GPU (Tesla T4 önerilir) ve internet erişimi açık olsun.
4. Tüm hücreleri sırayla çalıştırın.

### Çıktılar
Notebook çalıştırıldığında `/kaggle/working/` altında oluşur:
- `best_efficientnet_b0.pth` — Eğitilmiş model ağırlıkları
- `probabilities_efficientnet_b0.npy` — Test set softmax çıktıları (1497 × 7)
- `test_labels_efficientnet_b0.npy` — Test set gerçek etiketleri
- `confusion_matrix_efficientnet_b0.jpg` — Karışıklık matrisi
- `optuna_study.pkl` — Optuna study objesi

---

## 📁 Dosya Yapısı

```
skin-cancer/
├── notebook448638b0d6 (1).ipynb   # Ana notebook (uçtan uca pipeline)
└── README.md                       # Bu dosya
```

---

## 🔬 Geliştirme Önerileri

- [ ] **Daha büyük model:** EfficientNet-B3/B4 ile daha yüksek accuracy
- [ ] **Test-Time Augmentation (TTA):** Inference'ta birden fazla augmente edilmiş tahminin ortalaması
- [ ] **Ensemble:** EfficientNet + ResNet + ViT kombinasyonu
- [ ] **Focal Loss:** Class imbalance için cross-entropy yerine focal loss
- [ ] **Mixup / CutMix:** Augmentation'ı güçlendirmek için
- [ ] **Daha fazla Optuna trial:** 15 → 50+ (süre izin verirse)
- [ ] **Melanoma recall odaklı eğitim:** Klinik açıdan en kritik sınıf

---

## 📚 Kaynaklar

- **HAM10000 Dataset:** Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, *Sci. Data* 5, 180161 (2018).
- **EfficientNet:** Tan, M. & Le, Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, *ICML* (2019).
- **Optuna:** Akiba, T. et al. Optuna: A Next-generation Hyperparameter Optimization Framework, *KDD* (2019).

---

## 📝 Lisans

Bu proje akademik amaçla geliştirilmiştir. HAM10000 veri setinin kendi lisans koşullarına uyulmalıdır.
