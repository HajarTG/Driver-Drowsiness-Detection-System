# 🚗 Driver Drowsiness Detection System

Système intelligent de détection de la somnolence au volant basé sur le Deep Learning et la vision par ordinateur.

---

## 📌 Aperçu

Ce projet vise à améliorer la sécurité routière en détectant automatiquement l'état de somnolence d'un conducteur à partir d'une webcam.  
Deux approches sont comparées :

| Approche | Paramètres | Description |
|---|---|---|
| **CNN from Scratch** | ~9.8M | Architecture CNN personnalisée entraînée de zéro |
| **MobileNetV2 + Transfer Learning** | ~2.4M | MobileNetV2 pré-entraîné (ImageNet) + tête de classification |

---

## 📊 Résultats

| Modèle | Accuracy | Recall (Drowsy) | Paramètres |
|--------|----------|-----------------|------------|
| CNN from Scratch | **98.99%** | 0.98 | 9.8M |
| MobileNetV2 | **97.01%** | 0.94 | 2.4M |

---

## 🗂️ Structure du projet

```
Driver-Drowsiness-Detection-System/
├── models/
│   ├── cnn_model.py          # CNN from scratch (~9.8M params)
│   └── mobilenet_model.py    # MobileNetV2 transfer learning (~2.4M params)
├── utils/
│   ├── data_loader.py        # Chargement et augmentation des données
│   ├── face_detector.py      # Détection des visages/yeux (Haar Cascades)
│   └── metrics.py            # Métriques et visualisations
├── notebooks/
│   ├── 01_CNN_Training.ipynb           # Entraînement CNN
│   └── 02_MobileNetV2_Training.ipynb   # Entraînement MobileNetV2
├── data/                     # Dataset (non inclus — voir ci-dessous)
│   ├── train/
│   │   ├── Awake/
│   │   └── Drowsy/
│   ├── val/
│   │   ├── Awake/
│   │   └── Drowsy/
│   └── test/
│       ├── Awake/
│       └── Drowsy/
├── saved_models/             # Poids entraînés (générés après training)
├── results/                  # Courbes, matrices de confusion, ROC
├── train.py                  # Script d'entraînement
├── evaluate.py               # Script d'évaluation et comparaison
├── detect.py                 # Détection temps réel (webcam)
└── requirements.txt
```

---

## 🚀 Installation

```bash
git clone https://github.com/HajarTG/Driver-Drowsiness-Detection-System.git
cd Driver-Drowsiness-Detection-System
pip install -r requirements.txt
```

> **Note** : `dlib` nécessite CMake. Sur Ubuntu : `sudo apt-get install cmake`.  
> Une alternative sans `dlib` est possible en utilisant uniquement les Haar Cascades d'OpenCV.

---

## �� Dataset

Le projet est compatible avec :

- **[MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset)** — images d'yeux (ouverts/fermés)
- **[Driver Drowsiness Dataset (Kaggle)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-v0)** — visages labellisés

Placez les données dans le dossier `data/` en respectant la structure :
```
data/
├── train/Awake/   ← images de conducteurs éveillés
├── train/Drowsy/  ← images de conducteurs somnolents
├── val/Awake/
├── val/Drowsy/
├── test/Awake/
└── test/Drowsy/
```

---

## 🏋️ Entraînement

Entraîner les deux modèles :
```bash
python train.py --model both --data_dir data/ --epochs 50
```

Entraîner uniquement le CNN :
```bash
python train.py --model cnn --data_dir data/ --epochs 50
```

Entraîner MobileNetV2 avec fine-tuning :
```bash
python train.py --model mobilenet --data_dir data/ --epochs 30 --fine_tune
```

---

## �� Évaluation

```bash
python evaluate.py --data_dir data/
```

Génère dans `results/` :
- Matrices de confusion
- Courbes ROC
- Graphique de comparaison des modèles

---

## 🎥 Détection en temps réel

```bash
# Avec le modèle CNN
python detect.py --model_path saved_models/cnn_final.keras

# Avec MobileNetV2 (entrée 96×96)
python detect.py --model_path saved_models/mobilenet_final.keras --image_size 96

# Test sur une image unique
python detect.py --model_path saved_models/cnn_final.keras --image path/to/image.jpg
```

Appuyez sur **`q`** pour quitter la fenêtre de détection.

---

## 🧠 Architecture CNN from Scratch

```
Input (64×64×3)
  ↓ Block 1 : Conv2D(32) × 2 + BN + MaxPool + Dropout(0.25)
  ↓ Block 2 : Conv2D(64) × 2 + BN + MaxPool + Dropout(0.25)
  ↓ Block 3 : Conv2D(128) × 2 + BN + MaxPool + Dropout(0.25)
  ↓ Block 4 : Conv2D(256) × 2 + BN + MaxPool + Dropout(0.25)
  ↓ Block 5 : Conv2D(512) × 2 + BN + GAP + Dropout(0.4)
  ↓ Dense(4096) + BN + Dropout(0.5)
  ↓ Dense(1024) + BN + Dropout(0.5)
  ↓ Output Dense(1, sigmoid)
```

## 🧠 Architecture MobileNetV2 + Transfer Learning

```
Input (96×96×3)
  ↓ Preprocessing (pixels → [-1, 1])
  ↓ MobileNetV2 backbone (ImageNet, frozen puis fine-tuné)
  ↓ GlobalAveragePooling2D
  ↓ Dense(256) + BN + Dropout(0.5)
  ↓ Dense(128) + BN + Dropout(0.3)
  ↓ Output Dense(1, sigmoid)
```

**Phase 1** : backbone gelé, seule la tête est entraînée.  
**Phase 2** : les couches > 100 du backbone sont dégelées (fine-tuning, lr=1e-5).

---

## ⚙️ Algorithme de détection temps réel

1. Capture de la frame via la webcam
2. Détection du visage (Haar Cascade)
3. Détection des yeux dans la ROI du visage
4. Extraction et normalisation de chaque œil (64×64 ou 96×96)
5. Inférence par le modèle CNN/MobileNetV2
6. Maintien d'une fenêtre glissante de scores
7. Alerte sonore si le score moyen dépasse le seuil pendant `--alert_frames` frames consécutives

---

## 🛠️ Paramètres CLI

### `train.py`
| Argument | Défaut | Description |
|---|---|---|
| `--model` | `both` | `cnn`, `mobilenet`, ou `both` |
| `--data_dir` | `data/` | Répertoire du dataset |
| `--epochs` | `30` | Nombre d'époques max |
| `--batch_size` | `32` | Taille du batch |
| `--fine_tune` | `True` | Fine-tuning MobileNetV2 |
| `--class_weights` | `False` | Pondération des classes |

### `detect.py`
| Argument | Défaut | Description |
|---|---|---|
| `--model_path` | `saved_models/cnn_final.keras` | Chemin du modèle |
| `--image_size` | `64` | Taille d'entrée du modèle |
| `--threshold` | `0.5` | Seuil de décision |
| `--alert_frames` | `15` | Frames consécutives avant alerte |
| `--camera` | `0` | Index de la webcam |
| `--image` | — | Image unique (mode test) |

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `notebooks/01_CNN_Training.ipynb` | Exploration des données + entraînement CNN |
| `notebooks/02_MobileNetV2_Training.ipynb` | Transfer learning + fine-tuning + comparaison |

Compatible Google Colab ☁️ et exécution locale 💻.

---

## 📦 Dépendances principales

- `tensorflow >= 2.10`
- `opencv-python >= 4.7`
- `scikit-learn >= 1.2`
- `mediapipe >= 0.10`
- `matplotlib`, `seaborn`, `numpy`, `pandas`

---

## 📄 Licence

Ce projet est distribué sous licence MIT.
