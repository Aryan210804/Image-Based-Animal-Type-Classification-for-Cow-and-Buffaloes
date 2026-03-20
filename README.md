# 🐄🐃 Image-Based Animal Type Classification for Cow and Buffaloes
## Pretrained Model — Feature Extraction (MobileNetV2)

---

## 📌 Topic
**Pretrained Model - Feature Extraction**

Instead of training a CNN from scratch, this project uses **MobileNetV2** (pretrained on ImageNet) as a frozen feature extractor. Only a small classification head is trained on top — making it faster, more accurate, and less prone to overfitting.

---

## 🧠 What is Feature Extraction?

| Step | What Happens |
|---|---|
| 1 | Load MobileNetV2 pretrained on 1.28 million ImageNet images |
| 2 | Remove the final 1000-class classification head (`include_top=False`) |
| 3 | Freeze all backbone weights (`base_model.trainable = False`) |
| 4 | Feed image → get 1280-d feature vector out |
| 5 | Train only a small Dense head on top of those features |

> The backbone already knows edges, textures, shapes and patterns from ImageNet.
> We just teach the small head to distinguish Cow vs Buffalo using those features.

---

## 📁 Dataset

| Class | Images |
|---|---|
| 🐄 Cow | 515 |
| 🐃 Buffalo | 515 |
| **Total** | **1030** |

- **Split:** 80% Training / 20% Testing
- **Train:** 824 images (412 Cow + 412 Buffalo)
- **Test:** 206 images (103 Cow + 103 Buffalo)

---

## 🏗️ Model Architecture
```
Input Image (224 × 224 × 3)
        ↓
MobileNetV2 Backbone — FROZEN (ImageNet weights)
  • Already knows: edges, textures, fur, shapes
        ↓
GlobalAveragePooling2D → 1280-d Feature Vector
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.3)
        ↓
Dense(1, Sigmoid)
        ↓
  Cow 🐄  or  Buffalo 🐃
```

| Layer | Details |
|---|---|
| Backbone | MobileNetV2 (frozen, ImageNet weights) |
| Pooling | GlobalAveragePooling2D → 1280-d vector |
| Hidden Layer | Dense(128, activation='relu') |
| Regularisation | Dropout(0.3) |
| Output Layer | Dense(1, activation='sigmoid') |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |

---

## 📊 Trainable Parameters

| | Parameters |
|---|---|
| Backbone (frozen) | ~2,257,984 — never updated |
| Classification Head | ~164,000 — only these train |
| **Total** | **~2,421,984** |

---

## 📈 Results

| Model | Validation Accuracy |
|---|---|
| Scratch CNN (original project) | ~84% |
| **MobileNetV2 Feature Extraction** | **~95%+** |

---

## 🖼️ What the PCA Plot Shows

After extracting 1280-d feature vectors from all test images, we reduced them to 2D using **PCA (Principal Component Analysis)** and plotted them.

- Each point = one image
- Green points = Cow
- Blue points = Buffalo
- Separated clusters = MobileNetV2 features cleanly distinguish the two animals

---

## 🛠️ Libraries Used

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Model building and training |
| MobileNetV2 | Pretrained feature extractor |
| NumPy | Array operations |
| Matplotlib | Plotting graphs and images |
| Scikit-learn | PCA visualisation |
| ImageDataGenerator | Data loading and augmentation |

---

## ⚙️ Data Augmentation (Training Only)

| Technique | Value |
|---|---|
| Rotation | ±20° |
| Zoom | 20% |
| Shear | 20% |
| Horizontal Flip | True |
| Preprocessing | preprocess_input (scales to [-1, 1]) |

---

## 🔁 Callbacks Used

| Callback | Purpose |
|---|---|
| EarlyStopping | Stops training if val_accuracy doesn't improve for 5 epochs |
| ReduceLROnPlateau | Reduces learning rate if val_loss is stuck for 3 epochs |

---

## ▶️ How to Run

1. Open `classification.ipynb` in **Google Colab**
2. Mount your Google Drive
3. Make sure your dataset is at:
```
/content/drive/MyDrive/Colab Notebooks/datasets/
├── Cow/
│   ├── Cow_1.jpg
│   └── ...
└── Buffalo/
    ├── Buffalo_1.jpg
    └── ...
```
4. Run all cells in order

---

## 📂 Project Structure
```
📦 repository
 ├── 📓 classification.ipynb   ← main notebook
 ├── 📄 README.md              ← this file
 └── 📄 requirements.txt       ← dependencies
```

---

## 📦 Requirements
```
tensorflow
numpy
matplotlib
scikit-learn
Pillow
```

---

## 👤 Author
**Aryan** — [GitHub Profile](https://github.com/Aryan210804)
