## 🩺 Chest X-ray Analysis Using Deep Learning

## 🧠 Segmentation + Multi-Label Classification System

This project presents an end-to-end deep learning system for analyzing chest X-ray images. It combines:

🔷 Medical Image Segmentation using U-Net
🔷 Multi-label Disease Classification using ResNet18

The system is designed to assist in medical decision-making by highlighting disease regions and predicting multiple thoracic conditions.

## 📌 Project Objective

The goal of this project is to:

Detect and segment abnormal regions in chest X-ray images using bounding box annotations.
Classify multiple diseases present in a single image.
Provide visual explanations using:
Segmentation masks
Heatmaps
Bounding boxes

## 🏗️ System Architecture

1️⃣ Segmentation Model
Architecture: U-Net
Encoder: ResNet18
Input: Grayscale chest X-ray images
Output: Binary lesion mask

Loss Function:

BCEWithLogitsLoss
Dice Loss (to handle class imbalance)

Metrics:

Dice Score
IoU (Intersection over Union)

2️⃣ Classification Model
Architecture: ResNet18 (modified for grayscale input)
Task: Multi-label classification (4 disease classes)
Output: Sigmoid logits per class

Loss Function:

BCEWithLogitsLoss with class imbalance weighting

Metrics:

F1 Score (Micro)
ROC-AUC
Accuracy
📂 Dataset

## 📊 Dataset Source

This project uses a dataset from Kaggle:

👉 https://www.kaggle.com/datasets/nih-chest-xrays/data

The dataset includes chest X-ray images and metadata:

Data_Entry_2017.csv
Bounding box annotations (preprocessed_BBox_f_2.csv)
Preprocessed image folders:
preprocessed_images_001 → 012
🧪 Data Splitting Strategy

To avoid data leakage, splitting is performed at patient level:

🟢 Training set: 70%
🟡 Validation set: 15%
🔴 Test set: 15%
🧠 Technologies Used

PyTorch
segmentation_models_pytorch
Torchvision
OpenCV
Scikit-learn
Matplotlib
PIL
NumPy

## ⚙️ Training Configuration

🔷 Segmentation
Epochs: 50
Batch Size: 8
Learning Rate: 2e-4
Optimizer: Adam
Scheduler: ReduceLROnPlateau
Mixed Precision Training (AMP)

🔷 Classification
Epochs: 50
Batch Size: 32
Learning Rate: 1e-4
Optimizer: AdamW
Handles class imbalance using pos_weight
Dynamic threshold tuning for best F1-score

## 📊 Evaluation Metrics

Segmentation
Dice Score (primary metric)
IoU Score
Classification
Micro F1-score
ROC-AUC
Accuracy
Per-class F1-score analysis

## 📈 Outputs & Visualizations

The project generates:

📉 Training & validation loss curves
📊 Dice / IoU / F1 / AUC curves
🖼️ Segmentation masks vs predictions
🔲 Bounding box predictions
🔥 Heatmap visualizations
💾 Model Checkpoints

Saved models include:

seg_best_checkpoint.pth
cls_best_checkpoint.pth

Additionally:

Per-epoch checkpoints
Training metrics history (pickle files)

## 🚀 How to Run the Project

1️⃣ Install dependencies
pip install torch torchvision segmentation-models-pytorch scikit-learn matplotlib opencv-python pillow pandas
2️⃣ Run Segmentation Training
python segmentation.py
3️⃣ Run Classification Training
python classification.py

## 📁 Project Structure

Graduation_Project/
│
├── Data_Entry_2017.csv
├── preprocessed_BBox_f_2.csv
├── preprocessed_images_001 → 012/
│
├── segmentation.py
├── classification.py
│
├── seg_best_checkpoint.pth
├── cls_best_checkpoint.pth
│
├── seg_metrics/
├── cls_metrics/
├── seg_predictions/

## 🧩 Key Features

✔ End-to-end medical imaging pipeline
✔ Combines segmentation and classification
✔ Transfer learning with ResNet18
✔ Strong handling of class imbalance
✔ Data augmentation for medical robustness
✔ Mixed precision training for performance
✔ Automatic checkpointing and resuming

## 🏥 Medical Use Case

This system can support:

Early disease detection
Radiology decision support systems
Highlighting abnormal regions in X-rays
Reducing radiologist workload
⚠️ Disclaimer

This project is intended for educational and research purposes only and should not be used as a replacement for professional medical diagnosis.

## 👨‍💻 Future Improvements

Replace U-Net with Attention U-Net or TransUNet
Integrate Vision Transformers (ViT)
Deploy as a web application (Flask / Streamlit)
Build real-time inference API
Train on larger datasets (e.g., NIH ChestX-ray14)

## 👨‍💻 Author
Maged Hujira

## 📫 Contact Me

[![WhatsApp](https://img.shields.io/badge/WhatsApp-25D366?style=for-the-badge&logo=whatsapp)](https://wa.me/201234567890)

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram)](https://t.me/username)