# ============================================================
# كود التصنيف - Classification Only
# مُعدَّل للتدريب المحلي (Local Machine)
# ============================================================

import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle
import glob
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

# ============================================================
# الإعدادات العامة
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------
# 🔧 عدّل هذا المسار حسب جهازك:
#   Windows : PROJECT_PATH = r"C:\Users\YourName\chest_xray"
#   Linux   : PROJECT_PATH = "/home/username/chest_xray"
#   Mac     : PROJECT_PATH = "/Users/username/chest_xray"
# ---------------------------------------------------------------
PROJECT_PATH = r"D:\شبكات عصبونية\maged\__pycache__\VR\Graduation_Project"   # <-- غيّر هذا

#csv_path = os.path.join(PROJECT_PATH, "Data_Entry_2017.csv") 
csv_path = os.path.join(PROJECT_PATH, "Data_Entry_2017_single_label_dedup.csv")
img_dirs = [
    os.path.join(PROJECT_PATH, f"preprocessed_images_{str(i).zfill(3)}")
    for i in range(1, 13)
]

# Hyperparameters
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 0.0001
#LEARNING_RATE =0.00005
NUM_CLASSES   = 4

# ---------------------------------------------------------------
# 🔧 num_workers:
#   Windows : استخدم 0 أو 2 (قد يحدث crash مع قيم أعلى)
#   Linux/Mac: استخدم 4 أو 8
# ---------------------------------------------------------------
NUM_WORKERS = 0 # <-- غيّر حسب نظامك

# مسارات الحفظ
checkpoint_path = os.path.join(PROJECT_PATH, "cls_best_checkpoint.pth")
metrics_file    = os.path.join(PROJECT_PATH, "cls_metrics_history.pkl")
os.makedirs(os.path.join(PROJECT_PATH, "cls_checkpoints"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_PATH, "cls_metrics"),     exist_ok=True)

# ============================================================
# قراءة البيانات وتقسيمها بشكل صحيح (patient-level split)
# ============================================================
df = pd.read_csv(csv_path)
df['Finding Labels'] = df['Finding Labels'].fillna('No Finding')

# نقسم على مستوى المريض لتجنب data leakage
patients = df['Patient ID'].unique()
train_patients, temp_patients = train_test_split(patients, test_size=0.30, random_state=42)
val_patients,  test_patients  = train_test_split(temp_patients, test_size=0.50, random_state=42)

train_df = df[df['Patient ID'].isin(train_patients)].reset_index(drop=True)
val_df   = df[df['Patient ID'].isin(val_patients)].reset_index(drop=True)
test_df  = df[df['Patient ID'].isin(test_patients)].reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

"""ALL_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]"""
ALL_CLASSES = [
    'Atelectasis',
    'Effusion', 
    'Infiltration',
    'No Finding'
]

# ============================================================
# Dataset - Classification فقط (بدون masks)
# ============================================================
class ChestXrayClassificationDataset(Dataset):
    """
    Dataset مخصص للتصنيف فقط.
    - يقرأ الصورة الرمادية (Grayscale)
    - يحوّل التسميات إلى multi-hot vector
    - لا حاجة لأي BBox أو mask هنا
    """

    def __init__(self, df, img_dirs, transform=None):
        self.df        = df.copy()
        self.img_dirs  = img_dirs
        self.transform = transform

        # ابحث عن مسار كل صورة عبر المجلدات المتعددة
        self.df['img_path'] = self.df['Image Index'].apply(self._find_image)
        # احتفظ فقط بالصور الموجودة فعلاً
        self.df = self.df[self.df['img_path'].notnull()].reset_index(drop=True)

    def _find_image(self, img_name):
        for d in self.img_dirs:
            path = os.path.join(d, img_name)
            if os.path.exists(path):
                return path
        return None  # الصورة غير موجودة في أي مجلد

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # تحميل الصورة كـ grayscale
        image = Image.open(row['img_path']).convert("L")
        if self.transform:
            image = self.transform(image)

        # بناء multi-hot label vector (طول 15)
        labels = torch.zeros(len(ALL_CLASSES))
        for i, cls in enumerate(ALL_CLASSES):
            if cls in str(row['Finding Labels']).split('|'):
                labels[i] = 1.0

        return image, labels

# ============================================================
# Transforms
# ============================================================
"""# للتدريب: augmentation بسيط + تطبيع
train_transform = transforms.Compose([
    transforms.ToTensor(),                           # [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])      # [-1,1]
])

# للتحقق والاختبار: بدون augmentation
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
"""
# ------------------- Transform أثناء التدريب -------------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])   # هنا فقط ✅
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])   # بدون augmentation للـ val
])
# ============================================================
# DataLoaders
# ============================================================
train_dataset = ChestXrayClassificationDataset(train_df, img_dirs, train_transform)
val_dataset   = ChestXrayClassificationDataset(val_df,   img_dirs, val_transform)
test_dataset  = ChestXrayClassificationDataset(test_df,  img_dirs, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))

# ============================================================
# النموذج - Classification فقط
# ============================================================
class ChestClassifier(nn.Module):
    """
    نموذج تصنيف متعدد التسميات لأشعة الصدر.
    يستخدم ResNet18 مع تعديل أول طبقة لقبول صورة رمادية (1 channel).
    """

    def __init__(self, num_classes=15):
        super().__init__()
        # نحمّل ResNet18 مع ImageNet weights (أفضل من البداية من الصفر)
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # تعديل أول طبقة: 3 channels -> 1 channel
        # نحسب متوسط وزن القنوات الثلاث ليكون وزن القناة الواحدة
        original_weight = backbone.conv1.weight.data  # shape: (64, 3, 7, 7)
        new_weight      = original_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)

        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data = new_weight  # نقل الأوزان المعدّلة

        # استخدم كل ResNet ما عدا الطبقة الأخيرة (fc)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 512, 1, 1)

        # رأس التصنيف
        """ self.classifier = nn.Sequential(
            nn.Flatten(),          # (B, 512)
            nn.Dropout(p=0.25),     # تقليل overfitting
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)  # logits (بدون sigmoid هنا)
        )"""
        self.classifier = nn.Sequential(
            nn.Flatten(),          # (B, 512)
            nn.Dropout(p=0.25),     # تقليل overfitting
            nn.Linear(512, num_classes)  # logits (بدون sigmoid هنا)
        )

    def forward(self, x):
        features = self.features(x)   # (B, 512, 1, 1)
        return self.classifier(features)  # (B, num_classes)

model = ChestClassifier(num_classes=NUM_CLASSES).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# Loss, Optimizer, Scheduler
# ============================================================
# BCEWithLogitsLoss مناسب لـ multi-label classification
#criterion = nn.BCEWithLogitsLoss()
# ============================================================
# Loss مع أوزان لمعالجة Class Imbalance
# ============================================================
counts = [2643, 1984, 5466, 14116]  # ترتيب ALL_CLASSES
total = sum(counts)
pos_weight = torch.tensor(
    [(total - c) / c for c in counts], dtype=torch.float32
).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# يخفض LR بمقدار 0.5 إذا لم يتحسن الـ validation F1 لـ 3 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5
)

# ============================================================
# استئناف من checkpoint (إن وجد)
# ============================================================
ckpts = glob.glob(os.path.join(PROJECT_PATH, "cls_checkpoints", "checkpoint_epoch_*.pth"))
if ckpts:
    latest_ckpt = max(ckpts, key=os.path.getctime)
    ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch']
    print(f"✅ Resumed from epoch {start_epoch}")
else:
    start_epoch = 0

# تحميل أفضل F1 سابق
best_f1 = 0.0
if os.path.exists(checkpoint_path):
    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    best_f1 = best_ckpt.get('val_f1', 0.0)
    print(f"⭐ Best F1 so far: {best_f1:.4f}")

# تحميل مقاييس سابقة
if os.path.exists(metrics_file):
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
else:
    metrics = {k: [] for k in ['train_loss', 'train_f1', 'train_auc', 'train_acc',
                                 'val_loss',   'val_f1',   'val_auc',   'val_acc']}

# ============================================================
# حلقة التدريب
# ============================================================
# Mixed Precision: يعمل فقط مع CUDA، وإلا يُعطَّل تلقائياً
use_amp = device.type == 'cuda'
scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

early_stop_patience = 10
no_improve_counter  = 0
MIN_DELTA           = 0.0001

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_start = time.time()

    # ---------- مرحلة التدريب ----------
    model.train()
    running_loss = 0.0
    all_targets, all_outputs = [], []

    for step, (images, labels) in enumerate(train_loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        all_targets.append(labels.detach().cpu())
        all_outputs.append(logits.detach().cpu())

        if step % 100 == 0:
            print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f}")

    # حساب مقاييس التدريب
    train_loss = running_loss / len(train_loader)
    all_targets = torch.cat(all_targets).numpy()
    all_probs   = torch.sigmoid(torch.cat(all_outputs)).numpy()
    all_preds   = all_probs > 0.5

    train_f1  = f1_score(all_targets, all_preds, average='micro', zero_division=0)
    train_auc = roc_auc_score(all_targets, all_probs, average='micro')
    train_acc = accuracy_score(all_targets.flatten(), all_preds.flatten())

    # ---------- مرحلة التحقق ----------
    model.eval()
    val_loss_sum = 0.0
    val_targets, val_outputs = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            val_loss_sum += criterion(logits, labels).item()
            val_targets.append(labels.cpu())
            val_outputs.append(logits.cpu())

    val_loss    = val_loss_sum / len(val_loader)
    val_targets = torch.cat(val_targets).numpy()
    val_probs   = torch.sigmoid(torch.cat(val_outputs)).numpy()

    # البحث عن أفضل threshold (بدلاً من 0.5 الثابتة)
    best_thresh, best_val_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        score = f1_score(val_targets, val_probs > t, average='micro', zero_division=0)
        if score > best_val_f1:
            best_val_f1, best_thresh = score, t

    val_preds   = val_probs > best_thresh
    val_auc     = roc_auc_score(val_targets, val_probs, average='micro')
    val_acc     = accuracy_score(val_targets.flatten(), val_preds.flatten())

    # تحديث scheduler
    scheduler.step(best_val_f1)

    # حفظ المقاييس
    for key, val in zip(
        ['train_loss','train_f1','train_auc','train_acc','val_loss','val_f1','val_auc','val_acc'],
        [train_loss, train_f1, train_auc, train_acc, val_loss, best_val_f1, val_auc, val_acc]
    ):
        metrics[key].append(val)

    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)

    # طباعة النتائج
    elapsed = time.time() - epoch_start
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] ({elapsed:.0f}s)")
    print(f"  Train | Loss: {train_loss:.4f} | F1: {train_f1:.4f} | AUC: {train_auc:.4f} | Acc: {train_acc:.4f}")
    print(f"  Val   | Loss: {val_loss:.4f} | F1: {best_val_f1:.4f} (thresh={best_thresh:.2f}) | AUC: {val_auc:.4f} | Acc: {val_acc:.4f}")

    # حفظ checkpoint للـ epoch الحالي
    ckpt_save_path = os.path.join(PROJECT_PATH, "cls_checkpoints", f"checkpoint_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1':  best_val_f1,
        'val_loss': val_loss,
    }, ckpt_save_path)

    # حفظ أفضل نموذج
    if best_val_f1 > best_f1 + MIN_DELTA:
        best_f1 = best_val_f1
        no_improve_counter = 0
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1':  best_f1,
            'val_auc': val_auc,
            'val_acc': val_acc,
            'best_thresh': best_thresh,
        }, checkpoint_path)
        print("  ✅ Best model saved!")
    else:
        no_improve_counter += 1
        print(f"  ⏸ No improvement ({no_improve_counter}/{early_stop_patience})")
        if no_improve_counter >= early_stop_patience:
            print("  🛑 Early stopping!")
            break

# ============================================================
# التقييم على مجموعة الاختبار
# ============================================================
print("\nLoading best model for test evaluation...")
best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(best_ckpt['model_state_dict'])
best_thresh = best_ckpt.get('best_thresh', 0.5)
model.eval()

test_targets, test_outputs = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        test_outputs.append(model(images).cpu())
        test_targets.append(labels.cpu())

test_targets = torch.cat(test_targets).numpy()
test_probs   = torch.sigmoid(torch.cat(test_outputs)).numpy()
test_preds   = test_probs > best_thresh

print("\n===== Test Evaluation =====")
print(f"  F1  (micro): {f1_score(test_targets, test_preds, average='micro', zero_division=0):.4f}")
print(f"  AUC (micro): {roc_auc_score(test_targets, test_probs, average='micro'):.4f}")
print(f"  Acc:         {accuracy_score(test_targets.flatten(), test_preds.flatten()):.4f}")

# طباعة نتائج كل فئة على حدة
print("\nPer-class F1:")
class_f1 = f1_score(test_targets, test_preds, average=None, zero_division=0)
for cls, score in zip(ALL_CLASSES, class_f1):
    print(f"  {cls:<22}: {score:.4f}")

# ============================================================
# رسم منحنيات التدريب
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Classification Training Metrics", fontsize=14)
epochs_range = range(1, len(metrics['train_loss']) + 1)

axes[0,0].plot(epochs_range, metrics['train_loss'], 'bo-', label='Train')
axes[0,0].plot(epochs_range, metrics['val_loss'],   'ro-', label='Val')
axes[0,0].set_title("Loss"); axes[0,0].legend(); axes[0,0].set_xlabel("Epoch")

axes[0,1].plot(epochs_range, metrics['train_f1'], 'go-', label='Train')
axes[0,1].plot(epochs_range, metrics['val_f1'],   'mo-', label='Val')
axes[0,1].set_title("F1 Score"); axes[0,1].legend(); axes[0,1].set_xlabel("Epoch")

axes[1,0].plot(epochs_range, metrics['train_acc'], 'co-', label='Train')
axes[1,0].plot(epochs_range, metrics['val_acc'],   'yo-', label='Val')
axes[1,0].set_title("Accuracy"); axes[1,0].legend(); axes[1,0].set_xlabel("Epoch")

axes[1,1].plot(epochs_range, metrics['train_auc'], 'ko-', label='Train')
axes[1,1].plot(epochs_range, metrics['val_auc'],   'co-', label='Val')
axes[1,1].set_title("AUC"); axes[1,1].legend(); axes[1,1].set_xlabel("Epoch")

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_PATH, "cls_metrics", "classification_metrics.png"), dpi=150)
plt.show()
print("✅ Metrics plot saved.")
