# ============================================================
# كود التجزئة - Segmentation Only
# مُعدَّل للتدريب المحلي (Local Machine)
# ============================================================

import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2 as T2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import numpy as np
import pickle
import glob
import cv2
from segmentation_models_pytorch import Unet
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
PROJECT_PATH  = r"D:\شبكات عصبونية\maged\__pycache__\VR\Graduation_Project"   # <-- غيّر هذا

csv_path      = os.path.join(PROJECT_PATH, "Data_Entry_2017.csv")
bbox_csv_path = os.path.join(PROJECT_PATH, "preprocessed_BBox_f_2.csv")
img_dirs = [
    os.path.join(PROJECT_PATH, f"preprocessed_images_{str(i).zfill(3)}")
    for i in range(1, 13)
]

BATCH_SIZE    = 8
NUM_EPOCHS    = 50
LEARNING_RATE = 2e-4

# ---------------------------------------------------------------
# 🔧 num_workers:
#   Windows : استخدم 0 أو 2 (قد يحدث crash مع قيم أعلى)
#   Linux/Mac: استخدم 4 أو 8
# ---------------------------------------------------------------
NUM_WORKERS = 0  # <-- غيّر حسب نظامك

checkpoint_path = os.path.join(PROJECT_PATH, "seg_best_checkpoint.pth")
metrics_file    = os.path.join(PROJECT_PATH, "seg_metrics_history.pkl")
os.makedirs(os.path.join(PROJECT_PATH, "seg_checkpoints"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_PATH, "seg_metrics"),     exist_ok=True)
os.makedirs(os.path.join(PROJECT_PATH, "seg_predictions"), exist_ok=True)

# ============================================================
# قراءة البيانات
# ============================================================
df = pd.read_csv(csv_path)
df['Finding Labels'] = df['Finding Labels'].fillna('No Finding')

# نبقي فقط الصور التي لها BBox (لأن بدونها لا يوجد ground truth للتجزئة)
bbox_df = pd.read_csv(bbox_csv_path)
bbox_df.columns = [c.strip() for c in bbox_df.columns]

# الصور التي لها annotations فقط
images_with_bbox = set(bbox_df['Image Index'].unique())
df_seg = df[df['Image Index'].isin(images_with_bbox)].reset_index(drop=True)
print(f"Images with BBox annotations: {len(df_seg)}")

# تقسيم على مستوى المريض
patients = df_seg['Patient ID'].unique()
train_patients, temp_patients = train_test_split(patients, test_size=0.30, random_state=42)
val_patients,  test_patients  = train_test_split(temp_patients, test_size=0.50, random_state=42)

train_df = df_seg[df_seg['Patient ID'].isin(train_patients)].reset_index(drop=True)
val_df   = df_seg[df_seg['Patient ID'].isin(val_patients)].reset_index(drop=True)
test_df  = df_seg[df_seg['Patient ID'].isin(test_patients)].reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ============================================================
# Dataset - Segmentation فقط
# ============================================================
class ChestXraySegDataset(Dataset):
    """
    Dataset مخصص للتجزئة فقط.
    - يُرجع: الصورة + الـ mask المُشتق من BBox
    - يدعم تطبيق نفس التحويلات المكانية على الصورة والـ mask معاً
    """

    def __init__(self, df, img_dirs, bbox_df, spatial_transform=None, normalize=None):
        self.df                = df.copy()
        self.img_dirs          = img_dirs
        self.bbox_df           = bbox_df
        self.spatial_transform = spatial_transform  # يُطبَّق على الصورة والـ mask معاً
        self.normalize         = normalize           # يُطبَّق على الصورة فقط

        self.df['img_path'] = self.df['Image Index'].apply(self._find_image)
        self.df = self.df[self.df['img_path'].notnull()].reset_index(drop=True)

    def _find_image(self, img_name):
        for d in self.img_dirs:
            path = os.path.join(d, img_name)
            if os.path.exists(path):
                return path
        return None

    def __len__(self):
        return len(self.df)

    def _build_mask(self, image_index):
        """
        يبني binary mask من إحداثيات BBox.
        كل pixel داخل أي BBox = 1، وما عدا ذلك = 0.
        """
        mask = torch.zeros((224, 224), dtype=torch.float32)
        bboxes = self.bbox_df[self.bbox_df['Image Index'] == image_index]
        for _, bb in bboxes.iterrows():
            x  = float(bb['Bbox [x'])
            y  = float(bb['y'])
            w  = float(bb['w'])
            h  = float(bb['h]'])
            x1, y1 = max(0, int(x)),     max(0, int(y))
            x2, y2 = min(223, int(x+w)), min(223, int(y+h))
            mask[y1:y2, x1:x2] = 1.0
        return mask  # shape: (224, 224)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        image    = Image.open(row['img_path']).convert("L")
        mask     = self._build_mask(row['Image Index'])

        # تحويل إلى Tensor قبل التحويلات المكانية
        image_t = transforms.ToTensor()(image)  # (1, 224, 224)
        mask_t  = mask.unsqueeze(0)              # (1, 224, 224)

        # تطبيق نفس التحويلات المكانية على الصورة والـ mask معاً
        # الطريقة الصحيحة: نضمهما في tensor واحد ونطبق التحويل
        if self.spatial_transform is not None:
            # دمج الصورة والـ mask في tensor واحد (2, 224, 224)
            combined = torch.cat([image_t, mask_t], dim=0)
            combined = self.spatial_transform(combined)
            image_t  = combined[0:1]  # الصورة (1, 224, 224)
            mask_t   = combined[1:2]  # الـ mask (1, 224, 224)
            # نحوّل الـ mask إلى binary مرة أخرى (قد تسبب interpolation قيماً وسطية)
            mask_t = (mask_t > 0.5).float()

        # تطبيع الصورة فقط
        if self.normalize is not None:
            image_t = self.normalize(image_t)

        # اسم المرض (للعرض لاحقاً)
        disease_label = str(row['Finding Labels']).split('|')[0]

        return image_t, mask_t, disease_label

# ============================================================
# Transforms
# ============================================================
# التحويلات المكانية (تُطبَّق على الصورة والـ mask معاً)
spatial_aug = T2.Compose([
    T2.RandomHorizontalFlip(p=0.5),
    T2.RandomVerticalFlip(p=0.2),
    T2.RandomRotation(degrees=20),
    T2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T2.ElasticTransform(alpha=50.0),  # مهم جداً للصور الطبية
])

# تطبيع الصورة فقط
normalize = transforms.Normalize(mean=[0.5], std=[0.5])

# ============================================================
# Datasets و DataLoaders
# ============================================================
train_dataset = ChestXraySegDataset(train_df, img_dirs, bbox_df,
                                    spatial_transform=spatial_aug,
                                    normalize=normalize)
val_dataset   = ChestXraySegDataset(val_df,   img_dirs, bbox_df,
                                    spatial_transform=None,  # بدون augmentation للتحقق
                                    normalize=normalize)
test_dataset  = ChestXraySegDataset(test_df,  img_dirs, bbox_df,
                                    spatial_transform=None,
                                    normalize=normalize)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))

# ============================================================
# النموذج - U-Net للتجزئة
# ============================================================
# نستخدم U-Net الكامل من segmentation_models_pytorch
# مع ResNet34 كـ encoder (أقوى من ResNet18 وما زال سريعاً)
model = Unet(
    encoder_name="resnet18",         # encoder أقوى
    encoder_weights="imagenet",       # pretrained weights
    in_channels=1, 
    decoder_dropout=0.7,# صورة رمادية
    classes=1,                        # mask ثنائي (lesion / not lesion)
    activation=None,                  # نطبق sigmoid يدوياً في الـ loss
    decoder_use_batchnorm=True,
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# دوال الخسارة - مزيج من BCE و Dice
# ============================================================
def dice_loss(pred_logits, target, smooth=1.0):
    """
    Dice Loss: تقيس مدى التداخل بين التنبؤ والـ mask الحقيقي.
    تعمل بشكل جيد مع عدم توازن الفئات (حين الـ mask صغير جداً).
    """
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum(dim=(1,2,3))  # لكل عينة
    union        = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice         = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def bce_dice_loss(pred, target, bce_weight=0.5):
    """
    مزيج من BCE (تعمل على pixel-level) وDice (تعمل على region-level).
    BCE: تعاقب كل pixel على حدة.
    Dice: تقيس جودة التداخل الكلي.
    """
    bce  = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + (1 - bce_weight) * dice

# ============================================================
# Metrics للتجزئة
# ============================================================
def compute_seg_metrics(pred_logits, target, threshold=0.5):
    """
    يحسب Dice Score و IoU (Intersection over Union) للتجزئة.
    هذه هي المقاييس الصحيحة للتجزئة (وليس F1 العام).
    """
    pred  = (torch.sigmoid(pred_logits) > threshold).float()
    target = target.float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union_dice   = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    union_iou    = union_dice - intersection

    dice_score = ((2 * intersection + 1) / (union_dice + 1)).mean().item()
    iou_score  = ((intersection + 1) / (union_iou + 1)).mean().item()

    return dice_score, iou_score

# ============================================================
# Optimizer و Scheduler
# ============================================================
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5
)

# ============================================================
# استئناف من checkpoint
# ============================================================
ckpts = glob.glob(os.path.join(PROJECT_PATH, "seg_checkpoints", "checkpoint_epoch_*.pth"))
if ckpts:
    latest_ckpt = max(ckpts, key=os.path.getctime)
    ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch']
    print(f"✅ Resumed from epoch {start_epoch}")
else:
    start_epoch = 0

best_dice = 0.0
if os.path.exists(checkpoint_path):
    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    best_dice = best_ckpt.get('val_dice', 0.0)
    print(f"⭐ Best Dice so far: {best_dice:.4f}")

# تحميل مقاييس سابقة
if os.path.exists(metrics_file):
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
else:
    metrics = {k: [] for k in ['train_loss', 'train_dice', 'train_iou',
                                 'val_loss',   'val_dice',   'val_iou']}

# ============================================================
# حلقة التدريب
# ============================================================
use_amp = device.type == 'cuda'
scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

early_stop_patience = 10
no_improve_counter  = 0
MIN_DELTA           = 0.001  # Dice تحسن بمقدار 0.1%

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_start = time.time()

    # ---------- مرحلة التدريب ----------
    model.train()
    running_loss = 0.0
    train_dice_sum, train_iou_sum = 0.0, 0.0

    for step, (images, masks, _) in enumerate(train_loader, 1):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            seg_out = model(images)
            loss    = bce_dice_loss(seg_out, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # حساب metrics بدون gradient
        with torch.no_grad():
            d, iou = compute_seg_metrics(seg_out, masks)
            train_dice_sum += d
            train_iou_sum  += iou

        if step % 50 == 0:
            print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{step}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} | Dice: {d:.4f}")

    train_loss = running_loss     / len(train_loader)
    train_dice = train_dice_sum   / len(train_loader)
    train_iou  = train_iou_sum    / len(train_loader)

    # ---------- مرحلة التحقق ----------
    model.eval()
    val_loss_sum  = 0.0
    val_dice_sum  = 0.0
    val_iou_sum   = 0.0

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            seg_out = model(images)
            val_loss_sum += bce_dice_loss(seg_out, masks).item()
            d, iou = compute_seg_metrics(seg_out, masks)
            val_dice_sum += d
            val_iou_sum  += iou

    val_loss = val_loss_sum / len(val_loader)
    val_dice = val_dice_sum / len(val_loader)
    val_iou  = val_iou_sum  / len(val_loader)

    scheduler.step(val_dice)

    # حفظ المقاييس
    for key, val in zip(
        ['train_loss','train_dice','train_iou','val_loss','val_dice','val_iou'],
        [train_loss, train_dice, train_iou, val_loss, val_dice, val_iou]
    ):
        metrics[key].append(val)

    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)

    elapsed = time.time() - epoch_start
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] ({elapsed:.0f}s)")
    print(f"  Train | Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}")
    print(f"  Val   | Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

    # حفظ checkpoint
    ckpt_save_path = os.path.join(PROJECT_PATH, "seg_checkpoints", f"checkpoint_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_dice,
        'val_loss': val_loss,
    }, ckpt_save_path)

    # حفظ أفضل نموذج
    if val_dice > best_dice + MIN_DELTA:
        best_dice = val_dice
        no_improve_counter = 0
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': best_dice,
            'val_iou':  val_iou,
            'val_loss': val_loss,
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
model.eval()

test_dice_sum, test_iou_sum, test_loss_sum = 0.0, 0.0, 0.0
with torch.no_grad():
    for images, masks, _ in test_loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        seg_out = model(images)
        test_loss_sum += bce_dice_loss(seg_out, masks).item()
        d, iou = compute_seg_metrics(seg_out, masks)
        test_dice_sum += d
        test_iou_sum  += iou

print("\n===== Test Evaluation =====")
print(f"  Dice Score: {test_dice_sum / len(test_loader):.4f}")
print(f"  IoU Score:  {test_iou_sum  / len(test_loader):.4f}")
print(f"  Loss:       {test_loss_sum / len(test_loader):.4f}")

# ============================================================
# رسم منحنيات التدريب
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Segmentation Training Metrics", fontsize=14)
epochs_range = range(1, len(metrics['train_loss']) + 1)

axes[0].plot(epochs_range, metrics['train_loss'], 'bo-', label='Train')
axes[0].plot(epochs_range, metrics['val_loss'],   'ro-', label='Val')
axes[0].set_title("BCE + Dice Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

axes[1].plot(epochs_range, metrics['train_dice'], 'go-', label='Train')
axes[1].plot(epochs_range, metrics['val_dice'],   'mo-', label='Val')
axes[1].set_title("Dice Score"); axes[1].legend(); axes[1].set_xlabel("Epoch")

axes[2].plot(epochs_range, metrics['train_iou'], 'co-', label='Train')
axes[2].plot(epochs_range, metrics['val_iou'],   'yo-', label='Val')
axes[2].set_title("IoU Score"); axes[2].legend(); axes[2].set_xlabel("Epoch")

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_PATH, "seg_metrics", "segmentation_metrics.png"), dpi=150)
plt.show()

# ============================================================
# تصور عينات من الاختبار مع masks
# ============================================================
def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def mask_to_bbox(mask_np):
    mask_bin = (mask_np > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x, y, x + w, y + h

model.eval()
fig, axes = plt.subplots(3, 4, figsize=(16, 12))  # 3 عينات × 4 أعمدة
fig.suptitle("Segmentation Predictions vs Ground Truth", fontsize=14)

for sample_idx in range(3):
    image, mask_gt, label = test_dataset[sample_idx]
    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        seg_out  = model(input_tensor)
        seg_pred = torch.sigmoid(seg_out[0, 0]).cpu().numpy()

    # إلغاء تطبيع الصورة للعرض
    img_display = denormalize(image).numpy().squeeze()
    img_rgb     = cv2.cvtColor((img_display * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mask_gt_np  = mask_gt.numpy().squeeze()

    # الصورة الأصلية
    axes[sample_idx, 0].imshow(img_display, cmap='gray')
    axes[sample_idx, 0].set_title(f"Original\n{label}")
    axes[sample_idx, 0].axis('off')

    # Ground Truth mask
    axes[sample_idx, 1].imshow(img_display, cmap='gray')
    axes[sample_idx, 1].imshow(mask_gt_np, cmap='Reds', alpha=0.5)
    axes[sample_idx, 1].set_title("Ground Truth Mask")
    axes[sample_idx, 1].axis('off')

    # Predicted mask كـ heatmap
    axes[sample_idx, 2].imshow(img_display, cmap='gray')
    axes[sample_idx, 2].imshow(seg_pred, cmap='hot', alpha=0.5)
    axes[sample_idx, 2].set_title("Predicted Mask\n(Heatmap)")
    axes[sample_idx, 2].axis('off')

    # Bounding box من الـ mask المتنبأ به
    img_bbox = img_rgb.copy()
    bbox     = mask_to_bbox(seg_pred)
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
    axes[sample_idx, 3].imshow(cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB))
    axes[sample_idx, 3].set_title("Predicted BBox")
    axes[sample_idx, 3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_PATH, "seg_predictions", "segmentation_samples.png"), dpi=150)
plt.show()
print("✅ Saved segmentation prediction samples.")
