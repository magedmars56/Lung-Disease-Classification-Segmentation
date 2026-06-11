"""import os
from PIL import Image
import torch
from torchvision import transforms

# ------------------- الإعدادات -------------------
input_dir = "images_001/images"        # مجلد الصور الأصلية
output_dir = "preprocessed_images_001"     # مجلد حفظ الصور بعد preprocessing
os.makedirs(output_dir, exist_ok=True)

# ------------------- التحويلات -------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),      # تغيير الحجم
    transforms.ToTensor(),              # تحويل إلى Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # التطبيع
])

# ------------------- قائمة الصور -------------------
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# ------------------- Preprocessing -------------------
for img_name in image_files:
    img_path = os.path.join(input_dir, img_name)
    img = Image.open(img_path).convert("L")  # تحويل للصورة الرمادية

    # تطبيق التحويلات
    img_tensor = preprocess(img)

    # لإعادة الحفظ كـ صورة PIL بعد preprocessing (مع الحفاظ على القيم بين 0-1)
    img_to_save = img_tensor.clone()
    # التحويل من Tensor [1,H,W] إلى [H,W] وضرب 255 ثم تحويل إلى uint8
    img_to_save = img_to_save.squeeze(0) * 0.5 + 0.5  # عكس normalize
    img_to_save = (img_to_save * 255).byte()
    img_pil = Image.fromarray(img_to_save.numpy())

    # حفظ الصورة بنفس الاسم أو مع إضافة _preprocessed
    #save_name = os.path.splitext(img_name)[0] + ".png"
    save_name = os.path.splitext(img_name)[0] + ".png"
    save_path = os.path.join(output_dir, save_name)
    img_pil.save(save_path)

    print(f"✅ Saved preprocessed image: {save_name}")
"""
import os
from PIL import Image
import torch
from torchvision import transforms
import time  # لإضافة التوقيت

# ------------------- الإعدادات -------------------
input_dir = "images_007/images"             # مجلد الصور الأصلية
output_dir = "preprocessed_images_007"     # مجلد حفظ الصور بعد preprocessing
os.makedirs(output_dir, exist_ok=True)

# ------------------- التحويلات -------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),         # تغيير الحجم
    transforms.ToTensor(),                  # تحويل إلى Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # التطبيع
])

# ------------------- قائمة الصور -------------------
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# ------------------- قياس الوقت -------------------
start_time = time.time()  # بدء التوقيت

# ------------------- Preprocessing -------------------
for img_name in image_files:
    img_path = os.path.join(input_dir, img_name)
    img = Image.open(img_path).convert("L")  # تحويل للصورة الرمادية

    # تطبيق التحويلات
    img_tensor = preprocess(img)

    # لإعادة الحفظ كـ صورة PIL بعد preprocessing
    img_to_save = img_tensor.clone()
    img_to_save = img_to_save.squeeze(0) * 0.5 + 0.5  # عكس normalize
    img_to_save = (img_to_save * 255).byte()
    img_pil = Image.fromarray(img_to_save.numpy())

    # حفظ الصورة بنفس الاسم والامتداد الأصلي
    save_name = img_name
    save_path = os.path.join(output_dir, save_name)
    img_pil.save(save_path)

    print(f"✅ Saved preprocessed image: {save_name}")

end_time = time.time()  # نهاية التوقيت
elapsed_time = end_time - start_time
print(f"\n⏱ Total preprocessing time: {elapsed_time:.2f} seconds for {len(image_files)} images")
