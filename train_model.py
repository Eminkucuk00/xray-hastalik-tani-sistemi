# ==========================================
# Proje: X-ray Hastalik Tani Sistemi
# Model Egitim Scripti
# Ogrenci: Mehmet Emin Kucuk
# Ogrenci No: 233908073
# ==========================================

import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Cihaz seçimi (GPU varsa onu kullan)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanilan cihaz: {device}")

# ==========================================
# 1. VERİ AUGMENTASYONU (Küçük veri seti için çok önemli)
# ==========================================

# ImageNet normalizasyon değerleri (pretrained model için gerekli)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Eğitim verisi için güçlü augmentasyon
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),       # Rastgele kırpma
    transforms.RandomHorizontalFlip(p=0.5),                     # Yatay çevirme
    transforms.RandomRotation(15),                               # ±15 derece döndürme
    transforms.ColorJitter(brightness=0.2, contrast=0.2),        # Parlaklık/kontrast
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),    # Kaydırma
    transforms.RandomGrayscale(p=0.1),                           # Gri tonlama
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# Doğrulama verisi için sadece resize ve normalize
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# ==========================================
# 2. VERİ YÜKLEMESİ
# ==========================================

train_dir = 'dataset/train_split'
val_dir = 'dataset/val_split'

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)

print(f"Siniflar: {train_data.classes}")
print(f"Egitim verisi: {len(train_data)} goruntu")
print(f"Dogrulama verisi: {len(val_data)} goruntu")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)

# ==========================================
# 3. MODEL OLUŞTURMA (Transfer Learning)
# ==========================================

# Pretrained ResNet18 yükle
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# İlk katmanları dondur (sadece son katmanları eğit - küçük veri seti için önemli)
for param in model.parameters():
    param.requires_grad = False

# Son tam bağlantılı katmanı değiştir (2 sınıf için)
model.fc = nn.Sequential(
    nn.Dropout(0.3),                                   # Overfitting önleme
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 2)
)

model = model.to(device)

# ==========================================
# 4. EĞİTİM AYARLARI
# ==========================================

criterion = nn.CrossEntropyLoss()

# Sadece eğitilebilir parametreleri optimize et
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001,
    weight_decay=1e-4  # L2 regularization
)

# Learning rate scheduler - her 10 epoch'ta lr'yi 0.1 ile çarp
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ==========================================
# 5. EĞİTİM DÖNGÜSÜ
# ==========================================

epochs = 30
best_val_acc = 0.0

print("\n" + "=" * 50)
print("EGITIM BASLIYOR")
print("=" * 50)

for epoch in range(epochs):
    # --- Eğitim ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_acc = 100 * correct_train / total_train

    # --- Doğrulama ---
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_acc = 100 * correct_val / total_val

    # Learning rate güncelle
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch+1:2d}/{epochs}] | "
          f"Loss: {running_loss:.4f} | "
          f"Egitim Acc: %{train_acc:.1f} | "
          f"Dogrulama Acc: %{val_acc:.1f} | "
          f"LR: {current_lr:.6f}")

    # En iyi modeli kaydet
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model.pth")
        print(f"  >> En iyi model kaydedildi! (Dogrulama: %{best_val_acc:.1f})")

# ==========================================
# 6. SONUÇ
# ==========================================

print("\n" + "=" * 50)
print(f"EGITIM TAMAMLANDI!")
print(f"En iyi dogrulama dogrulugu: %{best_val_acc:.1f}")
print("Model 'model.pth' olarak kaydedildi.")
print("=" * 50)