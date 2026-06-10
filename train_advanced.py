import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanilan cihaz: {device}")

# ==========================================
# 1. VERİ AUGMENTASYONU VE NORMALİZASYONU
# ==========================================
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Strong augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# Validation transform (no augmentation, just resize)
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
# 3. MODEL OLUŞTURMA (ResNet-18)
# ==========================================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Modify classifier head for 2 classes
model.fc = nn.Sequential(
    nn.Dropout(0.4),                                   # Overfitting'i önlemek için biraz arttırdık
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2)
)

model = model.to(device)
criterion = nn.CrossEntropyLoss()

# ==========================================
# 4. EĞİTİM YÖNTEMİ: İKİ AŞAMALI İNCE AYAR (FINE-TUNING)
# ==========================================

# AŞAMA 1: Sadece yeni eklediğimiz sınıflandırıcı kafayı eğit (Omurga dondurulmuş)
print("\n" + "=" * 50)
print("ASAMA 1: SINIFLANDIRICI KAFANIN ISITILMASI (WARM-UP)")
print("=" * 50)

for param in model.parameters():
    param.requires_grad = False

# Sadece fc katmanının parametrelerini eğitime aç
for param in model.fc.parameters():
    param.requires_grad = True

optimizer_phase1 = torch.optim.Adam(model.fc.parameters(), lr=0.001)
epochs_phase1 = 5
best_val_acc = 0.0

for epoch in range(epochs_phase1):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_phase1.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_phase1.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_acc = 100 * correct_train / total_train

    # Validation
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_acc = 100 * correct_val / total_val
    print(f"Warmup Epoch [{epoch+1}/{epochs_phase1}] | Loss: {running_loss:.4f} | Egitim Acc: %{train_acc:.1f} | Dogrulama Acc: %{val_acc:.1f}")

    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model.pth")


# AŞAMA 2: Tüm katmanları aç ve çok düşük öğrenme katsayısı ile ince ayar yap
print("\n" + "=" * 50)
print("ASAMA 2: TUM KATMANLARDA INCE AYAR (FINE-TUNING)")
print("=" * 50)

# Tüm parametreleri eğitime aç
for param in model.parameters():
    param.requires_grad = True

# Diferansiyel öğrenme hızı: Omurga için çok düşük lr (1e-5), kafa için biraz daha yüksek (1e-4)
optimizer_phase2 = torch.optim.Adam([
    {'params': [p for name, p in model.named_parameters() if 'fc' not in name], 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-4}
], weight_decay=1e-4)

# Learning rate scheduler - her 10 epoch'ta lr'yi 0.5 ile çarp
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_phase2, step_size=10, gamma=0.5)

epochs_phase2 = 35
total_epochs = epochs_phase1 + epochs_phase2

for epoch in range(epochs_phase2):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_phase2.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_phase2.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_acc = 100 * correct_train / total_train

    # Validation
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_acc = 100 * correct_val / total_val
    scheduler.step()
    
    current_lr = optimizer_phase2.param_groups[0]['lr']
    print(f"Fine-tune Epoch [{epoch+1}/{epochs_phase2}] | Loss: {running_loss:.4f} | Egitim Acc: %{train_acc:.1f} | Dogrulama Acc: %{val_acc:.1f} | LR: {current_lr:.6f}")

    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model.pth")
        print(f"  >> En iyi model güncellendi! (Dogrulama: %{best_val_acc:.1f})")

print("\n" + "=" * 50)
print(f"EGITIM TAMAMLANDI!")
print(f"Ulasilan En Yuksek Dogruluk: %{best_val_acc:.1f}")
print("En iyi model 'model.pth' olarak kaydedildi.")
print("=" * 50)
