# ======================================================
# CLOUD COVER PREDICTION USING GOOGLENET (NO LOGIN)
# DATASET: EuroSAT (AUTO DOWNLOAD)
# ======================================================

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision.models import GoogLeNet_Weights
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report

# ===============================
# 1. CONFIG
# ===============================
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 2. TRANSFORMS
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# 3. AUTO DOWNLOAD DATASET
# ===============================
dataset = datasets.EuroSAT(
    root="data",
    download=True,
    transform=transform
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class_names = dataset.classes
print("Classes:", class_names)

# ===============================
# 4. LOAD GOOGLE NET
# ===============================
model = models.googlenet(weights=GoogLeNet_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===============================
# 5. TRAIN MODEL
# ===============================
train_acc, test_acc = [], []
train_loss, test_loss = [], []

for epoch in range(EPOCHS):
    model.train()
    correct, total, running_loss = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss.append(running_loss / len(train_loader))
    train_acc.append(100 * correct / total)

    # ===============================
    # TEST
    # ===============================
    model.eval()
    correct, total, val_loss = 0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_loss.append(val_loss / len(test_loader))
    test_acc.append(100 * correct / total)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Acc: {train_acc[-1]:.2f}% "
          f"Test Acc: {test_acc[-1]:.2f}%")

# ===============================
# 6. CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ===============================
# 7. ACCURACY & LOSS CURVES
# ===============================
plt.figure()
plt.plot(train_acc, label="Train Accuracy")
plt.plot(test_acc, label="Test Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.show()

plt.figure()
plt.plot(train_loss, label="Train Loss")
plt.plot(test_loss, label="Test Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()

# ===============================
# 8. SAVE MODEL
# ===============================
torch.save(model.state_dict(), "googlenet_cloud_model.pth")
print("Model saved successfully.")
