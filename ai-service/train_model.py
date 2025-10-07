import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# ---------------- Parameters ----------------
DATA_DIR = "dataset"          # folder with subfolders for each class
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3
MODEL_PATH = "artifacts/pest_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Transforms ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ---------------- Dataset ----------------
train_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------- Model ----------------
classes = train_dataset.classes
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model = model.to(DEVICE)

# ---------------- Loss & Optimizer ----------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- Training Loop ----------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Accuracy: {acc:.4f}")

# ---------------- Save Model ----------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes
}, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
