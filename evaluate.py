import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import get_transfer_model # Match your trained model type

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder("dataset/val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
class_names = val_dataset.classes

# Load Model
model = get_transfer_model(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("transfer_recycling.pth"))
model.eval()

all_preds = []
all_labels = []
misclassified_images = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                misclassified_images.append({
                    "image": images[i].cpu(),
                    "true": class_names[labels[i]],
                    "pred": class_names[preds[i]]
                })
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 1. Classification Metrics (Macro F1 & Per-class Recall)
print("\n--- Classification Report ---")
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report)

# 2. Confusion Matrix Plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# 3. Error Gallery Visualization (Plotting up to 6 misclassified items)
if len(misclassified_images) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for idx, item in enumerate(misclassified_images[:6]):
        img = item["image"].permute(1, 2, 0).numpy()
        # Unnormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[idx].imshow(img)
        axes[idx].set_title(f"True: {item['true']}\nPred: {item['pred']}", color='red')
        axes[idx].axis('off')
    plt.suptitle("Error Gallery (High-Confidence Failures)")
    plt.tight_layout()
    plt.savefig("error_gallery.png")
    plt.show()
else:
    print("No misclassifications found!")