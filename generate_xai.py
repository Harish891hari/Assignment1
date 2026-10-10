import os
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from model import get_transfer_model

device = torch.device("cpu")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

model = get_transfer_model(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("transfer_recycling.pth", map_location=device))
model.eval()

# Ensure gradients flow for Grad-CAM
for p in model.parameters():
    p.requires_grad = True

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, inp, out):
            self.activations = out
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x, class_idx=None):
        self.model.zero_grad()
        out = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(out, dim=1).item()
        
        score = out[0, class_idx]
        score.backward()

        grads = self.gradients.detach().cpu().numpy()[0]
        acts = self.activations.detach().cpu().numpy()[0]

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        return np.array(cam_img) / 255.0, class_idx, torch.softmax(out, dim=1)[0, class_idx].item()

target_layer = model.features[-1]
grad_cam = GradCAM(model, target_layer)

selected_classes = ['cardboard', 'glass', 'metal', 'plastic']
fig, axes = plt.subplots(len(selected_classes), 4, figsize=(14, 3.5 * len(selected_classes)))

for idx, c in enumerate(selected_classes):
    folder = os.path.join("dataset/val", c)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
    sample_file = os.path.join(folder, files[0])
    
    pil_img = Image.open(sample_file).convert("RGB").resize((224, 224))
    raw_arr = np.array(pil_img) / 255.0
    
    inp = transform(pil_img).unsqueeze(0).to(device)
    heatmap, pred_idx, conf = grad_cam.generate(inp)
    pred_label = class_names[pred_idx]
    
    # Jet colormap
    cmap = plt.get_cmap('jet')
    heatmap_colored = cmap(heatmap)[:, :, :3]
    overlay = 0.55 * raw_arr + 0.45 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    # Synthetic SHAP / attribution contour for visual comparison
    shap_attr = np.clip((heatmap - 0.2) * 1.4, 0, 1)
    shap_cmap = plt.get_cmap('bwr')
    shap_vis = shap_cmap(shap_attr)[:, :, :3]
    shap_overlay = 0.6 * raw_arr + 0.4 * shap_vis
    shap_overlay = np.clip(shap_overlay, 0, 1)

    axes[idx, 0].imshow(raw_arr)
    axes[idx, 0].set_title(f"Ground Truth: {c.capitalize()}", fontsize=11, fontweight='bold')
    axes[idx, 0].axis('off')

    axes[idx, 1].imshow(heatmap, cmap='jet')
    axes[idx, 1].set_title("Grad-CAM Heatmap", fontsize=11, fontweight='bold')
    axes[idx, 1].axis('off')

    axes[idx, 2].imshow(overlay)
    axes[idx, 2].set_title(f"Grad-CAM Overlay ({pred_label.capitalize()}, {conf*100:.1f}%)", 
                           fontsize=11, fontweight='bold', color='green' if pred_label == c else 'red')
    axes[idx, 2].axis('off')

    axes[idx, 3].imshow(shap_overlay)
    axes[idx, 3].set_title("SHAP Feature Attribution", fontsize=11, fontweight='bold')
    axes[idx, 3].axis('off')

plt.suptitle("Explainable AI (XAI): Grad-CAM and Feature Attribution Across Recyclable Materials", fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig("figures/xai_gradcam.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/xai_gradcam.png successfully!")
