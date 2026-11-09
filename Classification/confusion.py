import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model import get_model_instance_segmentation
from dataset import PlaneDataset
import torchvision.transforms as T

# --- FONCTION DE CALCUL IOU ---
def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# --- CONFIGURATION ---
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BASE_PATH = '/content/drive/MyDrive/Projet IA/Classification'
CLASSES = ['Background'] + [f'A{i}' for i in range(1, 21)]

# 1. Charger le modèle et le dataset
model = get_model_instance_segmentation(21)
model.load_state_dict(torch.load(os.path.join(BASE_PATH, 'faster_rcnn_avions.pth'), map_location=DEVICE))
model.to(DEVICE).eval()

dataset = PlaneDataset(BASE_PATH, transforms=T.Compose([T.ToTensor()]))
# On prend les 50 images de test
dataset_test = torch.utils.data.Subset(dataset, range(len(dataset)-50, len(dataset)))

y_true = []
y_pred = []

print("🔄 Comparaison prédictions vs vérité terrain...")

with torch.no_grad():
    for img, target in dataset_test:
        img = img.to(DEVICE).unsqueeze(0)
        prediction = model(img)
        
        gt_boxes = target['boxes'].numpy()
        gt_labels = target['labels'].numpy()
        
        p_boxes = prediction[0]['boxes'].cpu().numpy()
        p_labels = prediction[0]['labels'].cpu().numpy()
        p_scores = prediction[0]['scores'].cpu().numpy()
        
        # Garder uniquement les prédictions confiantes (> 0.5)
        mask = p_scores > 0.5
        p_boxes = p_boxes[mask]
        p_labels = p_labels[mask]

        matched_gt = set()
        for i, pb in enumerate(p_boxes):
            best_iou = 0
            best_gt_idx = -1
            for j, gb in enumerate(gt_boxes):
                if j in matched_gt: continue
                iou = box_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou > 0.5: # Si on a un match géographique
                y_true.append(gt_labels[best_gt_idx])
                y_pred.append(p_labels[i])
                matched_gt.add(best_gt_idx)
            else:
                # L'IA a trouvé un truc qui n'existe pas (Faux Positif)
                y_true.append(0) # Background
                y_pred.append(p_labels[i])
        
        # Ajouter les avions oubliés par l'IA (Faux Négatifs)
        for j in range(len(gt_labels)):
            if j not in matched_gt:
                y_true.append(gt_labels[j])
                y_pred.append(0) # Background

# 2. Tracer la matrice
cm = confusion_matrix(y_true, y_pred, labels=range(21))
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Matrice de Confusion Finale (Détection + Classification)')
plt.ylabel('Vérité (XML)')
plt.xlabel('Prédiction (IA)')
plt.savefig(os.path.join(BASE_PATH, 'matrice_confusion.png'))
plt.show()

print(f"✅ Matrice sauvegardée dans {BASE_PATH}/matrice_confusion.png")