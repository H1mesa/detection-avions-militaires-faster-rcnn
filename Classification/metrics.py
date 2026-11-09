import torch
import numpy as np
from dataset import PlaneDataset
from model import get_model_instance_segmentation
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os

# --- FONCTION UTILE : Calcul de l'IoU ---
def box_iou(box1, box2):
    # box = [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# --- CONFIG ---
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BASE_PATH = '/content/drive/MyDrive/Projet IA/Classification'
MODEL_PATH = os.path.join(BASE_PATH, 'faster_rcnn_avions.pth')

model = get_model_instance_segmentation(21)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

dataset = PlaneDataset(BASE_PATH, transforms=T.Compose([T.ToTensor()]))
# On prend les 50 dernières images pour le test
indices = list(range(len(dataset)))
dataset_test = torch.utils.data.Subset(dataset, indices[-50:])
data_loader = DataLoader(dataset_test, batch_size=1, collate_fn=lambda x: tuple(zip(*x)))

TP, FP, FN = 0, 0, 0
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.6

print("📊 Calcul des métriques détaillées...")

with torch.no_grad():
    for images, targets in data_loader:
        images = [img.to(DEVICE) for img in images]
        outputs = model(images)
        
        for i in range(len(outputs)):
            pred_boxes = outputs[i]['boxes'].cpu().numpy()
            pred_labels = outputs[i]['labels'].cpu().numpy()
            pred_scores = outputs[i]['scores'].cpu().numpy()
            
            gt_boxes = targets[i]['boxes'].cpu().numpy()
            gt_labels = targets[i]['labels'].cpu().numpy()
            
            # On ne garde que les prédictions confiantes
            mask = pred_scores > CONF_THRESHOLD
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]
            
            matched_gt = set()
            for pb, pl in zip(pred_boxes, pred_labels):
                match_found = False
                for idx, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                    if idx in matched_gt: continue
                    
                    iou = box_iou(pb, gb)
                    if iou >= IOU_THRESHOLD and pl == gl:
                        TP += 1
                        matched_gt.add(idx)
                        match_found = True
                        break
                
                if not match_found:
                    FP += 1 # L'IA a vu un truc qui n'existe pas ou s'est trompée de classe
            
            FN += (len(gt_boxes) - len(matched_gt)) # Les avions oubliés

# --- CALCULS FINAUX ---
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n--- RÉSULTATS POUR TON RAPPORT ---")
print(f"✅ True Positives : {TP}")
print(f"❌ False Positives : {FP}")
print(f"📉 False Negatives : {FN}")
print(f"----------------------------------")
print(f"🎯 Précision : {precision:.2%}")
print(f"📡 Rappel (Recall) : {recall:.2%}")
print(f"🏆 F1-Score : {f1_score:.4f}")