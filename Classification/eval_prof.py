import torch
import torchvision.transforms as T
from PIL import Image
import os
import json
from model import get_model_instance_segmentation
from tqdm import tqdm

# --- CONFIGURATION COLAB ---
# 1. Utilise ce chemin pour Colab
BASE_PATH = '/content/drive/MyDrive/Projet IA/Classification'

# 2. Le reste du code s'adaptera tout seul
MODEL_PATH = os.path.join(BASE_PATH, 'faster_rcnn_avions.pth')
EVAL_IMG_DIR = os.path.join(BASE_PATH, 'images_evaluation/eval-dataset/images')
OUTPUT_JSON = os.path.join(BASE_PATH, 'predictions_officielles.json')

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CLASSES = ['Background'] + [f'A{i}' for i in range(1, 21)]

def run_evaluation():
    print(f"🚀 Inférence sur : {DEVICE}")
    
    # 1. Chargement du modèle
    model = get_model_instance_segmentation(21)
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Modèle introuvable : {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    # 2. Liste des images
    if not os.path.exists(EVAL_IMG_DIR):
        print(f"❌ Dossier images introuvable : {EVAL_IMG_DIR}")
        return

    all_images = [f for f in os.listdir(EVAL_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    results = {}

    print(f"🔍 Analyse de {len(all_images)} images...")

    # 3. Détection
    with torch.no_grad():
        for img_name in tqdm(all_images):
            img_path = os.path.join(EVAL_IMG_DIR, img_name)
            img = Image.open(img_path).convert("RGB")
            img_t = T.Compose([T.ToTensor()])(img).unsqueeze(0).to(DEVICE)
            
            prediction = model(img_t)
            
            img_preds = []
            boxes = prediction[0]['boxes'].cpu().numpy()
            labels = prediction[0]['labels'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()

            for i in range(len(boxes)):
                if scores[i] > 0.5: # On garde les détections solides
                    box = boxes[i]
                    img_preds.append({
                        "class": CLASSES[labels[i]],
                        "coordinates": {
                            "xmin": int(box[0]), "ymin": int(box[1]),
                            "xmax": int(box[2]), "ymax": int(box[3])
                        }
                    })
            results[img_name] = img_preds

    # 4. Sauvegarde
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ JSON généré : {OUTPUT_JSON}")

if __name__ == "__main__":
    run_evaluation()