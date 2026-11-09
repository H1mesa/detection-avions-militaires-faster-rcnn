import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import os
from model import get_model_instance_segmentation
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"🚀 Traitement lancé sur : {DEVICE}")

BASE_PATH = '/content/drive/MyDrive/Projet IA/Classification'
MODEL_PATH = os.path.join(BASE_PATH, 'faster_rcnn_avions.pth')
IMG_DIR = os.path.join(BASE_PATH, 'images')
OUTPUT_DIR = os.path.join(BASE_PATH, 'RESULTATS_1331_IMAGES')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CLASSES = ['Background', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 
           'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20']

# ==========================================
# 2. CHARGEMENT DU MODÈLE
# ==========================================
model = get_model_instance_segmentation(21)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==========================================
# 3. BOUCLE DE TRAITEMENT (BATCH)
# ==========================================
all_images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Police pour les étiquettes
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 18)
except:
    font = ImageFont.load_default()

print(f"🔍 Analyse de {len(all_images)} images en cours...")

with torch.no_grad():
    for img_name in tqdm(all_images):
        img_path = os.path.join(IMG_DIR, img_name)
        img = Image.open(img_path).convert("RGB")
        
        # Préparation image
        img_tensor = T.Compose([T.ToTensor()])(img).unsqueeze(0).to(DEVICE)
        
        # Prédiction
        prediction = model(img_tensor)

        # Dessin des boîtes
        draw = ImageDraw.Draw(img)
        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()

        for i in range(len(boxes)):
            if scores[i] > 0.6: # On garde les détections solides
                box = boxes[i]
                label_txt = f"{CLASSES[labels[i]]} {scores[i]:.2f}"
                
                # Rectangle et étiquette
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="lime", width=3)
                draw.text((box[0], box[1] - 20), label_txt, fill="lime", font=font)

        # Sauvegarde
        img.save(os.path.join(OUTPUT_DIR, f"det_{img_name}"))

print(f"\n✅ TERMINÉ ! Va voir le dossier : {OUTPUT_DIR}")