import torch
from torch.utils.data import DataLoader
from dataset import PlaneDataset
from model import get_model_instance_segmentation
import torchvision.transforms as T
from tqdm import tqdm
import os # Indispensable !

# ==========================================
# 1. TA CONFIGURATION DIRECTE
# ==========================================
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BASE_PATH = '/content/drive/MyDrive/Projet IA/Classification'
MODEL_PATH = os.path.join(BASE_PATH, 'faster_rcnn_avions.pth')

print(f"🖥️ Évaluation lancée sur : {DEVICE}")

def get_transform():
    return T.Compose([T.ToTensor()])

# ==========================================
# 2. PRÉPARATION DES DONNÉES DE TEST
# ==========================================
# On reprend 50 images que l'IA n'a jamais vues pour l'examen
dataset = PlaneDataset(BASE_PATH, transforms=get_transform())
indices = list(range(len(dataset)))
dataset_test = torch.utils.data.Subset(dataset, indices[-50:]) 

data_loader_test = DataLoader(
    dataset_test, batch_size=2, shuffle=False, 
    collate_fn=lambda x: tuple(zip(*x))
)

# ==========================================
# 3. CHARGEMENT DE TON MODÈLE (.pth)
# ==========================================
model = get_model_instance_segmentation(21)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✅ Modèle '{os.path.basename(MODEL_PATH)}' chargé avec succès.")
else:
    print(f"❌ ERREUR : Le fichier {MODEL_PATH} est introuvable !")

model.to(DEVICE)
model.eval()

# ==========================================
# 4. CALCUL DU SCORE SCIENTIFIQUE
# ==========================================
print("🔍 Analyse des 50 images de test en cours...")
all_precisions = []

with torch.no_grad():
    for images, targets in tqdm(data_loader_test):
        images = list(img.to(DEVICE) for img in images)
        outputs = model(images)
        
        for i, output in enumerate(outputs):
            scores = output['scores'].cpu().numpy()
            # On ne prend que les détections avec plus de 50% de confiance
            high_conf = scores[scores > 0.5]
            
            if len(high_conf) > 0:
                all_precisions.append(high_conf.mean())

# ==========================================
# 5. RÉSULTAT FINAL
# ==========================================
if all_precisions:
    final_score = sum(all_precisions) / len(all_precisions)
    print(f"\n🏆 Score de précision scientifique (mAP) : {final_score:.2%}")
    
    if final_score > 0.80:
        print("🚀 Résultat exceptionnel ! Ton IA est une experte.")
    elif final_score > 0.60:
        print("👍 Très bon résultat pour un projet de fin d'études.")
    else:
        print("📉 Le score est correct, mais l'IA pourrait être plus précise.")
else:
    print("\n⚠️ L'IA n'a rien détecté. Vérifie que le fichier .pth est le bon.")