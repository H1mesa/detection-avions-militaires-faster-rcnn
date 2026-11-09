import os
import shutil
import random

# --- CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, "vignettes_avions")
dataset_dir = os.path.join(current_dir, "dataset")

# Création de l'arborescence (Uniquement Train et Val)
for f in ['train', 'val']: 
    os.makedirs(os.path.join(dataset_dir, f), exist_ok=True)

# Groupement des images par classe (A1, A2...)
classes = {}
for img in os.listdir(source_dir):
    if img.endswith('.jpg'):
        lbl = img.split('_')[0]
        if lbl not in classes: classes[lbl] = []
        classes[lbl].append(img)

# Répartition 80% Train / 20% Val
print("📦 Répartition des 7 870 images en Train (80%) et Val (20%)...")

for lbl, imgs in classes.items():
    random.shuffle(imgs) # Mélange pour l'impartialité
    n = len(imgs)
    
    # Calcul de la limite (80% pour l'entraînement)
    train_idx = int(0.8 * n)
    
    train_list = imgs[:train_idx]
    val_list = imgs[train_idx:]
    
    # Copie vers les dossiers respectifs
    for img in train_list:
        shutil.copy(os.path.join(source_dir, img), os.path.join(dataset_dir, 'train', img))
    for img in val_list:
        shutil.copy(os.path.join(source_dir, img), os.path.join(dataset_dir, 'val', img))

print(f"✅ Terminé ! Ton dossier 'dataset' est prêt avec {len(os.listdir(os.path.join(dataset_dir, 'train')))} images en Train et {len(os.listdir(os.path.join(dataset_dir, 'val')))} en Val.")