import torch
import os
from model import get_model_instance_segmentation
from dataset import PlaneDataset
import torchvision.transforms as T
from tqdm import tqdm  # Indispensable pour voir l'avancement !

# --- CONFIGURATION ---
BASE_PATH = '/content/drive/MyDrive/Projet IA/Classification'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 1. Charger le modèle avec tes poids actuels (0.69)
model = get_model_instance_segmentation(21)
checkpoint_path = os.path.join(BASE_PATH, 'faster_rcnn_avions.pth')

if os.path.exists(checkpoint_path):
    print("💎 Chargement du modèle existant (V1) pour le Fine-Tuning...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
else:
    print("⚠️ Attention : faster_rcnn_avions.pth non trouvé, départ à zéro !")

model.to(DEVICE)

# 2. Optimiseur : Learning Rate divisé par 10 pour la précision (0.0005)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)

# 3. DataLoader
# num_workers=0 est plus stable pour lire des fichiers sur Google Drive
dataset = PlaneDataset(BASE_PATH, transforms=T.Compose([T.ToTensor()]))
data_loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=0, 
    collate_fn=lambda x: tuple(zip(*x))
)

# 4. Entraînement de rattrapage
num_epochs = 7 
print(f"🚀 Début du Fine-Tuning ciblé sur {DEVICE}...")

for epoch in range(num_epochs):
    model.train()
    # On crée la barre de progression pour l'époque en cours
    prog_bar = tqdm(data_loader, desc=f"Époque {epoch+1}/{num_epochs}")
    
    epoch_loss = 0
    for images, targets in prog_bar:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_loss += losses.item()
        # On met à jour la barre avec la loss actuelle
        prog_bar.set_postfix(loss=losses.item())
    
    avg_loss = epoch_loss / len(data_loader)
    print(f"✅ Époque {epoch+1} terminée. Perte moyenne: {avg_loss:.4f}")

# Sauvegarde du nouveau modèle spécialisé
save_path = os.path.join(BASE_PATH, 'faster_rcnn_avions_V2.pth')
torch.save(model.state_dict(), save_path)
print(f"✨ Bravo ! Le modèle V2 est sauvegardé ici : {save_path}")