import os
import hashlib

def calculer_hash(image_path):
    """Calcule l'empreinte unique d'un fichier."""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

vignettes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vignettes_avions")
hashes = {}
doublons = []

print("Vérification des doublons en cours...")

for filename in os.listdir(vignettes_dir):
    if filename.endswith(".jpg"):
        path = os.path.join(vignettes_dir, filename)
        file_hash = calculer_hash(path)
        
        if file_hash in hashes:
            doublons.append((filename, hashes[file_hash]))
        else:
            hashes[file_hash] = filename

if doublons:
    print(f"⚠️ {len(doublons)} doublons trouvés !")
    # Optionnel : supprimer les doublons
    # for d, _ in doublons: os.remove(os.path.join(vignettes_dir, d))
else:
    print("✅ Aucun doublon détecté. La qualité des données est optimale.")