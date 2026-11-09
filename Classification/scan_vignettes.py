import os
from PIL import Image

# Chemin direct vers tes images
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vignettes_avions')

def find_max_dimensions():
    max_w, max_h = 0, 0
    count = 0
    
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Erreur : Je ne trouve pas le dossier ici : {IMAGE_DIR}")
        return

    print("🚀 Scan des vignettes en cours...")
    for f in os.listdir(IMAGE_DIR):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                with Image.open(os.path.join(IMAGE_DIR, f)) as img:
                    w, h = img.size
                    if w > max_w: max_w = w
                    if h > max_h: max_h = h
                count += 1
            except:
                continue
                
    print("-" * 30)
    print(f"📊 Résultat sur {count} images :")
    print(f"🔹 Largeur max : {max_w} px")
    print(f"🔹 Hauteur max : {max_h} px")
    print(f"💡 Ton carré de padding devra faire : {max(max_w, max_h)}x{max(max_w, max_h)} px")
    print("-" * 30)

if __name__ == "__main__":
    find_max_dimensions()