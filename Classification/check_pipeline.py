import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# --- CONFIGURATION ---
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vignettes_avions')
IMG_SIZE = (566, 566)

def processing_pipeline(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    # C'est la fonction clé qu'on veut vérifier :
    img = tf.image.resize_with_pad(img, IMG_SIZE[0], IMG_SIZE[1])
    return img / 255.0

# Récupérer 4 images au hasard
files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
sample_files = np.random.choice(files, 4)

print("🖼️ Génération de l'aperçu...")
plt.figure(figsize=(12, 12))

for i, file_path in enumerate(sample_files):
    img_processed = processing_pipeline(file_path)
    
    plt.subplot(2, 2, i + 1)
    plt.imshow(img_processed)
    plt.title(f"Format: {IMG_SIZE}\nPadding OK")
    plt.axis('off')

print("✅ Fenêtre d'affichage ouverte. Vérifie les bandes noires !")
plt.tight_layout()
plt.show()