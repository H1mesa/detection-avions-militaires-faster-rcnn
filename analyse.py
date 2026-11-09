import os
import matplotlib.pyplot as plt
from collections import Counter

# --- CONFIGURATION ---
# Le dossier où se trouvent tes 7173 vignettes
current_dir = os.path.dirname(os.path.abspath(__file__))
vignettes_dir = os.path.join(current_dir, "vignettes_avions")

# --- COLLECTE DES LABELS ---
labels = []
for filename in os.listdir(vignettes_dir):
    if filename.endswith(".jpg"):
        # On récupère le label (ex: "A1" dans "A1_image_0_1.jpg")
        # On prend tout ce qui est avant le premier underscore
        label = filename.split('_')[0]
        labels.append(label)

# --- COMPTAGE ET STATISTIQUES ---
stats = Counter(labels)
# Trier par nom de classe (A1, A2, ...)
sorted_classes = sorted(stats.items(), key=lambda x: (x[0][0], int(x[0][1:]) if x[0][1:].isdigit() else 0))
classes = [x[0] for x in sorted_classes]
counts = [x[1] for x in sorted_classes]

# --- VISUALISATION (EDA) ---
plt.figure(figsize=(12, 6))
bars = plt.bar(classes, counts, color='skyblue', edgecolor='navy')

# Ajouter les chiffres au-dessus des barres
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, yval, ha='center', va='bottom')

plt.title("Répartition des classes d'avions (Analyse Univariée)")
plt.xlabel("Classes d'avions (A1 à A20)")
plt.ylabel("Nombre de vignettes")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Sauvegarder le graphique pour votre rapport Word
plt.tight_layout()
plt.savefig(os.path.join(current_dir, "repartition_classes.png"))
print("Graphique sauvegardé sous 'repartition_classes.png'")

# Afficher les chiffres dans la console
print("\nDétail des classes :")
for c, count in sorted_classes:
    print(f"Classe {c} : {count} images")