import cv2
import os
import xml.etree.ElementTree as ET

# Configuration des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(current_dir, "student-dataset")
img_folder = os.path.join(base_path, "images")
xml_folder = os.path.join(base_path, "annotations")
output_folder = os.path.join(current_dir, "vignettes_avions")

os.makedirs(output_folder, exist_ok=True)

print("🛠️ Étape 1 : Extraction des 7 870 vignettes (Format Original)...")

count = 0
for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith('.xml'): continue
    
    # Correction du lien XML/Image pour sauver les 697 images
    base_name = os.path.splitext(xml_file)[0]
    img_path = os.path.join(img_folder, f"{base_name}.jpg")
    if not os.path.exists(img_path): 
        img_path = os.path.join(img_folder, f"{base_name}.png")
    
    img = cv2.imread(img_path)
    if img is None: continue

    # Conversion RGB pour la future compatibilité CNN
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    root = ET.parse(os.path.join(xml_folder, xml_file)).getroot()
    for i, obj in enumerate(root.findall('object')):
        label = obj.find('name').text
        box = obj.find('bndbox')
        
        xmin, ymin = int(box.find('xmin').text), int(box.find('ymin').text)
        xmax, ymax = int(box.find('xmax').text), int(box.find('ymax').text)
        
        # Découpage de l'avion selon ses coordonnées réelles
        crop = img[ymin:ymax, xmin:xmax]
        
        if crop.size > 0:
            # On ne fait plus de resize ici (on garde la taille native)
            save_name = f"{label}_{base_name}_{i}.jpg"
            # Sauvegarde (OpenCV nécessite BGR pour écrire le fichier correctement)
            cv2.imwrite(os.path.join(output_folder, save_name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            count += 1

print(f"✅ Terminé : {count} vignettes extraites avec succès.")