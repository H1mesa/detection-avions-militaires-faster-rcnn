import cv2
import os
import xml.etree.ElementTree as ET

# --- CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(current_dir, "student-dataset")
img_folder = os.path.join(base_path, "images")
xml_folder = os.path.join(base_path, "annotations")
output_folder = os.path.join(current_dir, "vignettes_avions")

os.makedirs(output_folder, exist_ok=True)

print("🛠️  Récupération intelligente des vignettes en cours...")

count = 0
erreurs_fatales = 0

for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith('.xml'): continue
    
    try:
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()
        
        # --- RÉPARATION DU NOM DE L'IMAGE ---
        # On récupère le nom de base (ex: '104') sans l'extension
        base_name = os.path.splitext(xml_file)[0]
        
        # On cherche l'image en priorité avec .jpg ou .png
        image_name = f"{base_name}.jpg"
        image_path = os.path.join(img_folder, image_name)
        
        # Si le .jpg n'existe pas, on tente le .png
        if not os.path.exists(image_path):
            image_name = f"{base_name}.png"
            image_path = os.path.join(img_folder, image_name)

        # Lecture de l'image
        img = cv2.imread(image_path)
        
        if img is None:
            erreurs_fatales += len(root.findall('object'))
            continue

        # --- CONVERSION RGB (Pour le futur CNN) ---
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i, obj in enumerate(root.findall('object')):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            
            xmin, ymin = int(bndbox.find('xmin').text), int(bndbox.find('ymin').text)
            xmax, ymax = int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)
            
            crop = img[ymin:ymax, xmin:xmax]
            
            if crop.size > 0:
                crop = cv2.resize(crop, (224, 224))
                save_name = f"{label}_{base_name}_{i}.jpg"
                # Sauvegarde en format compatible OpenCV (BGR)
                cv2.imwrite(os.path.join(output_folder, save_name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                count += 1
                
    except Exception as e:
        print(f"Erreur sur {xml_file}: {e}")

print(f"\n✅ Terminé !")
print(f"✈️  Vignettes sauvées : {count}")
print(f"❌ Vignettes définitivement perdues : {erreurs_fatales}")