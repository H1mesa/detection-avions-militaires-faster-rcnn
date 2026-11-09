import os
import xml.etree.ElementTree as ET

# --- CHEMINS DÉDUITS DE TA CAPTURE D'ÉCRAN ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# On cherche le dossier 'student-dataset' qui contient 'annotations'
base_path = os.path.join(current_dir, "student-dataset")
xml_folder = os.path.join(base_path, "annotations")

if not os.path.exists(xml_folder):
    # Deuxième chance : peut-être que le script est un niveau au-dessus
    base_path = os.path.join(current_dir, "military-aircraft-recognition-dataset.train", "student-dataset")
    xml_folder = os.path.join(base_path, "annotations")

# --- COMPTAGE ---
if os.path.exists(xml_folder):
    total_objets = 0
    fichiers_scannes = 0
    
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(xml_folder, xml_file))
            root = tree.getroot()
            
            # On additionne chaque balise <object> trouvée
            nb_dans_ce_fichier = len(root.findall('object'))
            total_objets += nb_dans_ce_fichier
            fichiers_scannes += 1
            
    print(f"✅ Analyse réussie !")
    print(f"📁 Dossier analysé : {xml_folder}")
    print(f"📄 Nombre de fichiers XML : {fichiers_scannes}")
    print(f"✈️  NOMBRE TOTAL D'OBJETS (Somme) : {total_objets}")
else:
    print(f"❌ ERREUR : Impossible de trouver le dossier 'annotations'.")
    print(f"Vérifiez que le dossier 'student-dataset' est bien à côté de votre script.")