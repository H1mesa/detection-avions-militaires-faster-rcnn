import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ====================================================
# C'EST LE "PLAN DE CONSTRUCTION" UNIQUE DE TON IA
# ====================================================

def get_model_instance_segmentation(num_classes):
    """
    Crée le modèle Faster R-CNN avec un backbone ResNet50 pré-entraîné.
    Args:
        num_classes (int): Nombre total de classes (tes avions + 1 pour le fond)
    """
    # 1. Charger le modèle pré-entraîné sur COCO (pour avoir une base solide)
    # weights='DEFAULT' est la nouvelle façon de dire pretrained=True
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # 2. Remplacer la "tête" de classification (le dernier étage)
    # pour qu'elle corresponde à TON nombre de classes d'avions.
    
    # On récupère le nombre de neurones d'entrée de la tête actuelle
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # On remplace la tête par une nouvelle, adaptée à ton dataset
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# --- Configuration pour ton projet ---
# 20 types d'avions + 1 classe "Background" (le fond, rien)
NUM_CLASSES_PROJECT = 20 + 1 

print("✅ model.py chargé avec succès. Prêt à être importé.")