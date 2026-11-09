import os
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

class PlaneDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_dir = os.path.join(root, 'images')
        self.xml_dir = os.path.join(root, 'annotations')
        self.imgs = list(sorted(os.listdir(self.img_dir)))
        
        # Liste des 20 classes (Assure-toi que ce sont les noms exacts dans tes XML)
        self.classes = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 
                        'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20']

    def __getitem__(self, idx):
        # 1. Charger l'image
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        # 2. Charger l'annotation XML correspondante
        xml_name = self.imgs[idx].replace('.jpg', '.xml').replace('.png', '.xml')
        xml_path = os.path.join(self.xml_dir, xml_name)
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.classes:
                continue
            
            # Index de la classe (On commence à 1 car 0 = Background)
            labels.append(self.classes.index(name) + 1)
            
            # Coordonnées [xmin, ymin, xmax, ymax]
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        # Conversion en tenseurs PyTorch
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        
        # Calcul de l'aire (nécessaire pour Faster R-CNN)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)