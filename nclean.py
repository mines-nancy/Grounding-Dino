# Importez les bibliothèques nécessaires
import torch
from PIL import Image
from pathlib import Path
import requests


# Importez YOLOv8 de Ultralytics
from yolov5 import YOLO

# Chargez le modèle YOLOv8 pré-entraîné (choisissez le modèle approprié)
model = YOLO(weights='yolov5s.pt')  # Utilisez yolov5s.pt, yolov5m.pt, yolov5l.pt ou yolov5x.pt en fonction de vos besoins

# Chargez la vidéo (remplacez "video.mp4" par le chemin de votre vidéo)
video_path = 'video.mp4'

# Ouvrez la vidéo en utilisant OpenCV
import cv2
video_capture = cv2.VideoCapture(video_path)

# Boucle à travers chaque image de la vidéo
while True:
    # Capturez l'image suivante de la vidéo
    ret, frame = video_capture.read()
    if not ret:
        break  # Sortez de la boucle si la vidéo est terminée
    
    # Convertissez l'image en format compréhensible par YOLOv8 (Numpy array)
    img = Image.fromarray(frame[:, :, ::-1])  # BGR to RGB
    img = model.preprocess(img)  # Pré-traitez l'image pour YOLOv8
    
    # Effectuez la détection d'objets
    results = model(img)
    
    # Dessinez les boîtes englobantes et les étiquettes sur l'image d'origine
    annotated_img = results.render()[0]
    
    # Affichez l'image annotée
    cv2.imshow('YOLOv8 Object Detection', annotated_img[:, :, ::-1])  # RGB to BGR
    
    # Appuyez sur la touche 'q' pour quitter la vidéo
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérez la capture vidéo et fermez la fenêtre d'affichage
video_capture.release()
cv2.destroyAllWindows()
