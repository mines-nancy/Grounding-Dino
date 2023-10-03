from groundingdino.util.inference import load_model, load_image, predict, annotate
from torchvision.ops import box_convert
from moviepy.editor import VideoFileClip
from shapely import Polygon
from shapely.geometry import Point
from PIL import Image
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import cv2
import os
import uuid
import fnmatch
import time
import hashlib
from ultralytics import YOLO
import requests
from io import BytesIO


TEXT_PROMPT = "vehicle"
model = YOLO(
    "groundingdino/config/GroundingDINO_SwinB_cfg.py",
    "weights/groundingdino_swinb_cogcoor.pth"
)

phrase_colors = {
    "car": (255, 0, 0),    # bleu
    "van": (0, 255, 0),    # vert
    "truck": (0, 0, 255),  # rouge
    "bike": (0, 0, 128),   # marron    "moto": (0, 215, 255), # jaune
    "person": (193, 182, 255),  # rose
    "vehicle": (255, 255, 255),  # noir
}

def calcul_nb_video(chemin_dossier, extensions_video):
    total = 0
    total_video = 0
    print("démarrage", end="")
    for dossier, sous_dossiers, fichiers in os.walk(chemin_dossier):
        for extension in extensions_video:
            for fichier in fnmatch.filter(fichiers, extension):
                total_video += 1
                video = VideoFileClip(os.path.join(dossier, fichier))
                total += video.duration * video.fps
                video.close()
                print("." if total_video % 5 == 0 else "", end="")
    print()
    return total, total_video


def generate_hash_image(image_data):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(image_data)
    return sha256_hash.hexdigest()(85, 140), (75, 110), (360, 60)

def create_dir():
    PATH = os.path.join(os.path.expanduser("~"), "simulation_Dino")
    if not (os.path.exists(PATH)) and not os.path.isdir(PATH):
        os.makedirs(PATH)
    nbr = 1
    while True:
        new_path = os.path.join(PATH, "simulation_" + str(nbr))
        if not (os.path.exists(new_path)) and not os.path.isdir(new_path):
            os.makedirs(new_path)
            os.makedirs(os.path.join(new_path, "image"))
            os.makedirs(os.path.join(new_path, "super_image"))
            os.makedirs(os.path.join(new_path, "label"))

            break
        nbr += 1
    return new_path


def gestion_dir(Path, image_modif, image, txt):
    #uid = str(uuid.uuid4()) ancien systeme avec UID, maintenant on utilise un hash
    uid = generate_hash_image(image)
    cv2.imwrite(os.path.join(Path + "/super_image", uid + ".jpg"), image_modif)
    cv2.imwrite(os.path.join(Path + "/image", uid + ".jpg"), image)
    with open(os.path.join(Path + "/label",  uid + ".txt"), "w") as fichier:
        fichier.write(txt)


def compute(image_source, image, BOX_TRESHOLD, boxes, logits, phrases):
    roi_polygon = Polygon([(0, 200), (330, 95), (85, 140), (75, 110), (360, 60), (385, 100), (240, 478), (0, 478)])
    img_h, img_w, _ = image_source.shape

    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

    data = []
    for (xyxy, logit, phrase) in zip(xyxy_boxes.numpy(), logits.numpy(), phrases):
        x1, y1, x2, y2 = map(int, xyxy)
        w = abs(x2-x1)
        h = abs(y2-y1)
        if logit >= BOX_TRESHOLD and w*h < 0.3 * img_h * img_w and roi_polygon.contains(Point(x1, y1)) or roi_polygon.contains(Point(x2, y2)):
            data.append(((x1, y1), (x2, y2), logit, phrase))

    new_data = []
    already_merged_ids = set()
    for i in range(0, len(data)):
        if i not in already_merged_ids:
            merge = []
            data_i = data[i]
            x1_i = data_i[0][0]
            x2_i = data_i[1][0]
            y1_i = data_i[0][1]
            y2_i = data_i[1][1]
            for j in range(i + 1, len(data)):
                if j not in already_merged_ids:
                    data_j = data[j]
                    x1_j = data_j[0][0]
                    x2_j = data_j[1][0]
                    y1_j = data_j[0][1]
                    y2_j = data_j[1][1]
                    rect_i = Polygon([[x1_i, y1_i], [x2_i, y1_i], [x2_i, y2_i], [x1_i, y2_i]])
                    rect_j = Polygon([[x1_j, y1_j], [x2_j, y1_j], [x2_j, y2_j], [x1_j, y2_j]])
                    intersection = rect_i.intersection(rect_j)
                    if intersection.area == 0 or rect_i.area == 0 or rect_j.area == 0:
                        continue
                    if intersection.area / rect_i.area > 0.7 or intersection.area / rect_j.area > 0.7:
                        already_merged_ids.add(i)                        already_merged_ids.add(j)
                        merge.append(data_i)
                        merge.append(data_j)
            if len(merge) == 0:
                new_data.append(data_i)
            else:
                x1 = min([el[0][0] for el in merge])
                x2 = min([el[1][0] for el in merge])
                y1 = min([el[0][1] for el in merge])
                y2 = min([el[1][1] for el in merge])

                max_score = 0
                label = ""
                conf = 0
                for el in merge:
                    x1_j = el[0][0]
                    x2_j = el[1][0]
                    y1_j = el[0][1]
                    y2_j = el[1][1]
                    rect_j = Polygon([[x1_j, y1_j], [x2_j, y1_j], [x2_j, y2_j], [x1_j, y2_j]])

                    score = rect_j.area + el[2] * rect_j.area * 0.1
                    if score > max_score:
                        max_score = score
                        label = el[3]
                        conf = el[2]
                new_data.append(((x1, y1), (x2, y2), conf, label))

    new_data2 = []
    for ((x1, y1), (x2, y2), logit, phrase) in new_data:
        new_phrase = phrase
        if logit < 0.4:
            new_phrase = "vehicle"
        new_data2.append(((x1, y1), (x2, y2), logit, new_phrase))

    return new_data2

def calcul(path_Dir, cap, nb_fps, nb_img_total, total, image_recup, nb_video, total_video, temps_estime):
    video = VideoFileClip(os.path.join(dossier, fichier))
    fps = video.fps
    video.close()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            nb_img_total += 1
            if nb_img_total % (fps / nb_fps) != 0:
                continue
            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            image_source = Image.fromarray(frame.astype('uint8'), 'RGB')
            image = np.asarray(image_source)
            image_transformed, _ = transform(image_source, None)

            height, width, _ = image.shape
            boxes, logits, phrases = predict(
                model=model,
                image=image_transformed,
                caption=TEXT_PROMPT,
                box_threshold=0.15,
                text_threshold=0.15,
                remove_combined=True
            )
            results_035 = compute(frame, image, 0.2, boxes, logits, phrases)
            results_020 = compute(frame, image, 0.15, boxes, logits, phrases)

            if len(results_020) != 0 and float(len(results_035)) / len(results_020) > 0.6:
                image_recup += 1
                txt = ""
                for ((x1, y1), (x2, y2), logit, phrase) in results_035:
                    txt += str(x1) + ", " + str(y1) + ", " + str(x2) + ", " + str(y2) + ", " + str(logit) + ", " + phrase + "\n"
                    color = phrase_colors.get(phrase, (0, 0, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                #cv2.imshow("annotated_image", cv2.resize(frame, None, fx=2, fy=2))
                #cv2.waitKey(100)

                gestion_dir(path_Dir, frame, image, txt)
            print("{:.2f}".format((nb_img_total / total) * 100), "%", "  video:", nb_video, "/", total_video, "  image recup: ", image_recup, end="\n")

            if len(temps_estime) >= 10:
                temps_estime.pop(0)
            temps_estime.append((time.time() - start_time) * (total / nb_img_total - 1))
            average_estimate = sum(temps_estime) / len(temps_estime)

            temps_restant = (total / nb_img_total) * average_estimate
            print("Temps restant : ", time.strftime(" %H:%M:%S", time.gmtime(temps_restant)))

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    nb_video += 1
    return nb_img_total, image_recup, nb_video, temps_estime

chemin_dossier = "/home/pc_techlab_ia_2/workspace/GroundingDINO/"      # Mettre un dossier contenant les video a traiter
nb_fps = 0.3


if not (os.path.exists(chemin_dossier)) and not os.path.isdir(chemin_dossier):
    print("le dossier contenant les videos n'existent pas")
    exit(42)

extensions_video = ["*.mp4", "*.avi", "*.mkv", "*.mov"]
x = 0
image_recup = 0
nb_img_total = 0
nb_video = 0
start_time = time.time()
temps_estime = []

path_Dir = create_dir()
total, total_video = calcul_nb_video(chemin_dossier, extensions_video)

for dossier, sous_dossiers, fichiers in os.walk(chemin_dossier):
    for extension in extensions_video:
        for fichier in fnmatch.filter(fichiers, extension):
            nb_img_total, image_recup, nb_video, temps_estime = calcul(path_Dir, cv2.VideoCapture(os.path.join(dossier, fichier)), nb_fps, nb_img_total, total, image_recup, nb_video, total_video, temps_estime)
print("la simulation se trouve : ", path_Dir)
