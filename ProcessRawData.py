import math
import os
from PIL import Image
import numpy as np
from collections import defaultdict
from random import shuffle

# affectnet
affectnet_labels = {
    '0' : 'Anger',
    '1' : 'Contempt',
    '2' : 'Disgust',
    '3' : 'Fear',
    '4' : 'Happy',
    '5' : 'Neutral',
    '6' : 'Sad',
    '7' : 'Surprise'
}

affect_data = defaultdict(list)
for type in ["train", "test", "valid"]:
    for fileName in os.listdir(f"./rawdataset/AffectNet/{type}/images"):
        with Image.open(os.path.join(f"./rawdataset/AffectNet/{type}/images", fileName), mode="r") as image:
            data = image.copy()
        
        with open(os.path.join(f"./rawdataset/AffectNet/{type}/labels", fileName.split(".")[0]+".txt"), mode="r") as file:
            labels = file.read()
            label = labels.split(" ")[0]

        affect_data[affectnet_labels[label]].append((fileName, data))


# raf
raf_labels = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happy",
    5: "Sad",
    6: "Anger",
    7: "Neutral"
}
raf_data = defaultdict(list)
for type in ["train", "test"]:
    for label in range(1, 8):
        for fileName in os.listdir(f"./rawdataset/RAF/DATASET/{type}/{label}"):
            with Image.open(os.path.join(f"./rawdataset/RAF/DATASET/{type}/{label}", fileName), mode="r") as image:
                data = image.copy()

            raf_data[raf_labels[label]].append((fileName, data))


# sfew
sfew_data = defaultdict(list)
for type in ["Train", "Validation"]:
    for label in ["Surprise","Fear","Disgust","Happy","Sad","Angry","Neutral"]:
        for fileName in os.listdir(f"./rawdataset/SFEW/{type}/{label}"):
            with Image.open(os.path.join(f"./rawdataset/SFEW/{type}/{label}", fileName), mode="r") as image:
                data = image.copy()
            if label == "Angry":
                sfew_data["Anger"].append((fileName, data))
            else:
                sfew_data[label].append((fileName, data))


# train, valid, test 0.8 0.1 0.1
flag = 1
for data in [affect_data, raf_data, sfew_data]:
    if flag == 1:
        dataset_name = "AffectNet"
    elif flag == 2:
        dataset_name = "RAF"
    else:
        dataset_name = "SFEW"

    for label, images in data.items():
        shuffle(images)
        valid_index, test_index = math.floor(0.8 * len(images)), math.floor(0.9 * len(images))
        print(label, len(images), valid_index, test_index)
        for index, image in enumerate(images):

            if index < valid_index:
                outdir = f"./Dataset/{dataset_name}/Train/{label}"
            elif index < test_index:
                outdir = f"./Dataset/{dataset_name}/Valid/{label}"
            else:
                outdir = f"./Dataset/{dataset_name}/Test/{label}"
        
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            image[1].save(os.path.join(outdir, image[0]))
    
    flag += 1


