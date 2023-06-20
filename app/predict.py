import numpy as np
import pandas as pd
import timm as timm
from PIL import Image

import torch
import torchvision.transforms as transforms
from PIL.Image import Image
from matplotlib import pyplot as plt

from flask import url_for
import os
# Linux: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
from flask import flash

IMAGE_SIZE = 224
NUM_CLASSES = 101

bdd = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=NUM_CLASSES)

data_transforms = transforms.Compose([
    transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

from app import app


def init():
    global bdd, model, data_transforms, device
    try:
        #        bdd = pd.read_csv(url_for('static', filename='data/species.csv'), index_col='nom_scientifique')
        # bdd = pd.read_csv(app.config['APPLICATION_ROOT'] + 'data/species.csv', index_col='nom_scientifique')
        print(app.config['APPLICATION_ROOT'])
        print(app.instance_path)
        bdd = pd.read_csv(app.instance_path + '/../app/data/species.csv', index_col='nom_scientifique')
        model.load_state_dict(
            torch.load(app.instance_path + '/../app/recofish_classification_model.pt', map_location=device))

        model.eval()

        return True
    except OSError as err:
        print("OS error:", err)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        return False


plt.rc('font', size=8)


# preprocess img and call the model to predict the k most likely classes
# return: the image, the k species, sorted by likelihood
def predict_img(img, k=3):
    if bdd is None:
        init()
    image = data_transforms(img.convert('RGB')).to(device)
    output = model(image.unsqueeze(0))
    vals, preds = torch.topk(output.data, k, 1)
    # print(zip(vals.numpy()[0], preds.numpy()[0]))
    species = pd.DataFrame(vals.numpy()[0], index=preds.numpy()[0], columns=["value"]) \
        .merge(bdd, how='left', left_index=True, right_on='label')
    # .set_index('nom_scientifique')
    return species


def show_topk(img, k=3):
    fig = plt.figure(figsize=(10, 7))
    columns, rows = k, 1

    with torch.no_grad():
        species = predict_img(img)
        model_images: list[Image] = load_model_images(species)
        for i in range(k):
            cur = species.reset_index().iloc[i]
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(np.asarray(model_images[i]))
            plt.axis('off')
            plt.title("{} ({}) : {:.1f}%".format(cur['nom_scientifique'], cur['nom_commun'], cur['value']))


# Display results


def load_model_images(species, indices=(2, 3, 4)):
    images = []
    for _, sp in species.iterrows():
        images.append(
            ['images/' + str(sp['ID']) + '/' + str(sp['ID']) + '_' + str(indices[j]) + '.jpg' for j in range(len(indices))])
    return images


# Display species info
WORMS_BASE_URl = "https://www.marinespecies.org/aphia.php?p=taxdetails&id="  # Add id_worms from species table
DORIS_BASE_URL = "https://doris.ffessm.fr/ref/specie/"  # Add id_doris from species table

# Pour les familles
# familles = pd.read_csv(data_dir +"/familles.csv", index_col='ID')
