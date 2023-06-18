import glob
from pathlib import Path

import numpy as np
import pandas as pd
import timm as timm
from PIL import Image

import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
# Linux: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

PATH = "."
img_dir = Path(PATH + '/images')
data_dir = Path(PATH + '/data')
IMAGE_SIZE = 224

bdd = pd.read_csv(data_dir +"/species.csv", index_col='nom_scientifique')

# Prediction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=101)
model.load_state_dict(torch.load(PATH+"/recofish_classification_model.pt", map_location=device))

model.eval()
# # Model Prediction
data_transforms = transforms.Compose([
    transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

plt.rc('font', size=8)

# preprocess img and call the model to predict the k most likely classes
# return: the image, the k species, sorted by likelihood
def predict_img(img, k=3):
    image = data_transforms(img.convert('RGB')).to(device)
    output = model(image.unsqueeze(0))
    vals, preds = torch.topk(output.data, k, 1)
    # print(zip(vals.numpy()[0], preds.numpy()[0]))
    species = pd.DataFrame(vals.numpy()[0], index=preds.numpy()[0], columns=["value"])\
        .merge(bdd, how='left', left_index=True, right_on='label')
        # .set_index('nom_scientifique')
    return species

    # return {classes[i]: float(prediction[i]) for i in range(num_classes)}
    return img, img_name, best_k, values


def show_topk(img, k=3):
    fig = plt.figure(figsize=(10, 7))
    columns, rows = k, 1

    with torch.no_grad():
        species = predict_img(img)
        model_images = load_model_images(species)
        for i in range(k):
            cur = species.reset_index().iloc[i]
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(np.asarray(model_images[i]))
            plt.axis('off')
            plt.title("{} ({}) : {:.1f}%".format(cur['nom_scientifique'], cur['nom_commun'], cur['value']))

# Display results
import os
def load_model_images(species):
    images = [Image.open(os.path.join(img_dir, sp['ID'], sp['class_img'])) for sp in species.iterrows()]
    return images

# Display species info
WORMS_BASE_URl = "https://www.marinespecies.org/aphia.php?p=taxdetails&id="  # Add id_worms from species table
DORIS_BASE_URL = "https://doris.ffessm.fr/ref/specie/"  # Add id_doris from species table

# Pour les familles
# familles = pd.read_csv(data_dir +"/familles.csv", index_col='ID')
