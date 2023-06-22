import flask
import numpy as np
import pandas as pd
import timm as timm

import torch
import torchvision.transforms as transforms

import os
# Linux: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# from flask import flash

IMAGE_SIZE = 224
NUM_CLASSES = 101

WORMS_BASE_URl = "https://www.marinespecies.org/aphia.php?p=taxdetails&id="  # Add id_worms from species table
DORIS_BASE_URL = "https://doris.ffessm.fr/ref/specie/"  # Add id_doris from species table

bdd: pd.DataFrame = None
# bdd = pd.read_csv(app.config['APPLICATION_ROOT'] + 'data/species.csv', index_col='nom_scientifique')
families: pd.DataFrame = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=NUM_CLASSES)

data_transforms = transforms.Compose([
    transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def init():
    global bdd, families, model, data_transforms, device
    try:
        app = flask.current_app
        root_path = os.path.dirname(app.instance_path)
        bdd = pd.read_csv(root_path + '/app/data/species.csv', index_col='nom_scientifique')
        families = pd.read_csv(root_path + '/app/data/families.csv', index_col='ID')
        model.load_state_dict(
            torch.load(root_path + '/app/recofish_classification_model.pt', map_location=device))

        model.eval()

        return True
    except OSError as err:
        print("OS error:", err)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        return False


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


def load_model_images(species, indices=None):
    images = []
    root_path = os.path.dirname(flask.current_app.instance_path)
    for sp in species:
        path = 'images/' + str(sp) + '/'
        if indices is None:
            images.append([path + f for f in os.listdir(root_path + '/app/static/' + path)])
        else:
            images.append(
                [path + str(sp) + '_' + str(indices[j]) + '.jpg' for j in range(len(indices))])
    return images


fields_left = ['ID', 'nom_commun', 'autres_noms', 'id_famille',
              'taille_adulte_₋', 'taille_adulte_₊', 'profondeur_habituelle_-', 'profondeur_habituelle_₊', 'Colonne_d_eau',
              'reglementation', 'danger', 'mode_de_vie',
              'entrainement_modele', 'A_entrainer', 'label', 'nb_images'
              'Cotes_françaises', 'Mediterranée', 'Atlantique', 'Manche', 'Mer du Nord',
              'Pêche_commerciale', 'IUCN_red_list',
              'citation_ouvrages', 'citation_doris', 'lien_fiche_doris']


def fetch_species_info(id):
    if bdd is None:
        init()
    full_dict = bdd.loc[bdd.ID == id].iloc[0].dropna().to_dict()
    dict_map = {'bio': ['taille_adulte_₋', 'taille_adulte_₊', 'profondeur_habituelle_-', 'profondeur_habituelle_₊', 'Colonne_d_eau', 'danger', 'mode_de_vie'],
                'reglementation': [],
                'pratiques': []}
    info_dict = {}
    for key, fields in dict_map.items():
        info_dict[key] = {}
        for field in fields:
            if full_dict.get(field):
                fld = field.replace('_', ' ').capitalize()
                if fld.endswith('₊'):
                    fld = fld[:-2]
                    info_dict[key][fld] = "{} - {}".format(info_dict[key].get(fld, ""), full_dict[field])
                elif fld.endswith('-') or fld.endswith('₋'):
                    fld = fld[:-2]
                    info_dict[key][fld] = "{}{}".format(full_dict[field], info_dict[key].get(fld, ""))
                else:
                    info_dict[key][fld] = full_dict[field]
                print("   field=", field, " -> ", fld)
                print("   ", info_dict)
        if info_dict[key].get("Profondeur habituelle") is not None:
            info_dict[key]["Profondeur habituelle"] += " m"
        elif info_dict[key].get("Taille adulte") is not None:
            info_dict[key]["Profondeur habituelle"] += " cm"
    info_dict['bio']['famille'] = families.iloc[full_dict['id_famille']]
    return info_dict
