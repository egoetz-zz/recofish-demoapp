import json

import flask
import pandas as pd
import timm as timm

import torch
import torchvision.transforms as transforms
import sqlite3
import os
# Linux: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# from flask import flash
from flask import g
from itertools import chain

IMAGE_SIZE = 224
NUM_CLASSES = 101

WORMS_BASE_URl = "https://www.marinespecies.org/aphia.php?p=taxdetails&id="  # Add id_worms from species table
DORIS_BASE_URL = "https://doris.ffessm.fr/ref/specie/"  # Add id_doris from species table


basic_fields = {'species.ID', 'nom_scientifique', 'nom_commun', 'famille', 'label'}
dict_map = {'bio': ['taille_adulte_min', 'taille_adulte_max', 'profondeur_habituelle_min', 'profondeur_habituelle_max', 'Colonne_d_eau', 'danger', 'mode_de_vie'],
            'reglementation': [],
            'pratiques': [],
            'sources_externes': ['id_DORIS']}
fields_needed = set(chain.from_iterable(dict_map.values())).union(basic_fields)


class Predictor:
    def __init__(self, model_path, model='efficientnet_b4', num_classes=NUM_CLASSES):
        try:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.data_transforms = transforms.Compose([
                transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        except OSError as err:
            print("OS error:", err)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")

    def predict_img(self, img, k=3):
        image = self.data_transforms(img.convert('RGB')).to(self.device)
        output = self.model(image.unsqueeze(0))
        vals, preds = torch.topk(output.data, k, 1)
        return vals, preds


predictor = None


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        print("DB PATH=", flask.current_app.config['DATABASE_PATH'])
        db = g._database = sqlite3.connect(flask.current_app.config['DATABASE_PATH'])
    return db


def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


# preprocess img and call the model to predict the k most likely classes
# return: the image, the k species, sorted by likelihood
def predict_img(img, filename, k=3):
    global predictor
    if predictor is None:
        predictor = Predictor(flask.current_app.config['MODEL_PATH'])

    vals, preds = predictor.predict_img(img, k)
    labels, probas = [int(label) for label in preds.numpy()[0]], [round(float(proba), 2) for proba in vals.numpy()[0]]
    lps = dict(zip(labels, probas))
    print("vals, preds= {}".format(lps))
    s = [str(label) for label in labels]
    query = "SELECT " + ",".join(fields_needed) + " FROM species LEFT JOIN families ON id_famille = families.ID WHERE label IN (" + ",".join(s) + ")"
    bdd = pd.read_sql_query(query, get_db())
    print(bdd[['ID', 'nom_scientifique', 'nom_commun', 'famille', 'label']])
    answers = pd.DataFrame(vals.numpy()[0], index=preds.numpy()[0], columns=["value"])
    species = answers.merge(bdd, how='left', left_index=True, right_on='label').set_index('ID')
    ids = species.index.tolist()
    print("species ids=", ids)
    ips = dict(zip(ids, probas))
    c = get_db().cursor()
    c.execute("INSERT INTO connections (img_file_name, prediction) VALUES (?, ?)", (filename, json.dumps(ips)))
    get_db().commit()
    return species, c.lastrowid


def store_selection(conn_id, id, sel):
    c = get_db().cursor()
    c.execute("UPDATE connections SET valid_ID=%i, valid_rank=%i WHERE ID=%i" % (id, sel, conn_id))
    get_db().commit()


# indices: None to get all images available (catalog), otherwise indices of the illustrative images to help validation
def load_model_images(species, indices=None):
    images = []
    for sp in species:
        path = str(sp) + '/'
        if indices is None:
            images.append([path + f for f in os.listdir(flask.current_app.config['IMAGE_PATH'] + path)])
        else:
            images.append(
                [path + str(sp) + '_' + str(indices[j]) + '.jpg' for j in range(len(indices))])
    return images


fields_left = ['ID', 'nom_commun', 'autres_noms', 'id_famille',
              'reglementation', 'danger', 'mode_de_vie',
              'entrainement_modele', 'A_entrainer', 'label', 'nb_images'
              'Cotes_françaises', 'Mediterranée', 'Atlantique', 'Manche', 'Mer du Nord',
              'Pêche_commerciale', 'IUCN_red_list',
              'citation_ouvrages', 'citation_doris']


def fetch_species_info(id):
    query = "SELECT " + ",".join(fields_needed) + " FROM species LEFT JOIN families ON id_famille = families.ID WHERE species.ID={}".format(id)
    bdd = pd.read_sql_query(query, get_db())
    print(bdd[['ID', 'nom_scientifique', 'nom_commun', 'famille', 'label']])
    full_dict = bdd.iloc[0].to_dict()
    info_dict = {'bio': {}, 'reglementation': {}, 'pratiques': {}, 'sources_externes': {}}
    info_dict['bio']['Famille'] = bdd.at[0, 'famille']
    for key, fields in dict_map.items():
        for field in fields:
            if full_dict.get(field) is not None:
                fld = field.replace('_', ' ').capitalize()
                if field.endswith('_max'):
                    fld = fld[:-4]
                    info_dict[key][fld] = "{}{}".format(info_dict[key].get(fld, " - "), full_dict[field])
                elif field.endswith('_min'):
                    fld = fld[:-4]
                    info_dict[key][fld] = "{}{}".format(full_dict[field], info_dict[key].get(fld, " - "))
                else:
                    info_dict[key][fld] = full_dict[field]
            else:
                print("Field '", field, "' ABSENT")
    if info_dict['bio'].get("Profondeur habituelle") is not None:
        info_dict['bio']["Profondeur habituelle"] += " m"
    if info_dict['bio'].get("Taille adulte") is not None:
        info_dict['bio']["Taille adulte"] += " cm"
    html = {}
    for key, fields in info_dict.items():
        concat = ""
        for k, v in fields.items():
            if k == "Id doris":
                k = "Lien vers fiche DORIS"
                v_txt = str(v) if v is not None else "Non référencé"
                v_id = str(v) if v is not None else ""
                v = "<a href='{}{}'>{}</a>".format(DORIS_BASE_URL, v_id, v_txt)
            concat += "<tr class='property'><td class='property-key'>{}</td><td class='property-value'>{}</td></tr>".format(
                k, v)
        html[key] = "<table class='properties'>{}</table>".format(concat)
    html['ns'] = full_dict['nom_scientifique']
    html['nc'] = full_dict['nom_commun']
    print(html)
    return html
