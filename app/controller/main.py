from io import BytesIO

from flask import (
    Blueprint, render_template
)
from flask import request, redirect, flash, url_for
import requests
from werkzeug.utils import secure_filename
from PIL import Image
from ..predict import predict_img, load_model_images
from app import app
import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

k = 3

bp = Blueprint('main', __name__)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/')
def index():
    return render_template('main/index.html', title='Recofish-PWA')

@bp.route('/species')
def show_info():
    return render_template('main/species_info.html', title='Recofish-PWA')


@bp.route('/classify', methods=['GET', 'POST'])
def submit_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            flash('Submitting file ' + filename)
            # img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # file.save(img_path)
            # img = Image.open(img_path)
            img = Image.open(BytesIO(file.read())).convert('RGB')
            species = predict_img(img)

            model_images: list[list[Image]] = load_model_images(species, indices=(2, 3))
            predictions = {}
            for i in range(k):
                cur = species.reset_index().iloc[i]
                predictions['ns' + str(i + 1)] = cur['nom_scientifique']
                predictions['nc' + str(i + 1)] = cur['nom_commun']
                predictions['prob' + str(i + 1)] = "{:2.1f}".format(cur['value'])
                predictions['img' + str(i + 1) + '1'] = model_images[i][0]
                predictions['img' + str(i + 1) + '2'] = model_images[i][1]

            return render_template('success.html', predictions=predictions, img=filename)
    return render_template('main/index.html', title='Recofish-PWA')


def getclasses(image: Image.Image):
    args = request.args
    return predict_img(args)
