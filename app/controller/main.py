from flask import (
    Blueprint, render_template
)
from flask import request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from ..predict import predict_img, load_model_images
import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = '/static/uploads'

k=3

bp = Blueprint('main', __name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#@bp.route('/')
#def index():
#    return render_template('main/index.html', title='Recofish-PWA')

@bp.route('/classify', methods=['GET', 'POST'])
def submit_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            img_path = redirect(url_for('download_file', name=filename))
            img = Image.open(img_path)
            species = predict_img(img)

            model_images: list[Image] = load_model_images(species)
            predictions = {}
            for i in range(k):
                cur = species.reset_index().iloc[i]
                predictions['ns' + str(i+1)] = cur['nom_scientifique']
                predictions['nc' + str(i+1)] = cur['nom_commun']
                predictions['prob' + str(i+1)] = cur['value']
                predictions['img' + str(i+1)] = model_images[i]

            return render_template('success.html', predictions=predictions)
    return render_template('main/index.html', title='Recofish-PWA')

def getclasses(image: Image.Image):
    args = request.args
    return predict_img(args)

