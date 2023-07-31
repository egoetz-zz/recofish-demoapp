from io import BytesIO

from flask import (
    Blueprint, render_template
)
from flask import request, redirect, Response, flash
from werkzeug.utils import secure_filename
from PIL import Image
from ..predict2 import predict_img, load_model_images, fetch_species_info, store_selection

VERSION = 1.1
K_TOP = 3  # Number of predictions shown
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
FIXED_NUMBER_OF_SLIDER_IMAGES = 3  # temporary code limitation. Should become dynamic

bp = Blueprint('main', __name__)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@bp.route('/')
def index():
    return render_template('main/index.html', title='Recofish progressive web app', version=VERSION)

connection_id = 0
@bp.route('/select_species')
def select():
    # Form has 3 + 1 validation buttons -> check which was hit
    sel, id = None, 0
    for i in range(1, K_TOP+1):
        if request.args.get('chose_' + str(i)) is not None:
            sel, id = i, int(request.args['id' + str(i)])
    if request.args.get('sp_no') is not None:
        sel = 0
    store_selection(connection_id, id, sel)
    if sel > 0:
        return redirect('/species/' + str(id))
    return index()  # None of the species were validated -> return Home


@bp.route('/species/<id>')
def show_info(id):
    infos = fetch_species_info(int(id))
    images = load_model_images([int(id)])[0]
    # for some species, we have less than the  illustrating images
    i = 0
    while len(images) < FIXED_NUMBER_OF_SLIDER_IMAGES:
        images.append(images[i])
        i += 1
    images2 = {}
    for i in range(FIXED_NUMBER_OF_SLIDER_IMAGES):
        images2['img' + str(i + 1)] = 'images/' + images[i]
    return render_template('species_info.html', info=infos, images=images2)


@bp.route('/classify', methods=['GET', 'POST'])
def submit_file():
    global connection_id
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
            species, connection_id = predict_img(img, filename)

            model_images: list[list[Image]] = load_model_images(species.index.tolist(), indices=(2, 3))
            predictions = {}
            for i in range(K_TOP):
                cur = species.reset_index().iloc[i]
                predictions['id' + str(i + 1)] = cur['ID']
                predictions['ns' + str(i + 1)] = cur['nom_scientifique']
                predictions['nc' + str(i + 1)] = cur['nom_commun']
                predictions['prob' + str(i + 1)] = "{:2.1f}".format(cur['value'])
                predictions['img' + str(i + 1) + '1'] = 'images/' + model_images[i][0]
                predictions['img' + str(i + 1) + '2'] = 'images/' + model_images[i][1]

            return render_template('success.html', predictions=predictions, img=filename)
    return render_template('main/index.html', title='Recofish-PWA')
