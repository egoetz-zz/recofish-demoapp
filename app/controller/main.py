from io import BytesIO

from flask import (
    Blueprint, render_template
)
from flask import request, redirect, Response, flash
from werkzeug.utils import secure_filename
from PIL import Image
from ..predict import predict_img, load_model_images, fetch_species_info

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
    if True:
        return render_template('main/index.html', title='Recofish-PWA')
    else:
        return show_info(55)


@bp.route('/select_species')
def select():
    for i in range(1, K_TOP+1):
        if request.args.get('chose_' + str(i)) is not None:
            return redirect('/species/' + request.args['id' + str(i)])


@bp.route('/species/<id>')
def show_info(id):
    infos = fetch_species_info(int(id))
    images = load_model_images([int(id)])[0]
    while len(images) < FIXED_NUMBER_OF_SLIDER_IMAGES:  # for some species, we have less than 3 illustrating images
        images.append(images[0])
    images2 = {}
    for i in range(FIXED_NUMBER_OF_SLIDER_IMAGES):
        images2['img' + str(i + 1)] = images[i]
    return render_template('species_info.html', info=infos, images=images2)


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

            model_images: list[list[Image]] = load_model_images(species.index.tolist(), indices=(2, 3))
            predictions = {}
            for i in range(K_TOP):
                cur = species.reset_index().iloc[i]
                predictions['id' + str(i + 1)] = cur['ID']
                predictions['ns' + str(i + 1)] = cur['nom_scientifique']
                predictions['nc' + str(i + 1)] = cur['nom_commun']
                predictions['prob' + str(i + 1)] = "{:2.1f}".format(cur['value'])
                predictions['img' + str(i + 1) + '1'] = model_images[i][0]
                predictions['img' + str(i + 1) + '2'] = model_images[i][1]

            return render_template('success.html', predictions=predictions, img=filename)
    return render_template('main/index.html', title='Recofish-PWA')
