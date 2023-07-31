import os
from flask import Flask, url_for
import sqlite3
# from flask_sqlalchemy import SQLAlchemy

UPLOAD_FOLDER = 'static/uploads/'
IMAGE_FOLDER = 'static/images/'
DATA_FOLDER = 'data/'
DATABASE_PATH = DATA_FOLDER + 'bdd_poissons.db'
MODEL_PATH = 'recofish_classification_model.pt'

app, db = None, None
def create_app():
    global app, db
    app = Flask(
        __name__,
        instance_relative_config=True,
        # static_url_path=''
    )
    root_path = os.path.dirname(app.instance_path) + '/app/'

    app.config['UPLOAD_PATH'] = root_path + UPLOAD_FOLDER
    app.config['IMAGE_PATH'] = root_path + IMAGE_FOLDER
    app.config['DATABASE_PATH'] = root_path + DATABASE_PATH
    app.config['MODEL_PATH'] = root_path + MODEL_PATH
    #app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + DATABASE_PATH
    # db = SQLAlchemy(app)


    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY') or 'you-will-never-guess',
        # SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    # try:
    #     os.makedirs(app.instance_path)
    # except OSError:
    #     pass

    from flask_sslify import SSLify
    if 'DYNO' in os.environ:  # only trigger SSLify if the app is running on Heroku
        sslify = SSLify(app)

    # from app.model import db, migrate
    # db.init_app(app)
    # migrate.init_app(app, db)

    from app.controller import (
        main, pwa
    )
    app.register_blueprint(main.bp)
    app.register_blueprint(pwa.bp)

    return app, db
