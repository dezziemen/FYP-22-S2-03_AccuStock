from flask import Flask
from flaskr.models import db
import os


def create_app():
    path = os.path.abspath(os.path.dirname(__file__))
    
    app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
    app.config['SECRET_KEY'] = 'secret'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////' + os.path.join(path, 'accustock.db')
    db.init_app(app)

    from .views import views
    
    app.register_blueprint(views, url_prefix='/')

    return app
