# app.py: Script that Flask uses to run webapp

from flaskr import create_app
from flaskr.models import db

app = create_app()

if __name__ == '__main__':
    db.create_all()         # Initialize database
    app.run()               # Run app
