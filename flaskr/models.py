from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Search(db.Model):
    __tablename__ = 'search'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    time = db.Column(db.Integer, nullable=False)
    search_term = db.Column(db.String(), nullable=False)


class PredictedTable(db.Model):
    __tablename__ = 'predicted_table'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    time = db.Column(db.Integer, nullable=False)
    search_term = db.Column(db.String(), nullable=False)
    
    
class PredictedRow(db.Model):
    __tablename__ = 'predicted_row'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    time = db.Column(db.Integer, nullable=False)
    value = db.Column(db.Float, nullable=False)
    table_id = db.Column(db.Integer, db.ForeignKey('predicted_table.id'))