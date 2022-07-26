from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Search(db.Model):
    __tablename__ = 'search'
    row_id = db.Column('id', db.Integer, primary_key=True, autoincrement=True)
    time = db.Column(db.Integer, nullable=False)
    search_term = db.Column(db.String(), nullable=False)


class PredictedTable(db.Model):
    __tablename__ = 'predicted_table'
    row_id = db.Column('id', db.Integer, primary_key=True, autoincrement=True)
    time = db.Column(db.Integer, nullable=False)
    symbol = db.Column(db.String(), nullable=False)
    stock_type = db.Column(db.String(), nullable=False)

    
class PredictedRow(db.Model):
    __tablename__ = 'predicted_row'
    row_id = db.Column('id', db.Integer, primary_key=True, autoincrement=True)
    time = db.Column(db.Integer, nullable=False)
    value = db.Column(db.Float, nullable=False)
    table_id = db.Column(db.Integer, db.ForeignKey('predicted_table.id'), nullable=False)

    def __str__(self):
        return f'{self.row_id=}\n{self.time=}\n{self.value=}\n{self.table_id=}\n'

    def __repr__(self):
        return self.__str__()
