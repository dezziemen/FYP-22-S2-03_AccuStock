from flask import Blueprint, request, render_template
from .finance import CompanyStock
from .training import LSTMPrediction
import pandas as pd
import time
from flaskr.models import db, Search

views = Blueprint('views', __name__)

TABLE_RESPONSIVE_CLASS = ['table', 'table-striped', 'table-hover', 'table-bordered']


# Home page
@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # aapl = Search(time=time.time(), search_term='aapl')
        # msft = Search(time=time.time(), search_term='msft')
        # print(Search.query.all())
        # db.session.add(aapl)
        # db.session.add(msft)
        # db.session.commit()
        # print(Search.query.all())
        return render_template('home.html')
    return stock(request.form.get('search_symbol'))


# View stock page
@views.route('/stock/<string:symbol>')
def stock(symbol):
    company = CompanyStock(symbol)

    # If company stock symbol does not exist
    if company.get_symbol() is None:
        return render_template('home.html', search_error='Error: Stock symbol does not exist.')

    history = company.get_history().reset_index(level='Date')                       # Convert Date index to column
    history['Time'] = history['Date']                                               # Create Time column
    history['Date'] = pd.to_datetime(history['Date']).dt.strftime('%d %b %Y')       # Convert Timestamp to Datetime

    # Get news and convert timestamp to datetime
    news = company.get_news()
    for article in news:
        article['providerPublishTime'] = pd.to_datetime(article.get('providerPublishTime'), unit='s').strftime('%d %b %Y, %H:%M:%S')

    return render_template(
        'stock.html',
        company_symbol=company.get_symbol(),
        company=company.get_info('longName'),
        table=history.loc[:, history.columns != 'Time'].to_html(classes=TABLE_RESPONSIVE_CLASS, justify='left'),        # Exclude 'Time' column
        # titles=history.columns.values,
        news=news,
        data=history.to_json(),
    )


# Forecast button
@views.route('/forecast/<string:symbol>/<string:type>')
def forecast(symbol, type):
    time_now = time.time()
    company = CompanyStock(symbol)
    data = company.get_item(type)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%d %b %Y')       # Convert Timestamp to Datetime
    prediction = LSTMPrediction(data)

    # Start prediction
    folder_name = 'flaskr/static/images/'
    graph_filename = f'{str(time_now)}_{symbol}_{type}.png'             # Save time, symbol, and type
    prediction.start(days=30, fig_path=folder_name + graph_filename)    # Start prediction and save figure

    return render_template(
        'forecast.html',
        company=company.get_info('longName'),
        type=type,
        table=data.to_html(classes=TABLE_RESPONSIVE_CLASS, justify='left'),
        graph_filename='/images/' + graph_filename,
    )
