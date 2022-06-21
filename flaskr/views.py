from flask import Blueprint, request, render_template
from .finance import CompanyStock
from .training import LSTMPrediction
import matplotlib.pyplot as plt
import pandas as pd
import time


views = Blueprint('views', __name__)

TABLE_RESPONSIVE_CLASS = ['table', 'table-striped', 'table-hover', 'table-bordered']


# Home page
@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html')
    return stock(request.form.get('searchSymbol'))


# View stock page
@views.route('/stock/<string:symbol>')
def stock(symbol):
    company = CompanyStock(symbol)

    # If company stock symbol does not exist
    if company.get_symbol() is None:
        return render_template('home.html', searchError='Error: Stock symbol does not exist.')

    history = company.get_history().reset_index(level='Date')                       # Convert Date index to column
    history['Time'] = history['Date']
    history['Date'] = pd.to_datetime(history['Date']).dt.strftime('%d %b %Y')       # Convert Timestamp to Datetime

    # Get news and convert timestamp to datetime
    news = company.get_news()
    for article in news:
        article['providerPublishTime'] = pd.to_datetime(article.get('providerPublishTime'), unit='s').strftime('%d %b %Y, %H:%M:%S')

    return render_template(
        'stock.html',
        company_symbol=company.get_symbol(),
        company=company.get_info('longName'),
        table=history.loc[:, history.columns != 'Time'].to_html(classes=TABLE_RESPONSIVE_CLASS, justify='left'),
        titles=history.columns.values,
        news=news,
        data=history.to_json()
    )


# Forecast button
@views.route('/forecast/<string:symbol>/<string:type>')
def forecast(symbol, type):
    time_now = time.time()
    company = CompanyStock(symbol)
    data = company.get_item(type)
    prediction = LSTMPrediction(data)

    # Prepare and start prediction
    look_back, x_train, x_test, y_train, y_test, test_data = prediction.reshape()
    model = prediction.prepare_model(look_back, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    prediction.train(model, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, test_data=test_data)
    prediction.predict(days=30, model=model, test_data=test_data)

    folder_name = 'flaskr/static/images/'
    graph_filename = f'{str(time_now)}.png'
    plt.savefig(folder_name + graph_filename)
    return render_template(
        'forecast.html',
        company=company.get_info('longName'),
        type=type,
        table=data.to_html(classes=TABLE_RESPONSIVE_CLASS, justify='left'),
        graph_filename='/images/' + graph_filename,
    )
