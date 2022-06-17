from flask import Flask, render_template, request
from finance import CompanyStock
from training import LSTMPrediction
import pandas as pd

app = Flask(__name__)

TABLE_RESPONSIVE_CLASS = ['table', 'table-striped', 'table-hover', 'table-bordered']


# Home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html')
    return stock(request.form.get('searchSymbol'))


# View stock page
@app.route('/stock/<string:symbol>')
def stock(symbol):
    company = CompanyStock(symbol)

    # If company stock symbol does not exist
    if company.get_symbol() is None:
        return render_template('home.html', searchError='Error: Stock symbol does not exist.')

    history = company.get_history().reset_index(level='Date')                       # Convert Date index to column
    history['Date'] = pd.to_datetime(history['Date']).dt.strftime('%d %b %Y')       # Convert Timestamp to Datetime

    # Get news and convert timestamp to datetime
    news = company.get_news()
    for article in news:
        article['providerPublishTime'] = pd.to_datetime(article.get('providerPublishTime'), unit='s').strftime('%d %b %Y, %H:%M:%S')

    return render_template(
        'stock.html',
        company_symbol=company.get_symbol(),
        company=company.get_info('longName'),
        table=history.to_html(classes=TABLE_RESPONSIVE_CLASS, justify='left'),
        titles=history.columns.values,
        news=news,
    )


# Forecast button
@app.route('/forecast/<string:symbol>/<string:type>')
def forecast(symbol, type):
    company = CompanyStock(symbol)
    table = company.get_item(type)
    print(table)
    return render_template(
        'forecast.html',
        company=company.get_info('longName'),
        type=type,
        table=table.to_html(classes=TABLE_RESPONSIVE_CLASS, justify='left'),
    )


if __name__ == '__main__':
    app.run()
