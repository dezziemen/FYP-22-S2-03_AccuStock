from flask import Flask, render_template, request
from finance import CompanyStock
from training import LSTMPrediction
import pandas as pd

app = Flask(__name__)


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
        company=company.get_info('longName'),
        table=history.to_html(classes=['table', 'table-striped', 'table-hover', 'table-bordered'], justify='left'),
        titles=history.columns.values,
        news=news,
    )


if __name__ == '__main__':
    app.run()
