import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from flaskr.finance import CompanyStock
from flaskr.training import LSTMPrediction
import pandas as pd
import time
import io
from flaskr import create_app

# app = Flask(__name__)
app = create_app()

# TABLE_RESPONSIVE_CLASS = ['table', 'table-striped', 'table-hover', 'table-bordered']


# # Home page
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'GET':
#         return render_template('home.html')
#     return stock(request.form.get('searchSymbol'))

#
# # View stock page
# @app.route('/stock/<string:symbol>')
# def stock(symbol):
#     company = CompanyStock(symbol)
#
#     # If company stock symbol does not exist
#     if company.get_symbol() is None:
#         return render_template('home.html', searchError='Error: Stock symbol does not exist.')
#
#     history = company.get_history().reset_index(level='Date')                       # Convert Date index to column
#     history['Date'] = pd.to_datetime(history['Date']).dt.strftime('%d %b %Y')       # Convert Timestamp to Datetime
#
#     # Get news and convert timestamp to datetime
#     news = company.get_news()
#     for article in news:
#         article['providerPublishTime'] = pd.to_datetime(article.get('providerPublishTime'), unit='s').strftime('%d %b %Y, %H:%M:%S')
#
#     return render_template(
#         'stock.html',
#         company_symbol=company.get_symbol(),
#         company=company.get_info('longName'),
#         table=history.to_html(classes=TABLE_RESPONSIVE_CLASS, justify='left'),
#         titles=history.columns.values,
#         news=news,
#         data=history.to_json()
#     )
#
#
# # Forecast button
# @app.route('/forecast/<string:symbol>/<string:type>')
# def forecast(symbol, type):
#     time_now = time.time()
#     company = CompanyStock(symbol)
#     data = company.get_item(type)
#     prediction = LSTMPrediction(data)
#
#     # Prepare and start prediction
#     look_back, x_train, x_test, y_train, y_test, test_data = prediction.reshape()
#     model = prediction.prepare_model(look_back, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
#     prediction.train(model, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, test_data=test_data)
#     prediction.predict(days=30, model=model, test_data=test_data)
#
#     graph_filename = f'static/{str(time_now)}.png'
#     plt.savefig(graph_filename)
#     return render_template(
#         'forecast.html',
#         company=company.get_info('longName'),
#         type=type,
#         table=data.to_html(classes=TABLE_RESPONSIVE_CLASS, justify='left'),
#         graph_filename='/' + graph_filename,
#     )


if __name__ == '__main__':
    app.run()
