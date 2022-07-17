from flask import Blueprint, request, render_template
from flaskr.models import db, Search, PredictedTable, PredictedRow
from .finance import CompanyStock
from .training import LSTMPrediction
import pandas as pd
import datetime
import time
from collections import Counter

views = Blueprint('views', __name__)

TABLE_RESPONSIVE_CLASS = ['table', 'table-striped', 'table-hover', 'table-bordered']


# Home page
@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        all_searches = [x.__dict__ for x in Search.query.all()]
        # for search in top_searches:
        #     search['time'] = datetime.datetime.fromtimestamp(search['time']).strftime('%d %b %Y %H:%M:%S')

        # Count all search terms
        count_searches = dict()
        for search in all_searches:
            count_searches[search['search_term'].upper()] = count_searches.get(search['search_term'].upper(), 0) + 1

        # Get top 3 search terms
        top_searches = Counter(count_searches).most_common(3)

        return render_template('home.html', searches=top_searches)
    search_symbol = request.form.get('search_symbol')
    search = Search(time=time.time(), search_term=search_symbol)
    db.session.add(search)
    db.session.commit()
    return stock(search_symbol)


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
        news=news,
        data=history.to_json(),
    )


# View accuracy page
@views.route('/accuracy/<string:symbol>/<stock_type>')
def accuracy(symbol, stock_type):
    company = CompanyStock(symbol)
    data = company.get_item(stock_type)
    return render_template(
        'accuracy.html',
        company=company.get_info('longName'),
        stock_type=stock_type,
    )


# Forecast button
@views.route('/forecast/<string:symbol>/<stock_type>')
def forecast(symbol, stock_type, days=None):
    if days is None:
        days = 30

    time_now = time.time()
    company = CompanyStock(symbol)
    data = company.get_item(stock_type)
    prediction = LSTMPrediction(data)

    # Start prediction
    folder_name = 'flaskr/static/images/'
    graph_filename = f'{str(time_now)}_{symbol}_{stock_type}.png'                           # Save time, symbol, and type
    # Start prediction and save figure
    predicted_data = prediction.start(
        days=days,
        fig_dir=folder_name,
        fig_name=graph_filename,
        save=True,
        save_path='models/',
        save_name=symbol,
    )
    predicted_data = [x[0] for x in predicted_data]

    # Data conversion
    last_date = data['Date'].iloc[-1]
    predicted_dates = [(last_date + datetime.timedelta(days=x+1)) for x in range(days)]
    predicted = pd.DataFrame(list(zip(predicted_dates, predicted_data)), columns=['Date', stock_type])
    combined_data = pd.concat([data, predicted], ignore_index=True)                         # Combine data
    combined_data['Date'] = pd.to_datetime(combined_data['Date']).dt.strftime('%d %b %Y')   # Convert Timestamp to Datetime

    # Store data
    # Create PredictedTable
    predicted_table = PredictedTable(
        time=int(time.time()),
        symbol=symbol,
        stock_type=stock_type
    )
    db.session.add(predicted_table)
    db.session.flush()              # Flush to get new table id

    # Create PredictedRow objects in a list
    predicted_rows = [
        PredictedRow(
            time=int(pd.to_datetime(row['Date']).timestamp()),
            value=row[stock_type],
            table_id=predicted_table.id)
        for _, row in predicted.iterrows()]
    db.session.add_all(predicted_rows)
    db.session.commit()

    return render_template(
        'forecast.html',
        company=company.get_info('longName'),
        stock_type=stock_type,
        table=combined_data.to_html(classes=TABLE_RESPONSIVE_CLASS, justify='left'),
        graph_filename='/images/' + graph_filename,
    )
