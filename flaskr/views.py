#

from flask import Blueprint, request, render_template
from flaskr.models import db, Search, PredictedTable, PredictedRow
from .finance import CompanyStock
from .training import LSTMPrediction
import pandas as pd
import datetime
import time
from collections import Counter
from sqlalchemy import Date
from statistics import mean
import json

views = Blueprint('views', __name__)

TABLE_RESPONSIVE_CLASS = ['table', 'table-striped', 'table-hover', 'table-bordered']


# Home page
@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return show_home()

    # Get search symbol
    search_symbol = request.form.get('search_symbol')
    search = Search(time=time.time(), search_term=search_symbol)

    # Get compare symbol
    compare_symbol = request.form.get('compare_symbol')

    db.session.add(search)
    db.session.commit()
    return stock(search_symbol, compare_symbol)


# Render home page with optional error messages
def show_home(**kwargs):
    all_searches = [x.__dict__ for x in Search.query.all()]
    
    # Count all search terms
    count_searches = dict()
    for search in all_searches:
        count_searches[search['search_term'].upper()] = count_searches.get(search['search_term'].upper(), 0) + 1

    # Get top 3 search terms
    top_searches = Counter(count_searches).most_common(3)

    if kwargs:
        return render_template('home.html', searches=top_searches, **kwargs)

    return render_template('home.html', searches=top_searches)


# Convert and return DataFrame 'Date' column from Timestamp to Datetime
def df_date_to_str(df):
    history = df.get_history().reset_index(level='Date')  # Convert Date index to column
    history['Time'] = history['Date']  # Create Time column
    history['Date'] = pd.to_datetime(history['Date']).dt.strftime('%d %b %Y')  # Convert Timestamp to Datetime
    
    return history


# View stock page
@views.route('/stock/stock=<string:symbol>')
@views.route('/stock/stock=<string:symbol>&compare=<string:compare>')
def stock(symbol, compare=''):
    search_error = 'Error: Stock symbol does not exist'
    same_compare_error = 'Error: Pointless to compare the same stock'
    company = CompanyStock(symbol)
    symbol = symbol.upper()

    if compare != '':
        compare = compare.upper()

        # If symbol and comparison symbol is same (why?)
        if compare == symbol:
            return show_home(compare_error=same_compare_error)      # Error: Stock and comparison is same

        compare_company = CompanyStock(compare)

        # If comparison stock does not exist
        if compare_company.get_symbol() is None:
            # If stock does not exist
            if company.get_symbol() is None:
                return show_home(search_error=search_error, compare_error=search_error)
            return show_home(compare_error=search_error)

        compare_history = df_date_to_str(compare_company)

    # If company stock symbol does not exist
    if company.get_symbol() is None:
        return show_home(search_error=search_error)

    history = df_date_to_str(company)

    # Get news and convert timestamp to datetime
    news = company.get_news()
    for article in news:
        article['providerPublishTime'] = pd.to_datetime(article.get('providerPublishTime'), unit='s').strftime('%d %b %Y, %H:%M:%S')

    # Return comparison data
    if compare != '':
        return render_template(
            'stock.html',
            company_symbol=company.get_symbol(),
            company=company.get_info('longName'),
            table=history.loc[:, history.columns != 'Time'].to_html(classes=TABLE_RESPONSIVE_CLASS, justify='left'),        # Exclude 'Time' column
            news=news,
            data=history.to_json(),
            compare_company_symbol=compare_company.get_symbol(),
            compare_company=compare_company.get_info('longName'),
            compare_data=compare_history.to_json(),
        )

    # Return without comparison data
    return render_template(
        'stock.html',
        company_symbol=company.get_symbol(),
        company=company.get_info('longName'),
        table=history.loc[:, history.columns != 'Time'].to_html(classes=TABLE_RESPONSIVE_CLASS, justify='left'),            # Exclude 'Time' column
        news=news,
        data=history.to_json(),
    )


# View accuracy page
@views.route('/accuracy/<string:symbol>/<stock_type>')
def accuracy(symbol, stock_type):
    company = CompanyStock(symbol)
    data = company.get_item(stock_type)
    ts = int(data['Date'].iloc[-1].timestamp())
    session = db.session()

    # Join (PredictedRow, PredictedTable), where symbol and time is earlier than today
    predicted_history = session.query(
        PredictedRow
    ).join(
        PredictedTable
    ).filter(
        PredictedTable.symbol == symbol
    ).filter(
        PredictedRow.time <= ts         # Filter future dates
    ).all()

    results_dict = dict()
    diff_dict = dict()

    # Get all rows and convert time to Datetime and add to list in results_dict
    for row in predicted_history:
        predicted_date = datetime.datetime.fromtimestamp(row.time).strftime('%Y-%m-%d')
        results_dict.setdefault(predicted_date, []).append(row.value)

    # Get all (key, value) pair in results_dict
    for key, value in results_dict.items():
        stock_value = data.loc[data['Date'] == key, stock_type].tolist()    # Get all dates from data that are the same date

        # If date exists
        if stock_value:
            results_dict[key] = mean(value)         # Get average of each day in results_dict
            difference = stock_value[0] - results_dict[key]
            # Create dict of date, actual stock value, predicted stock value, and difference
            diff_dict[len(diff_dict)] = {
                'Date': datetime.datetime.strptime(key, '%Y-%m-%d'),
                'Actual': stock_value[0],
                'Predicted': results_dict[key],
                'Difference': difference,
                '% difference': (difference/stock_value[0])*100,
            }

    return render_template(
        'accuracy.html',
        company_symbol=symbol,
        company=company.get_info('longName'),
        stock_type=stock_type,
        accuracy=diff_dict,
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
    # folder_name = 'flaskr/static/images/'
    # graph_filename = f'{str(time_now)}_{symbol}_{stock_type}.png'           # Save time, symbol, and type
    predicted_data = prediction.start(days=days, save=True, save_path='models/', save_name=symbol)      # Start prediction and save figure
    predicted_data = [x[0] for x in predicted_data]

    # Data conversion
    last_date = data['Date'].iloc[-1]
    predicted_dates = [(last_date + datetime.timedelta(days=x+1)) for x in range(days)]
    predicted = pd.DataFrame(list(zip(predicted_dates, predicted_data)), columns=['Date', stock_type])
    combined_data = pd.concat([data, predicted], ignore_index=True)                         # Combine data
    combined_data['Date'] = pd.to_datetime(combined_data['Date']).dt.strftime('%d %b %Y')   # Convert Timestamp to Datetime

    # Store data
    predicted_table = PredictedTable(time=int(time.time()), symbol=symbol, stock_type=stock_type)
    db.session.add(predicted_table)
    db.session.flush()

    predicted_rows = [PredictedRow(time=int(pd.to_datetime(row['Date']).timestamp()), value=row[stock_type], table_id=predicted_table.row_id) for _, row in predicted.iterrows()]
    db.session.add_all(predicted_rows)
    db.session.commit()

    return render_template(
        'forecast.html',
        company_symbol=symbol,
        company=company.get_info('longName'),
        stock_type=stock_type,
        table=combined_data.to_html(classes=TABLE_RESPONSIVE_CLASS, justify='left'),
        # graph_filename='/images/' + graph_filename,
        data=data.to_json(),
        predicted=predicted.to_json()
    )
