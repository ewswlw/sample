import xlwings as xw
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

@xw.func
@xw.arg('stocks', doc='Comma-separated list of stock symbols')
@xw.arg('start_date', doc='Start date in YYYY-MM-DD format')
@xw.arg('end_date', doc='End date in YYYY-MM-DD format')
def get_stock_data(stocks, start_date, end_date):
  """Fetch stock data and return as Excel table"""
  stocks = [s.strip() for s in stocks.split(',')]
  start_date = datetime.strptime(start_date, '%Y-%m-%d')
  end_date = datetime.strptime(end_date, '%Y-%m-%d')

  data = yf.download(stocks, start=start_date, end=end_date)['Close']
  data.reset_index(inplace=True)

  headers = ['Date'] + stocks
  data_list = [headers] + data.values.tolist()

  return data_list

@xw.func
def get_stock_list():
  """Return a list of popular stock symbols"""
  return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']

@xw.sub  # Use @xw.sub for macros
def setup_stock_dashboard():
  wb = xw.Book.caller()
  sheet = wb.sheets.add('Stock Dashboard')

  sheet.range('A1').value = 'Stock Data Dashboard'

  sheet.range('A3').value = 'Stock 1:'
  sheet.range('B3').add_dropdown(get_stock_list())
  sheet.range('A4').value = 'Stock 2:'
  sheet.range('B4').add_dropdown(get_stock_list())
  sheet.range('A5').value = 'Stock 3:'
  sheet.range('B5').add_dropdown(get_stock_list())

  sheet.range('A7').value = 'Start Date:'
  sheet.range('B7').value = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
  sheet.range('A8').value = 'End Date:'
  sheet.range('B8').value = datetime.now().strftime('%Y-%m-%d')

  sheet.range('A10').value = 'Refresh Data'
  sheet.range('A10').api.Interior.Color = 0x00FF00  # Green background

@xw.sub
def refresh_data():
  wb = xw.Book.caller()
  sheet = wb.sheets['Stock Dashboard']
  stocks = ','.join([sheet.range(f'B{i}').value for i in range(3, 6)])
  start_date = sheet.range('B7').value
  end_date = sheet.range('B8').value
  data = get_stock_data(stocks, start_date, end_date)
  sheet.range('A12').value = data
  sheet.range('A12').expand().autofit()

# No need for if __name__ == '__main__': block