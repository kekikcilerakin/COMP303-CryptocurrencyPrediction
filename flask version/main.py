from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import SelectField, DateField

app = Flask(__name__)
app.config['SECRET_KEY'] = 'AKN'

import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

predictionDays = 60
scaler = MinMaxScaler(feature_range=(0, 1))
model = Sequential()

def trainData(selectedCryptocurrency, date):
    import secrets
    startDate = dt.datetime(2015, 1, 1)
    endDate = dt.datetime.now()

    data = yf.download(f'{selectedCryptocurrency}-USD', startDate, endDate)

    #preparing the data
    
    scaledData = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    x_train = []
    y_train = []

    for x in range(predictionDays, len(scaledData)):
        x_train.append(scaledData[x - predictionDays:x, 0])
        y_train.append(scaledData[x, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    #creating the neural network
    model.add(LSTM(units = 50, name = secrets.token_hex(5), return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, name = secrets.token_hex(5), return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, name = secrets.token_hex(5)))
    model.add(Dropout(0.2)) #prevents overlap
    model.add(Dense(units = 1)) #outputs predicted value
    model.compile(optimizer = "adam", loss = "mean_squared_error")
    model.fit(x_train, y_train, epochs = 1, batch_size = 32)

    testModel(selectedCryptocurrency, date, data)

import pandas as pd
from datetime import datetime
def testModel(selectedCryptocurrency, date, data):
    date.strftime("%d/%m/%y")
    year = date.year
    month = date.month
    day = date.day

    testStartDate = dt.datetime(int(year), int(month), int(day))
    testEndDate = dt.datetime.now()
    testData = yf.download(f'{selectedCryptocurrency}-USD', testStartDate, testEndDate)

    actualPrices = testData['Close'].values

    concatedDataset = pd.concat((data['Close'], testData['Close']), axis = 0)

    modelInputs = concatedDataset[len(concatedDataset) - len(testData) - predictionDays:].values
    modelInputs = modelInputs.reshape(-1, 1)
    modelInputs = scaler.fit_transform(modelInputs)

    x_test = []

    for x in range(predictionDays, len(modelInputs)):
        x_test.append(modelInputs[x - predictionDays:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictionPrices = model.predict(x_test)
    predictionPrices = scaler.inverse_transform(predictionPrices)
    showPlot(selectedCryptocurrency, testData, actualPrices, predictionPrices)

@app.route('/', methods = ['GET', 'POST'])
def form():
    form = PredictionForm()

    if form.validate_on_submit():
        trainData(form.cryptocurrency.data, form.date.data)
    return render_template('index.html', form = form)

#GET CRYPTOCURRENCY NAMES FOR DROPDOWN START
import requests
from bs4 import BeautifulSoup

def getCryptocurrencyNames():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
    url = 'https://finance.yahoo.com/crypto/'

    response = requests.get(url, headers = headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    rows = soup.find('table').find('tbody').find_all('tr')

    cryptocurrList = []

    for row in rows:
        cryptocurrList.append(row.find_all('td')[0].text.replace('-USD', ''))

    return cryptocurrList
#GET CRYPTOCURRENCY NAMES FOR DROPDOWN END

class PredictionForm(FlaskForm):
    cryptocurrency = SelectField('cryptocurrency', choices = getCryptocurrencyNames())
    date = DateField('date')

from matplotlib.figure import Figure
def showPlot(selectedCryptocurrency, testData, actualPrices, predictionPrices):
    fig = Figure(figsize = (5, 5),dpi = 100)
    fig.subplots_adjust(bottom=0.2, left=0.2) 
    plot1 = fig.add_subplot(111)
    plot1.plot(testData.index, actualPrices, color = "black", label = "Actual Prices")
    plot1.plot(testData.index, predictionPrices, color = "green", label = "Predicted Prices")
    plot1.set_title(f'{selectedCryptocurrency} to USD Price Prediction')
    plot1.legend(loc = "upper left")
    plot1.set_xlabel("Time")
    plot1.set_ylabel("Price")

    for tick in plot1.get_xticklabels():
        tick.set_rotation(45)
    
    fig.savefig('static/plot.png')

if __name__ == "__main__":
    app.run()