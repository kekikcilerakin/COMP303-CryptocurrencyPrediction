import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def button_event():
    for widget in canvasFrame.winfo_children():
        widget.destroy()

    

    #testing the model
    
    matplotlib.use('TkAgg')

    
    # the figure that will contain the plot
    fig = Figure(figsize = (5, 5),dpi = 100)
    fig.subplots_adjust(bottom=0.2, left=0.2) 
    plot1 = fig.add_subplot(111)
    plot1.plot(testData.index, actualPrices, color = "black", label = "Actual Prices")
    plot1.plot(testData.index, predictionPrices, color = "green", label = "Predicted Prices")
    plot1.set_title(f'{cryptoSelectDropdown.get()} to USD Price Prediction')
    plot1.legend(loc = "upper left")
    plot1.set_xlabel("Time")
    plot1.set_ylabel("Price")
    
    #plot1.set_xticks(plot1.get_xticks(), rotation = 45)
    for tick in plot1.get_xticklabels():
        tick.set_rotation(45)
    canvas = FigureCanvasTkAgg(fig,master = canvasFrame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=25)
    # plt.plot(testData.index, actualPrices, color = "black", label = "Actual Prices")
    # plt.plot(testData.index, predictionPrices, color = "green", label = "Predicted Prices")
    # plt.title(f'{cryptoSelectDropdown.get()} to {againstSelectDropdown.get()} Price Prediction')
    # plt.xticks(rotation = 45)
    # plt.xlabel("Time")
    # plt.ylabel("Price")
    # plt.legend(loc = "upper left")
    # plt.show()

    # predict next day
    realData = [modelInputs[len(modelInputs) + 1 - predictionDays:len(modelInputs) + 1, 0]]
    realData = np.array(realData)
    realData = np.reshape(realData, (realData.shape[0], realData.shape[1], 1))

    prediction = model.predict(realData)

    prediction = scaler.inverse_transform(prediction)
    print(prediction)

#GUI START
import tkinter
import customtkinter as ctk
from tkcalendar import DateEntry

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.resizable(False, False)

frame = ctk.CTkFrame(root)
frame.pack(pady = 10, padx = 10, fill = "both", expand = False)

canvasFrame = ctk.CTkFrame(root)
canvasFrame.pack(pady = 10, padx = 10, fill = "both", expand = False)

title = ctk.CTkLabel(frame, text = "Select Cryptocurrency to predict")
title.pack()

cryptoSelectDropdown = ctk.CTkOptionMenu(frame, values=["BTC", "ETC", "ETH", "USDT", "USDC", "BNB", "BUSD", "XRP", "DOGE", "ADA", "MATIC", "DAI", "WTRX", "DOT", "TRX", "SHIB", "LTC", "SOL", "HEX", "STETH", "UNI7083", "AVAX", "LEO", "WBTC", "LINK", "TON11419"])
cryptoSelectDropdown.set("BTC")
cryptoSelectDropdown.pack(padx = 20, pady = 10)

testMinDate = dt.datetime(2015, 1, 1)
testMaxDate = dt.datetime(2022, 1, 1)
testDateStart = DateEntry(frame, mindate = testMinDate, maxdate = testMaxDate)
testDateStart.set_date(dt.datetime(2015, 1, 1))
testDateStart.pack(padx = 20, pady = 10)

button = ctk.CTkButton(master=frame, text="Tahmini ve Grafiği Gör", command=button_event)
button.pack(padx = 20, pady = 10)

root.mainloop()

#GUI END