import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
from matplotlib import pyplot as plt

import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import tkinter as tk
from tkinter import ttk

import urllib
import json

import pandas as pd
import numpy as np


LARGE_FONT = ("Verdana", 12)
NORM_FONT = ("Verdana", 10)
SMALL_FONT = ("Verdana", 8)

style.use("ggplot")

f = plt.figure()

# a = f.add_subplot(111)

exchange = "BTC-e"
DatCounter = 9000
programName = "btce"
resampleSize = "15Min"
DataPace = "tick"
candleWidth = 0.008

paneCount = 1

topIndicator = "none"
bottomIndicator = "none"
middleIndicator = "none"
chartLoad = True

darkColor = "#183A54"
lightColor = "#00A3E0"

def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command=popup.destroy)
    B1.pack()
    popup.mainloop()

def changeTimeFrame(tf):
    global DataPace
    global DatCounter
    if tf == "7d" and resampleSize == "1Min":
        popupmsg("Too much data chosen, choose a smaller time frame or higher OHLC interval")
    else:
        DataPace = tf
        DatCounter = 9000


def changeExchange(toWhat, pn):
    global exchange
    global DatCounter
    global programName

    exchange = toWhat
    programName = pn
    DatCounter = 9000


class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "Sea of BTC client")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save settings", command=lambda: popupmsg("Not supported just yet!"))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)

        exchangeChoice = tk.Menu(menubar, tearoff=1)
        exchangeChoice.add_command(label="BTC-e",
                                   command=lambda: changeExchange("BTC-e", "btce"))
        exchangeChoice.add_command(label="Bitfinex",
                                   command=lambda: changeExchange("Bitfinex", "bitfinex"))
        exchangeChoice.add_command(label="Bitstamp",
                                   command=lambda: changeExchange("Bitstamp", "bitstamp"))
        exchangeChoice.add_command(label="Huobi",
                                   command=lambda: changeExchange("Huobi", "huobi"))

        menubar.add_cascade(label="Exchange", menu=exchangeChoice)

        dataTF = tk.Menu(menubar, tearoff=1)
        dataTF.add_command(label="Tick",
                           command=lambda: changeTimeFrame('tick'))
        dataTF.add_command(label="1 Day",
                           command=lambda: changeTimeFrame('1d'))
        dataTF.add_command(label="3 Day",
                           command=lambda: changeTimeFrame('3d'))
        dataTF.add_command(label="1 Week",
                           command=lambda: changeTimeFrame('7d'))
        menubar.add_cascade(label="Data Time Frame", menu=dataTF)

        OHLCI = tk.Menu(menubar, tearoff=1)
        OHLCI.add_command(label="Tick",
                          command=lambda: changeTimeFrame('tick'))
        OHLCI.add_command(label="1 minute",
                          command=lambda: changeSampleSize('1Min', 0.0005))
        OHLCI.add_command(label="5 minute",
                          command=lambda: changeSampleSize('5Min', 0.003))
        OHLCI.add_command(label="15 minute",
                          command=lambda: changeSampleSize('15Min', 0.008))
        OHLCI.add_command(label="30 minute",
                          command=lambda: changeSampleSize('30Min', 0.016))
        OHLCI.add_command(label="1 Hour",
                          command=lambda: changeSampleSize('1H', 0.032))
        OHLCI.add_command(label="3 Hour",
                          command=lambda: changeSampleSize('3H', 0.096))

        menubar.add_cascade(label="OHLC Interval", menu=OHLCI)

        topIndi = tk.Menu(menubar, tearoff=1)
        topIndi.add_command(label="None",
                            command=lambda: addTopIndicator('none'))
        topIndi.add_command(label="RSI",
                            command=lambda: addTopIndicator('rsi'))
        topIndi.add_command(label="MACD",
                            command=lambda: addTopIndicator('macd'))

        menubar.add_cascade(label="Top Indicator", menu=topIndi)

        mainI = tk.Menu(menubar, tearoff=1)
        mainI.add_command(label="None",
                          command=lambda: addMiddleIndicator('none'))
        mainI.add_command(label="SMA",
                          command=lambda: addMiddleIndicator('sma'))
        mainI.add_command(label="EMA",
                          command=lambda: addMiddleIndicator('ema'))

        menubar.add_cascade(label="Main/middle Indicator", menu=mainI)

        bottomI = tk.Menu(menubar, tearoff=1)
        bottomI.add_command(label="None",
                            command=lambda: addBottomIndicator('none'))
        bottomI.add_command(label="RSI",
                            command=lambda: addBottomIndicator('rsi'))
        bottomI.add_command(label="MACD",
                            command=lambda: addBottomIndicator('macd'))

        menubar.add_cascade(label="Bottom Indicator", menu=bottomI)

        tradeButton = tk.Menu(menubar, tearoff=1)
        tradeButton.add_command(label="Manual Trading",
                                command=lambda: popupmsg("This is not live yet"))
        tradeButton.add_command(label="Automated Trading",
                                command=lambda: popupmsg("This is not live yet"))

        tradeButton.add_separator()
        tradeButton.add_command(label="Quick Buy",
                                command=lambda: popupmsg("This is not live yet"))
        tradeButton.add_command(label="Quick Sell",
                                command=lambda: popupmsg("This is not live yet"))

        tradeButton.add_separator()
        tradeButton.add_command(label="Set-up Quick Buy/Sell",
                                command=lambda: popupmsg("This is not live yet"))

        menubar.add_cascade(label="Trading", menu=tradeButton)

        startStop = tk.Menu(menubar, tearoff=1)
        startStop.add_command(label="Resume",
                              command=lambda: loadChart('start'))
        startStop.add_command(label="Pause",
                              command=lambda: loadChart('stop'))
        menubar.add_cascade(label="Resume/Pause client", menu=startStop)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Tutorial", command=tutorial)

        menubar.add_cascade(label="Help", menu=helpmenu)

        tk.Tk.config(self, menu=menubar)

        self.frames = {}

        for F in (StartPage, BTCe_Page):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

        # tk.Tk.iconbitmap(self, default="clienticon.ico")

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()