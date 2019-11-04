import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import functools
import random


data_path = Path('/home/acanacar/Desktop/data/')

bar_path = str(data_path / 'bar/')
tickall_path = str(data_path / 'tickall.pkl')

hist_store = str(
    '/home/acanacar/Desktop/projects/pycharm/VaR/sources/hist_data.h5')


# hist_store = str(
#     r'C:\Users\a.acar\PycharmProjects\Advanced-Deep-Trading-master\analysistemporary\VaR\hist_data.h5')


def Tickers():
    tickers = [
        'EURUSD=X', 'EURTRY=X', 'TRY=X',
        'XU100.IS', 'XU030.IS',
        'AKBNK.IS',
        'ARCLK.IS',
        'ASELS.IS',
        'BIMAS.IS',
        'DOHOL.IS',
        'EKGYO.IS',
        'EREGL.IS',
        'FROTO.IS',
        'GARAN.IS',
        'HALKB.IS',
        'ISCTR.IS',
        'KCHOL.IS',
        'KOZAA.IS',
        'KOZAL.IS',
        'KRDMD.IS',
        'PETKM.IS',
        'PGSUS.IS',
        'SAHOL.IS',
        'SISE.IS',
        'SODA.IS',
        'TAVHL.IS',
        'TCELL.IS',
        'THYAO.IS',
        'TKFEN.IS',
        'TOASO.IS',
        'TTKOM.IS',
        'TUPRS.IS',
        'VAKBN.IS',
        'YKBNK.IS']
    return tickers
