import pandas as pd
import numpy as np
from scipy.stats import norm
import math


class ValueAtRisk(object):
    def __init__(self, interval, matrix, weights, return_method='log'):
        if (interval > 0 and interval < 1):
            self.ci = interval
        else:
            raise Exception("Invalid confidence interval", interval)

        if isinstance(matrix, pd.DataFrame):
            matrix = matrix.values

        if matrix.ndim != 2:
            raise Exception("Only accept 2 dimensions matrix", matrix.ndim)

        if len(weights) != matrix.shape[1]:
            raise Exception("Weights Length doesn't match")

        self.input = matrix

        if return_method == 'log':
            self.returnMatrix = np.diff(np.log(self.input), axis=0)
        elif return_method == 'pct':
            self.returnMatrix = np.diff(self.input, axis=0) / self.input[:-1, ]
        else:
            raise Exception("Unvalid return method")
        if not isinstance(weights, np.ndarray):
            self.weights = np.array(weights)
        else:
            self.weights = weights

    def covMatrix(self):
        # variables are my securities,single obs of all variables must be each column of matrix
        # thats why we r transposing
        return np.cov(self.returnMatrix.T)

    def calculateScaledWeights(self, lookbackWindow, lambda_decay):
        Range = np.array(range(lookbackWindow))
        Range[:] = Range[::-1]
        sma_weights = (1 - lambda_decay) * (lambda_decay ** Range)
        self.scaledweights = sma_weights / (1 - (lambda_decay ** lookbackWindow))

    def calculateVariance(self, Approximation=False, lookbackWindow=252, sma=False, lambda_decay=.9):
        if Approximation == True:
            self.variance = np.var(np.dot(self.returnMatrix[-lookbackWindow:], self.weights))
        else:
            if sma == False:
                self.variance = np.dot(np.dot(self.weights, np.cov(self.returnMatrix[-lookbackWindow:].T)),
                                       self.weights.T)

            if sma == True:
                self.calculateScaledWeights(lookbackWindow, lambda_decay)
                self.portfolio_return = np.dot(self.returnMatrix, self.weights)
                self.variance = np.dot(self.portfolio_return[-lookbackWindow:], self.scaledweights)
        return self.variance

    def vaR(self, marketValue=0, Approximation=False, sma=False, timescaler=1, lookbackWindow=252,
            lambda_decay=.9):

        if (self.returnMatrix.shape[1] != len(self.weights)):
            raise Exception("The weights and portfolio doesn't match")
        self.calculateVariance(Approximation, lookbackWindow, sma)

        if marketValue <= 0:
            return abs(norm.ppf(self.ci) * np.sqrt(self.variance)) * math.sqrt(timescaler)
        else:
            return abs(
                norm.ppf(self.ci) * np.sqrt(self.variance)) * marketValue * math.sqrt(
                timescaler)


class HistoricalVaR(ValueAtRisk):
    def vaR(self, marketValue=0, window=0):
        self.portfolioReturn = np.dot(self.returnMatrix, self.weights)

        if window > len(self.portfolioReturn) + 1:
            raise Exception("invalid Window, cannot excess", len(self.portfolioReturn))

        if window > 0 and window < len(self.portfolioReturn):
            PercentageVaR = abs(
                np.percentile(self.portfolioReturn[-window:], 100 * (1 - self.ci), interpolation='nearest'))
        else:
            PercentageVaR = abs(
                np.percentile(self.portfolioReturn, 100 * (1 - self.ci), interpolation='nearest'))

        if marketValue <= 0:
            return PercentageVaR
        else:
            return PercentageVaR * marketValue
