import pandas as pd
import numpy as np
from scipy.stats import norm
import math


class ValueAtRisk(object):
    def __init__(self, interval, matrix, weights, return_method='log', lookbackWindow=252):
        if (interval > 0 and interval < 1):
            self.ci = interval
        else:
            raise Exception("Invalid confidence interval", interval)

        if isinstance(matrix, pd.DataFrame):
            self.input_index = matrix.index
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
        if lookbackWindow > len(self.returnMatrix) + 1:
            raise Exception("invalid Window, cannot excess", len(self.returnMatrix))

        self.portfolioReturn = np.dot(self.returnMatrix[-lookbackWindow:], self.weights)

    def setHistoricalPortfolioReturns(self):
        self.HistoricalPortfolioReturns = np.dot(self.returnMatrix, self.weights)

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
            if sma is False:
                self.variance = np.dot(np.dot(self.weights, np.cov(self.returnMatrix[-lookbackWindow:].T)),
                                       self.weights.T)

            if sma is True:
                self.calculateScaledWeights(lookbackWindow, lambda_decay)
                # self.portfolioReturn = np.dot(self.returnMatrix[-lookbackWindow:], self.weights)
                self.variance = np.dot(self.portfolioReturn, self.scaledweights)
        return self.variance

    def vaR(self, marketValue=0, Approximation=False, sma=False, timescaler=1, lookbackWindow=252,
            lambda_decay=.9):

        if self.returnMatrix.shape[1] != len(self.weights):
            raise Exception("The weights and portfolio doesn't match")
        self.calculateVariance(Approximation, lookbackWindow, sma)

        if marketValue <= 0:
            return abs(norm.ppf(self.ci) * np.sqrt(self.variance)) * math.sqrt(timescaler)
        else:
            return abs(
                norm.ppf(self.ci) * np.sqrt(self.variance)) * marketValue * math.sqrt(
                timescaler)


class HistoricalVaR(ValueAtRisk):
    def __init__(self, interval, matrix, weights, return_method='log', lookbackWindow=252, hybrid=False):
        self.hybrid = hybrid
        super(HistoricalVaR, self).__init__(interval, matrix, weights, return_method='log',
                                            lookbackWindow=252, )

    def getAgeWeightedVar(self, data):
        dx_data = {'portfolio_return': data, 'scaled_weights': self.scaledweights}
        dx = pd.DataFrame(data=dx_data)
        dx = dx.sort_values(by=['portfolio_return'])
        dx['cum_scaled_weights'] = dx['scaled_weights'].cumsum()
        PercentageVaR = dx[dx['cum_scaled_weights'] > (1 - self.ci)].iloc[0].portfolio_return
        return abs(PercentageVaR)

    def setVaRseries(self, lookbackWindow):
        if self.hybrid is False:
            ValueAtRisk = pd.Series(index=self.input_index, name='var_series')
            for i in range(0, len(self.returnMatrix) - lookbackWindow):
                if i == 0:
                    current_portfolio_window = self.HistoricalPortfolioReturns[-lookbackWindow:]
                else:
                    current_portfolio_window = self.HistoricalPortfolioReturns[-(lookbackWindow + i):-i]
                ValueAtRisk[-i - 1] = abs(
                    np.percentile(current_portfolio_window, 100 * (1 - self.ci), interpolation='nearest'))
            self.VaR_Series = ValueAtRisk
        if self.hybrid is True:
            ValueAtRisk = pd.Series(index=self.input_index, name='var_series')
            for i in range(0, len(self.returnMatrix) - lookbackWindow):
                if i == 0:
                    current_portfolio_window = self.HistoricalPortfolioReturns[-lookbackWindow:]
                else:
                    current_portfolio_window = self.HistoricalPortfolioReturns[-(lookbackWindow + i):-i]
                PercentageVaR = self.getAgeWeightedVar(data=current_portfolio_window)

                ValueAtRisk[-i - 1] = PercentageVaR
            self.VaR_Series = ValueAtRisk

    def vaR(self, marketValue=0, lookbackWindow=252, lambda_decay=.9, series=False):
        if self.hybrid is True:
            if series is False:
                self.calculateScaledWeights(lookbackWindow, lambda_decay)
                PercentageVaR = self.getAgeWeightedVar(data=self.portfolioReturn)

            if series is True:
                self.calculateScaledWeights(lookbackWindow, lambda_decay)
                self.setHistoricalPortfolioReturns()
                self.setVaRseries(lookbackWindow=lookbackWindow)
                if marketValue <= 0:
                    return self.VaR_Series
                else:
                    return self.VaR_Series * marketValue
        if self.hybrid is False:
            if series is False:
                PercentageVaR = abs(
                    np.percentile(self.portfolioReturn, 100 * (1 - self.ci), interpolation='nearest'))
            if series is True:
                self.setHistoricalPortfolioReturns()
                self.setVaRseries(lookbackWindow)
                if marketValue <= 0:
                    return self.VaR_Series
                else:
                    return self.VaR_Series * marketValue
        if marketValue <= 0:
            return PercentageVaR
        else:
            return PercentageVaR * marketValue


class MonteCarloVaR(ValueAtRisk):
    def setPortfolioPrices(self):
        self.portfolioPrices = np.dot(self.input, self.weights.T)

    def setCovarianceMatrix(self, lookbackWindow):
        self.covarianceMatrix = np.cov(self.returnMatrix[-lookbackWindow:].T)

    def calculateStdMean(self, lookbackWindow):
        self.setCovarianceMatrix(lookbackWindow)
        self.variance = np.dot(np.dot(self.weights, self.covarianceMatrix), self.weights.T)
        self.std = self.variance ** .5
        self.mean = self.portfolioReturn.mean()

    def simulate(self, predictedDays=252, numSimulations=1000):
        simulation_df = pd.DataFrame()

        last_price = self.portfolioPrices[-1]
        for x in range(numSimulations):

            price_series = []
            price_series.append(last_price)
            count = 0

            # Series for Predicted Days
            for i in range(predictedDays):
                if count == predictedDays:
                    break

                price = price_series[count] * (1 + np.random.normal(self.mean, self.std))

                price_series.append(price)
                count += 1

            simulation_df[x] = price_series
        self.simulation_df = simulation_df
        self.last_price = last_price
        self.simulated_price = np.percentile(simulation_df.iloc[-1, :], 100 * (1 - self.ci))

    def vaR(self, marketValue=0, lookbackWindow=252, predictedDays=1, numSimulations=1000):
        self.setPortfolioPrices()
        self.calculateStdMean(lookbackWindow=lookbackWindow)
        self.simulate(predictedDays, numSimulations)

        return abs((self.simulated_price - self.last_price) / self.last_price)
