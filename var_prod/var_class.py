import pandas as pd
import numpy as np
from scipy.stats import norm
import math


# class ValueAtRiskFactory:
#     @staticmethod
#     def createObjectFromParameters(**kwargs):
#         for k,v in kwargs.items():


class ValueAtRisk(object):
    def __init__(self, interval, matrix, weights, return_method='log', lookbackWindow=252):
        print('val')
        self.lookbackWindow = lookbackWindow
        self.input = matrix
        if interval > 0 and interval < 1:
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

        self.portfolioReturn = np.dot(self.returnMatrix[-self.lookbackWindow:], self.weights)
        self.HistoricalPortfolioReturns = np.dot(self.returnMatrix, self.weights)


    def set_vaR_series(self):
        self.ValueAtRisk = pd.Series(index=self.input_index, name='var_series')

    def calculateScaledWeights(self):
        Range = np.array(range(self.lookbackWindow))
        Range[:] = Range[::-1]
        sma_weights = (1 - self.lambda_decay) * (self.lambda_decay ** Range)
        self.scaledweights = sma_weights / (1 - (self.lambda_decay ** self.lookbackWindow))

    def get_market_value_vaR(self, marketValue):
        print(self.ValueAtRisk * marketValue)
        return self.ValueAtRisk * marketValue


class ParametricVaR(ValueAtRisk):
    # series = True icin bir subclass a ihtiyac var mi.Aslinda her sey ayni fakat output olarak sayi yerine
    # bir seri donuyoruz
    def __init__(self, interval, matrix, weights, return_method, lookbackWindow, timeScaler):
        super().__init__(interval, matrix, weights, return_method, lookbackWindow, timeScaler)
        self.timescaler=timeScaler
    def getCovarianceMatrix(self, current_portfolio_window):
        return np.cov(current_portfolio_window.T)

    def get_variance(self, cov_mat):
        print('c')
        self.variance = np.dot(np.dot(self.weights, cov_mat), self.weights.T)

    def get_vaR_value(self, variance):
        return abs(norm.ppf(self.ci) * np.sqrt(variance)) * math.sqrt(self.timescaler)

    def vaR(self):
        cov_mat = self.getCovarianceMatrix(
            current_portfolio_window=self.returnMatrix[-self.lookbackWindow:])
        self.covariance_matrix = cov_mat
        self.get_variance(cov_mat=cov_mat)
        self.ValueAtRisk = self.get_vaR_value(variance=self.variance)

    def vaRSeries(self):
        self.set_vaR_series()
        for i in range(0, len(self.returnMatrix) - self.lookbackWindow):
            if i == 0:
                current_portfolio_window = self.HistoricalPortfolioReturns[-self.lookbackWindow:]
            else:
                current_portfolio_window = self.HistoricalPortfolioReturns[-(self.lookbackWindow + i):-i]

            cov_mat = self.getCovarianceMatrix(current_portfolio_window=current_portfolio_window)
            current_window_variance = self.get_variance(cov_mat=cov_mat)
            self.ValueAtRisk[-i - 1] = self.get_vaR_value(current_window_variance)


class ParametricVaREwma(ParametricVaR):
    def __init__(self, interval, matrix, weights, return_method, lookbackWindow, timeScaler, lambdaDecay):
        super().__init__(interval, matrix, weights, return_method, lookbackWindow, timeScaler)
        self.lambda_decay = lambdaDecay
        self.timescaler = timeScaler
        self.calculateScaledWeights()

    def get_variance(self, current_return_window):
        return np.dot(current_return_window, self.scaledweights)

    def vaR(self):
        self.variance = self.get_variance(current_return_window=self.returnMatrix[-self.lookbackWindow:])
        self.ValueAtRisk = self.get_vaR_value(variance=self.variance)

    def vaRSeries(self):
        self.set_vaR_series()
        for i in range(0, len(self.returnMatrix) - self.lookbackWindow):
            if i == 0:
                current_portfolio_window = self.HistoricalPortfolioReturns[-self.lookbackWindow:]
            else:
                current_portfolio_window = self.HistoricalPortfolioReturns[-(self.lookbackWindow + i):-i]
            current_window_variance = self.get_variance(current_return_window=current_portfolio_window)
            self.ValueAtRisk[-i - 1] = self.get_vaR_value(variance=current_window_variance)


class HistoricalVaR(ValueAtRisk):
    def __init__(self, interval, matrix, weights, return_method, lookbackWindow):
        print('ccc')
        super().__init__(interval, matrix, weights, return_method, lookbackWindow)

    def get_var_value(self, data):
        return abs(np.percentile(data, 100 * (1 - self.ci), interpolation='nearest'))

    def vaR(self):
        self.ValueAtRisk = self.get_var_value(data=self.portfolioReturn)

    def vaRSeries(self):
        self.set_vaR_series()
        for i in range(0, len(self.returnMatrix) - self.lookbackWindow):
            if i == 0:
                current_portfolio_window = self.HistoricalPortfolioReturns[-self.lookbackWindow:]
            else:
                current_portfolio_window = self.HistoricalPortfolioReturns[-(self.lookbackWindow + i):-i]
            self.ValueAtRisk[-i - 1] = self.get_var_value(data=current_portfolio_window)


class AgeWeightedHistoricalVaR(HistoricalVaR):
    def __init__(self, interval, matrix, weights, return_method, lookbackWindow, lambdaDecay):
        super().__init__(interval, matrix, weights, return_method, lookbackWindow)
        self.calculateScaledWeights()
        self.lambda_decay = lambdaDecay

    def get_var_value(self, data):
        dx_data = {'portfolio_return': data, 'scaled_weights': self.scaledweights}
        dx = pd.DataFrame(data=dx_data)
        dx = dx.sort_values(by=['portfolio_return'])
        dx['cum_scaled_weights'] = dx['scaled_weights'].cumsum()
        PercentageVaR = dx[dx['cum_scaled_weights'] > (1 - self.ci)].iloc[0].portfolio_return
        return abs(PercentageVaR)

    def vaR(self):
        self.ValueAtRisk = self.get_var_value(data=self.portfolioReturn)

    def vaRSeries(self):
        self.set_vaR_series()

        for i in range(0, len(self.returnMatrix) - self.lookbackWindow):
            if i == 0:
                current_portfolio_window = self.HistoricalPortfolioReturns[-self.lookbackWindow:]
            else:
                current_portfolio_window = self.HistoricalPortfolioReturns[-(self.lookbackWindow + i):-i]
            self.ValueAtRisk[-i - 1] = self.get_var_value(data=current_portfolio_window)


class MonteCarloVaR(ValueAtRisk):
    def __init__(self, interval, matrix, weights, return_method, lookbackWindow,
                 numSimulations=1000):
        self.numSimulations = numSimulations

        super(MonteCarloVaR, self).__init__(interval, matrix, weights, return_method='log',
                                            lookbackWindow=252, lambda_decay=.99, timeScaler=1)

    def setPortfolioPrices(self):
        self.portfolioPrices = np.dot(self.input, self.weights.T)

    def setCovarianceMatrix(self, lookbackWindow):
        self.covarianceMatrix = np.cov(self.returnMatrix[-lookbackWindow:].T)

    def calculateStdMean(self, lookbackWindow):
        self.setCovarianceMatrix(lookbackWindow)
        self.variance = np.dot(np.dot(self.weights, self.covarianceMatrix), self.weights.T)
        self.std = self.variance ** .5
        self.mean = self.portfolioReturn.mean()

    def simulate(self, predictedDays=252):
        simulation_df = pd.DataFrame()

        last_price = self.portfolioPrices[-1]
        for x in range(self.numSimulations):

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

    def vaR(self, marketValue=0, lookbackWindow=252, predictedDays=1):
        self.setPortfolioPrices()
        self.calculateStdMean(lookbackWindow=lookbackWindow)
        self.simulate(predictedDays=self.timescaler)

        return abs((self.simulated_price - self.last_price) / self.last_price)
