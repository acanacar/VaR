import pandas as pd
import numpy as np
from scipy.stats import norm
import math


# class ValueAtRiskFactory:
#     @staticmethod
#     def createObjectFromParameters(**kwargs):
#         for k,v in kwargs.items():


class ValueAtRisk(object):
    def __init__(self, interval, matrix, weights, return_method='log', lookbackWindow=252, timeScaler=1,
                 sma=False, lambda_decay=.99):
        self.timescaler = timeScaler
        self.sma = sma
        self.lookbackWindow = lookbackWindow
        self.lambda_decay = lambda_decay
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

    def setHistoricalPortfolioReturns(self):
        self.HistoricalPortfolioReturns = np.dot(self.returnMatrix, self.weights)

    def getCovarianceMatrix(self, current_portfolio_window, lookbackWindow):
        return np.cov(current_portfolio_window.T)

    def covMatrix(self):
        # variables are my securities,single obs of all variables must be each column of matrix
        # thats why we r transposing
        return np.cov(self.returnMatrix.T)

    def calculateScaledWeights(self, lookbackWindow):
        Range = np.array(range(lookbackWindow))
        Range[:] = Range[::-1]
        sma_weights = (1 - self.lambda_decay) * (self.lambda_decay ** Range)
        self.scaledweights = sma_weights / (1 - (self.lambda_decay ** lookbackWindow))

    def getParametricEWMAVaR(self):
        return abs(norm.ppf(self.ci) * np.sqrt(self.variance)) * math.sqrt(self.timescaler)

    def getParametricEWMAVaRSeries(self, para_ewma_variance, para_mean=0):
        return abs(norm.ppf(self.ci) * np.sqrt(para_ewma_variance)) * math.sqrt(self.timescaler)

    def getParametricVaR(self, para_variance, para_mean=0):
        return abs(norm.ppf(self.ci) * np.sqrt(para_variance)) * math.sqrt(self.timescaler)

    def calculateVariance(self, Approximation=False, lookbackWindow=252, lambda_decay=.9,
                          series=False, parametricInput=None):
        if Approximation == True:
            self.variance = np.var(self.portfolioReturn)

        else:
            if self.sma is False:
                if series is False:
                    cov_mat = self.getCovarianceMatrix(
                        current_portfolio_window=parametricInput,
                        lookbackWindow=lookbackWindow)
                    self.variance = np.dot(np.dot(self.weights, cov_mat), self.weights.T)
                if series is True:
                    cov_mat = self.getCovarianceMatrix(
                        current_portfolio_window=parametricInput,
                        lookbackWindow=lookbackWindow)

                    return np.dot(np.dot(self.weights, cov_mat), self.weights.T)

            if self.sma is True:
                if series is False:
                    self.calculateScaledWeights(lookbackWindow)
                    # self.portfolioReturn = np.dot(self.returnMatrix[-lookbackWindow:], self.weights)
                    self.variance = np.dot(self.portfolioReturn, self.scaledweights)

            return self.variance

    def setVarSeries(self, lookbackWindow):
        ValueAtRisk = pd.Series(index=self.input_index, name='var_series')
        if self.sma is True:
            self.calculateScaledWeights(lookbackWindow)
        for i in range(0, len(self.returnMatrix) - lookbackWindow):
            if i == 0:
                current_portfolio_window = self.HistoricalPortfolioReturns[-lookbackWindow:]
            else:
                current_portfolio_window = self.HistoricalPortfolioReturns[-(lookbackWindow + i):-i]
            if self.sma is False:
                para_variance = self.calculateVariance(series=True, parametricInput=current_portfolio_window)
                para_vaR = self.getParametricVaR(para_variance, self.timescaler)
                ValueAtRisk[-i - 1] = para_vaR
            if self.sma is True:
                para_ewma_variance = np.dot(current_portfolio_window, self.scaledweights)
                para_ewma_vaR = self.getParametricEWMAVaRSeries(para_ewma_variance)
                ValueAtRisk[-i - 1] = para_ewma_vaR

        self.VaR_Series = ValueAtRisk

    def vaR(self, marketValue=0, Approximation=False, lookbackWindow=252,
            lambda_decay=.9, series=False):

        if self.returnMatrix.shape[1] != len(self.weights):
            raise Exception("The weights and portfolio doesn't match")
        if series is False:
            self.calculateVariance(Approximation, lookbackWindow,
                                   parametricInput=self.returnMatrix[-lookbackWindow:])
            ValueAtRisk = self.getParametricEWMAVaR()
            if marketValue <= 0:
                return ValueAtRisk
            else:
                return ValueAtRisk * marketValue

        if series is True:
            self.setHistoricalPortfolioReturns()
            self.setVarSeries(lookbackWindow=lookbackWindow)
            if marketValue <= 0:
                return self.VaR_Series
            else:
                return self.VaR_Series * marketValue


class ParametricVaR(ValueAtRisk):
    # series = True icin bir subclass a ihtiyac var mi.Aslinda her sey ayni fakat output olarak sayi yerine
    # bir seri donuyoruz
    def __init__(self, interval, matrix, weights, return_method, lookbackWindow, timeScaler):
        super().__init__(interval, matrix, weights, return_method, lookbackWindow, timeScaler)

    def getCovarianceMatrix(self, current_portfolio_window):
        return np.cov(current_portfolio_window.T)

    def get_variance(self, cov_mat):
        print('c')
        self.variance = np.dot(np.dot(self.weights, cov_mat), self.weights.T)

    def getVaR(self, variance):
        return abs(norm.ppf(self.ci) * np.sqrt(variance)) * math.sqrt(self.timescaler)

    def set_vaR_series(self):
        self.ValueAtRisk = pd.Series(index=self.input_index, name='var_series')

    def vaR(self):
        cov_mat = self.getCovarianceMatrix(
            current_portfolio_window=self.returnMatrix[-self.lookbackWindow:])
        self.covariance_matrix = cov_mat
        self.get_variance(cov_mat=cov_mat)
        self.ValueAtRisk = self.getVaR(variance=self.variance)

    def vaRSeries(self):
        self.set_vaR_series()
        for i in range(0, len(self.returnMatrix) - self.lookbackWindow):
            if i == 0:
                current_portfolio_window = self.HistoricalPortfolioReturns[-self.lookbackWindow:]
            else:
                current_portfolio_window = self.HistoricalPortfolioReturns[-(self.lookbackWindow + i):-i]

            cov_mat = self.getCovarianceMatrix(current_portfolio_window=current_portfolio_window)
            current_window_variance = self.get_variance(cov_mat=cov_mat)
            self.ValueAtRisk[-i - 1] = self.getVaR(current_window_variance)


class ParametricVaREwma(ParametricVaR):
    def __init__(self, interval, matrix, weights, return_method, lookbackWindow, timeScaler, lambdaDecay):
        super().__init__(interval, matrix, weights, return_method, lookbackWindow, timeScaler)
        self.calculateScaledWeights()

    def calculateScaledWeights(self):
        Range = np.array(range(self.lookbackWindow))
        Range[:] = Range[::-1]
        sma_weights = (1 - self.lambda_decay) * (self.lambda_decay ** Range)
        self.scaledweights = sma_weights / (1 - (self.lambda_decay ** self.lookbackWindow))

    def get_variance(self, current_return_window):
        return np.dot(current_return_window, self.scaledweights)

    def vaRSeries(self):
        self.set_vaR_series()
        for i in range(0, len(self.returnMatrix) - self.lookbackWindow):
            if i == 0:
                current_portfolio_window = self.HistoricalPortfolioReturns[-self.lookbackWindow:]
            else:
                current_portfolio_window = self.HistoricalPortfolioReturns[-(self.lookbackWindow + i):-i]
            current_window_variance = self.get_variance(current_return_window=current_portfolio_window)
            self.ValueAtRisk[-i - 1] = self.getVaR(variance=current_window_variance)


class HistoricalVaR(ValueAtRisk):
    def __init__(self, interval, matrix, weights, return_method='log', lookbackWindow=252,
                 hybrid=False, lambda_decay_hist=.99):
        self.hybrid = hybrid
        self.lambda_decay_hist = lambda_decay_hist

        super(HistoricalVaR, self).__init__(interval, matrix, weights, return_method='log',
                                            lookbackWindow=252)

    def getAgeWeightedVar(self, data):
        dx_data = {'portfolio_return': data, 'scaled_weights': self.scaledweights}
        dx = pd.DataFrame(data=dx_data)
        dx = dx.sort_values(by=['portfolio_return'])
        dx['cum_scaled_weights'] = dx['scaled_weights'].cumsum()
        PercentageVaR = dx[dx['cum_scaled_weights'] > (1 - self.ci)].iloc[0].portfolio_return
        return abs(PercentageVaR)

    def calculateScaledWeightsHist(self, lookbackWindow, lambda_decay):
        print(lambda_decay)
        Range = np.array(range(lookbackWindow))
        Range[:] = Range[::-1]
        sma_weights = (1 - lambda_decay) * (lambda_decay ** Range)
        self.scaledweights = sma_weights / (1 - (lambda_decay ** lookbackWindow))

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
                self.calculateScaledWeightsHist(lookbackWindow, self.lambda_decay_hist)
                PercentageVaR = self.getAgeWeightedVar(data=self.portfolioReturn)

            if series is True:
                self.calculateScaledWeightsHist(lookbackWindow, self.lambda_decay_hist)
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
    def __init__(self, interval, matrix, weights, return_method='log', lookbackWindow=252,
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
