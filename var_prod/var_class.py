# individual bar eklenebilir
# risk free rate kullanarak sharpe ratio hesabi
# C:\Users\a.acar\PycharmProjects\VaR\sources\ExcelModelingofPortfolioVariance.xls
import pandas as pd
import numpy as np
from scipy.stats import norm
import math


class ValueAtRisk(object):
    def __init__(self, matrix, interval, weights, return_method, lookbackWindow, marketValue=1000):
        self.marketValue = marketValue
        self.lookbackWindow = lookbackWindow
        if not isinstance(weights, np.ndarray):
            self.weights = np.array(weights)
        else:
            self.weights = weights
        if interval > 0 and interval < 1:
            self.ci = interval
        else:
            raise Exception("Invalid confidence interval", interval)
        if matrix is not None:
            if matrix.ndim != 2:
                raise Exception("Only accept 2 dimensions matrix", matrix.ndim)
            if isinstance(matrix, pd.DataFrame):
                self.input_index = matrix.index
                matrix = matrix.values
            if len(weights) != matrix.shape[1]:
                raise Exception("Weights Length doesn't match")
        self.input = matrix

        if return_method == 'log':
            self.returnMatrix = np.diff(np.log(self.input), axis=0)
        elif return_method == 'pct':
            self.returnMatrix = np.diff(self.input, axis=0) / self.input[:-1, ]
        else:
            raise Exception("Unvalid return method")
        if self.lookbackWindow > len(self.returnMatrix) + 1:
            raise Exception("invalid Window, cannot excess", len(self.returnMatrix))

        self.portfolioReturn = np.dot(self.returnMatrix[-self.lookbackWindow:], self.weights)
        self.HistoricalPortfolioReturns = np.dot(self.returnMatrix, self.weights)

    def get_vaR_sigma(self):
        percentile_point = norm.ppf(self.ci)
        time_horizon_factor = np.sqrt(self.timescaler)
        return -percentile_point * self.sigma * time_horizon_factor

    def set_vaR_series(self):
        self.ValueAtRisk = pd.Series(index=self.input_index, name='VaR_series')
        self.MarginalVaR = pd.Series(index=self.input_index, name='MarginalVaR_series')

    def calculateScaledWeights(self):
        Range = np.array(range(self.lookbackWindow))
        Range[:] = Range[::-1]
        sma_weights = (1 - self.lambda_decay) * (self.lambda_decay ** Range)
        self.scaledweights = sma_weights / (1 - (self.lambda_decay ** self.lookbackWindow))

    def get_market_value_vaR(self, marketValue):
        print(self.ValueAtRisk * marketValue)
        return self.ValueAtRisk * marketValue

    def createBacktestDf(self):
        returns_ = np.append(np.nan, self.HistoricalPortfolioReturns)
        backtest_df = pd.DataFrame(index=self.input_index,
                                   data={'VaR': self.ValueAtRisk.values,
                                         'PortfolioReturn': returns_})
        # actual_return_col = 'actual_{timescale}_day_return'.format(
        #     timescale=self.timeScaler if hasattr(self, 'timeScaler') else "1")
        # if self.timeScaler:
        #     backtest_df[actual_return_col] = backtest_df['PortfolioReturn'].shift(-self.timeScaler)
        # else:
        #     backtest_df[actual_return_col] = backtest_df['PortfolioReturn'].shift(-1)
        if hasattr(self, 'timeScaler'):
            backtest_df['realized_return'] = backtest_df['PortfolioReturn'].shift(-self.timeScaler)
        else:
            backtest_df['realized_return'] = backtest_df['PortfolioReturn'].shift(-1)

        backtest_df['VaR_fail_flag'] = backtest_df.apply(
            lambda row_: 1 if row_['realized_return'] < -1 * row_['VaR'] else 0, axis=1)
        self.backtestDf = backtest_df


class ParametricVaR(ValueAtRisk):
    def __init__(self, matrix, interval, weights, return_method, lookbackWindow, timeScaler):
        super().__init__(matrix, interval, weights, return_method, lookbackWindow)
        self.timescaler = timeScaler

    def getBeta(self):
        marketvalue = 1000
        asset_values = np.dot(self.weights, marketvalue)
        cov_mat = self.getCovarianceMatrix(self.returnMatrix[-self.lookbackWindow:])
        pf_variance = np.dot(asset_values.T, np.dot(cov_mat, asset_values))
        covariance_asset_portfolio = np.dot(cov_mat, asset_values)
        self.beta = marketvalue * (covariance_asset_portfolio / pf_variance)
        # asset_variances = np.diag(cov_mat)
        # # individual var

    def getMarginalVaRs(self):
        self.getBeta()
        self.Marginal_VaRs = self.beta * self.ValueAtRisk

    def setBetaMarket(self):
        asset_values = np.dot(self.weights, self.marketValue)
        cov_mat = self.getCovarianceMatrix(self.returnMatrix[-self.lookbackWindow:])
        portfolio_variance = self.get_variance(cov_mat)
        self.beta_market = self.marketValue * np.dot(cov_mat, asset_values) / portfolio_variance

    def setBeta(self):
        cov_mat = self.getCovarianceMatrix(self.returnMatrix[-self.lookbackWindow:])
        portfolio_variance = self.get_variance(cov_mat)
        self.beta = np.dot(cov_mat, self.weights) / portfolio_variance

    def getMarginalVaR(self):
        self.setBeta()
        self.MarginalVaR = self.beta * self.ValueAtRisk
        self.ComponentVaR = self.MarginalVaR * self.weights

    def getCovarianceMatrix(self, current_portfolio_window):
        # diagonal std matrice x correlation matrice x diagonal std matrice
        return np.cov(current_portfolio_window.T)

    def get_variance(self, cov_mat):
        # weights matrice x covariance matrice x transposed weights matrice
        return np.dot(np.dot(self.weights, cov_mat), self.weights.T)

    def vaR(self):
        self.covariance_matrix = self.getCovarianceMatrix(self.returnMatrix[-self.lookbackWindow:])
        self.variance = self.get_variance(cov_mat=self.covariance_matrix)
        self.sigma = np.sqrt(self.variance)
        self.ValueAtRisk = self.get_vaR_sigma() * self.marketValue

    def vaRSeries(self):
        self.set_vaR_series()
        for i in range(0, len(self.returnMatrix) - self.lookbackWindow):
            if i == 0:
                current_portfolio_window = self.returnMatrix[-self.lookbackWindow:]
            else:
                current_portfolio_window = self.returnMatrix[-(self.lookbackWindow + i):-i]

            cov_mat = self.getCovarianceMatrix(current_portfolio_window=current_portfolio_window)
            current_rt_point_variance = self.get_variance(cov_mat=cov_mat)
            self.sigma = np.sqrt(current_rt_point_variance)
            self.ValueAtRisk[-i - 1] = self.get_vaR_sigma() * self.marketValue


class ParametricVaREwma(ParametricVaR):
    def __init__(self, matrix, interval, weights, return_method, lookbackWindow, timeScaler, lambda_decay):
        super().__init__(matrix, interval, weights, return_method, lookbackWindow, timeScaler)
        self.lambda_decay = lambda_decay
        self.timescaler = timeScaler
        self.calculateScaledWeights()

    def get_variance(self, current_return_window):
        return np.dot(current_return_window, self.scaledweights)

    def get_lambda(self):
        return self.lambda_decay

    def get_window_length(self):
        return self.lookbackWindow

    def getEwma_A(self):
        decay_factor = self.get_lambda()
        a = 1 - decay_factor
        window_length = self.get_window_length()
        return np.divide(1 - decay_factor, np.multiply(decay_factor, 1 - a ** window_length))

    def getEwma_c(self, constant):
        decay_factor = self.get_lambda()
        return decay_factor ** constant

    def getEwma_d(self, constant, return_frame):
        rt = self.portfolioReturn[constant]
        return rt ** 2

    def getEwma_B(self, return_frame):
        window_length = self.get_window_length()
        Ewma_B = 1
        for i in range(1, window_length):
            return_index = i - 1
            Ewma_c = self.getEwma_c(constant=i)
            Ewma_d = self.getEwma_d(constant=return_index, return_frame=return_frame)
            Ewma_B *= (Ewma_c * Ewma_d)
        return Ewma_B

    def get_variance_with_EWMA(self, current_return_window):
        return self.getEwma_A() * self.getEwma_B(return_frame=current_return_window)

    def vaR(self):
        self.variance = self.get_variance_with_EWMA(current_return_window=self.portfolioReturn)
        self.ValueAtRisk = self.get_vaR_sigma() * self.marketValue

    def vaRSeries(self):
        self.set_vaR_series()
        for i in range(0, len(self.returnMatrix) - self.lookbackWindow):
            if i == 0:
                current_portfolio_window = self.HistoricalPortfolioReturns[-self.lookbackWindow:]
            else:
                current_portfolio_window = self.HistoricalPortfolioReturns[-(self.lookbackWindow + i):-i]
            current_rt_point_variance = self.get_variance_with_EWMA(current_return_window=current_portfolio_window)
            self.sigma = np.sqrt(current_rt_point_variance)
            self.ValueAtRisk[-i - 1] = self.get_vaR_sigma() * self.marketValue


class HistoricalVaR(ValueAtRisk):
    def __init__(self, matrix, interval, weights, return_method, lookbackWindow):
        super().__init__(matrix, interval, weights, return_method, lookbackWindow)

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
    def __init__(self, matrix, interval, weights, return_method, lookbackWindow, lambda_decay):
        super().__init__(matrix, interval, weights, return_method, lookbackWindow)
        self.lambda_decay = lambda_decay
        self.calculateScaledWeights()

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
    def __init__(self, matrix, interval, weights, return_method,
                 lookbackWindow, timeScaler=1, numSimulations=1000):
        self.timeScaler = timeScaler
        self.numSimulations = numSimulations

        super().__init__(matrix, interval, weights, return_method, lookbackWindow)

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

    def vaR(self):
        self.setPortfolioPrices()
        self.calculateStdMean(lookbackWindow=self.lookbackWindow)
        self.simulate(predictedDays=self.timescaler)

        return abs((self.simulated_price - self.last_price) / self.last_price)
