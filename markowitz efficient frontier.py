import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as wb

# Data extraction
tickers = ['MSFT', 'AAPL', 'AMZN']
sec_data = pd.DataFrame()
for t in tickers:
    sec_data[t] = wb.DataReader(t, data_source='yahoo', start='1998-01-01')['Adj Close']

log_returns = np.log(sec_data.pct_change() + 1)

# fill in in the arrays with randomized weights

pfolio_returns = []
pfolio_vol = []
num_assets = len(tickers)

for i in range(20):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    pfolio_returns.append(np.sum(weights*log_returns.mean()*250))
    pfolio_vol.append(np.dot(weights.T, np.dot(log_returns.cov()*250, weights)))
pfolio_vol = np.array(pfolio_vol)
pfolio_returns = np.array(pfolio_returns)

# Assign keys into dictionary
Portfolio = pd.DataFrame({'Returns': pfolio_returns, 'Volatility': pfolio_vol})

(Portfolio.plot(x='Volatility', y='Returns', kind='scatter', figsize=(15, 6)))
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Returns')
plt.vlines()
plt.show()


# Portfolio Sharpe Ratio
annual_returns = log_returns.mean() * 250
risk_free_rate = 0.0148
standard_deviation = (log_returns.std()*250**0.5)

sharpe_ratio = (annual_returns - risk_free_rate)/(log_returns.std() * 250 ** 0.5)

