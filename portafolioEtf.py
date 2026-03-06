# %%
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from simulation import print_rendimiento, print_rendimiento_backtest
from cvxopt.solvers import qp, options
from cvxopt.blas import dot
from cvxopt import matrix
import requests
import tqdm
import scipy.optimize
import os
WEEK = False
MONTH = False
MIN_VARIANCE = False  # True = Minimal Variance, False = Mean-Variance (Sharpe)
MONTOUSD = 2300
MONTHLY_DELTA = 500  # USD added to the portfolio every month (DCA)

shares = [
    'XLE', 'XLU', 'QQQ', 'SCHD', 'SPY', 'GOOG', 'GLD', 'AAPL', 'MSFT', 'TSM', 'AMD',
]
W_LIMITS = (0.00, 1/len(shares)*1.5)

DAYS = 720  # calendar days for data analysis
SIM_DAYS = 252  # trading days for simulation (~1 year)
RISK_FREE_ANUL_PERC = 5
RISK_FREE = (1 + RISK_FREE_ANUL_PERC/100) ** (1/365) - 1

# %% FUNCTIONS


def str_to_datetime(col):
    col = col.apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d"))
    return col


def load_table(name, init_time, end_time):
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)

    # Define cache filename (name_day)
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache_file = f"temp/{name}_{today_str}.csv"

    # If file exists, load from cache
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=["Date"])

    # Otherwise fetch from API
    url = "https://query1.finance.yahoo.com/v8/finance/chart/"+name + \
        "?period1=" + str(init_time)+"&period2="+str(end_time) + \
        "&interval=1d"
    headers = {
        "User-Agent": "Mozilla/5.0",
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]

    # Convert timestamps to readable datetime
    dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps]

    # Build dataframe with only Date and Close
    table = pd.DataFrame({
        "Date": dates,
        name.split('.')[0]: closes
    })

    # Save to cache
    table.to_csv(cache_file, index=False)

    return table


def get_unix_time():
    init_time = datetime.now() - timedelta(days=DAYS)
    end_time = datetime.now()

    def unix(dt):
        timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
        return int(timestamp)

    init_time = unix(init_time)
    end_time = unix(end_time)
    return init_time, end_time


def bulk_stocks(shares):
    init_time, end_time = get_unix_time()
    for i, share in tqdm.tqdm(enumerate(shares), total=len(shares)):
        print(share)
        if i == 0:
            data = load_table(share, init_time, end_time)
        else:
            temp = load_table(share, init_time, end_time)
            data = data.merge(temp, on=['Date'])
    return data

# Calculates portfolio mean return


def port_mean(W, R):
    return np.sum(R*W)

# Calculates portfolio variance of returns


def port_var(W, C):
    return np.dot(np.dot(W, C), W)

# Combination of the two functions above - mean and variance of returns calculation


def port_mean_var(W, R, C):
    return port_mean(W, R), port_var(W, C)

# Given risk-free rate, assets returns and covariances, this
# function calculates weights of tangency portfolio with respect to
# sharpe ratio maximization


def solve_weights(R, C, rf):
    def fitness(W, R, C, rf):
        # calculate mean/variance of the portfolio
        mean, var = port_mean_var(W, R, C)
        util = (mean - rf) / np.sqrt(var)		# utility = Sharpe ratio
        return 1/util						# maximize the utility, minimize its inverse value
    n = len(R)
    W = np.ones([n])/n						# start optimization with equal weights
    # weights for boundaries between 0%..100%. No leverage, no shorting
    b_ = [W_LIMITS for i in range(n)]
    c_ = ({'type': 'eq', 'fun': lambda W: np.sum(W)-1.}
          )  # Sum of weights must be 100%
    optimized = scipy.optimize.minimize(
        fitness, W, (R, C, rf), method='SLSQP', constraints=c_, bounds=b_)
    if not optimized.success:
        raise BaseException(optimized.message)
    # w = np.diag(optimized.x)
    w = optimized.x
    return w


# %% MAIN
data = bulk_stocks(shares)

# %%

if WEEK:
    data['step'] = data.Date.apply(lambda x: str(
        (x).isocalendar()[1]) + '-' + str((x).isocalendar()[0]) + '-' + str(x.month))
    data = data.groupby('step').last()
elif MONTH:
    data['step'] = data.Date.apply(lambda x: str(
        (x).isocalendar()[0]) + '-' + str(x.month))
    data = data.groupby('step').last()


data = data.sort_values('Date', ascending=False)
data = data.set_index('Date')
names = data.columns.tolist()
# data.head()
RECORDS = len(data)
data.describe()


# %%
data.interpolate(method="time", limit_direction="backward", inplace=True)
data.head(10)
returns = data.pct_change(periods=-1)

# %%
# Calculamos las medias de cada columna, y la matriz de covarianza entre ellas
mean_returns = np.array(returns.mean())

cov_returns = np.array(returns.cov())

# %%
print('Días de análisis : ', DAYS)
print('Cantidad de records analizados: ', RECORDS)
print('Monto total de inversión: {} usd'.format(MONTOUSD))

# %% Optimization
if MIN_VARIANCE:
    print("{} Minimal Variance Optimization {}".format("#"*10, "#"*10))
    options["show_progress"] = False
    n_prices = len(names)
    low_up_bound = [0.0 for _ in shares] + [W_LIMITS[1] for _ in shares]
    P = matrix(np.array(cov_returns, dtype=float))
    q = matrix(0.0, (n_prices, 1))
    G = matrix(np.append(np.diag([-1.0]*n_prices), np.diag([1.0]*n_prices), 0))
    h = matrix(np.stack([[float(i)] for i in low_up_bound]))
    A = matrix(1.0, (1, n_prices))
    b = matrix(1.0)
    sol = qp(P, q, G, h, A, b)
    weights = np.array(sol["x"]).flatten()
    mean = np.sum(mean_returns * weights)
    var = np.dot(np.dot(weights, cov_returns), weights)
    std = np.sqrt(var)
else:
    print("{} Mean-Variance Optimization (historical) {}".format("#"*10, "#"*10))
    weights = solve_weights(mean_returns.copy(), cov_returns.copy(), RISK_FREE)
    mean, var = port_mean_var(weights, mean_returns.copy(), cov_returns.copy())
    std = np.sqrt(var)

for name, fp in zip(names, weights):
    print('{} : {:.2f}% -> {} USD'.format(name, fp*100, round(fp*MONTOUSD, 2)))

print("Portafolio return: {:.4%} -> {} USD".format(mean,
      round(MONTOUSD*mean, 2)))
print("Portafolio standard deviation: {:.4%} -> {} USD".format(
    std, round(MONTOUSD*std, 2)))
print_rendimiento(MONTOUSD, SIM_DAYS, mean, std, 4000)
print_rendimiento_backtest(MONTOUSD, shares, weights, SIM_DAYS, RISK_FREE_ANUL_PERC/100, MONTHLY_DELTA)
print("#"*50)
# %%
