# %%
from cvxopt.solvers import qp, options
from cvxopt.blas import dot
from cvxopt import matrix, spdiag
import random as rd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from simulation import print_rendimiento
from urllib.request import urlopen, Request
import json

from numpy.linalg import inv, pinv
import scipy.optimize
import random

from market_tools import forecast_12months, create_views, create_views_and_link_matrix

WEEK = False
MONTH = False
MONTOUSD = 4900

shares = ['GOOG', 'AAPL', 'MSFT', 'AMZN',
          'ACN', 'TREX', 'TSLA', 'NVDA', 'AMD']  # '0700.HK']
shares_view = [['TSLA', 'NVDA'], ['ACN', 'AMZN'],
               ['AAPL', 'AMZN'], ['TREX', 'NVDA']]
low_up_bound = [-0.02 for _ in shares] + \
    [2.0/len(shares) for _ in shares]

DAYS = 30*12
RISK_FREE = 0.02
TAU = 0.025
# https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/MSFT?lang=en-US&region=US&symbol=MSFT&padTimeSeries=true&type=trailingMarketCap&period1=493590046&period2=1637385555

# %% FUNCTIONS


def str_to_datetime(col):
    col = col.apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d"))
    return col


def load_table(name, init_time, end_time):
    url = 'https://query1.finance.yahoo.com/v7/finance/download/'+name + \
        '?period1=' + str(init_time)+'&period2='+str(end_time) + \
        '&interval=1d&events=history&includeAdjustedClose=true'
    table = pd.read_csv(url, )  # usecols=['Date', 'Close'],)
    table = table[['Date', 'Close']]
    table.Date = str_to_datetime(table.Date)
    table.rename(columns={'Close': name.split('.')[0]}, inplace=True)
    return table


def request_url(url):
    if not url.startswith("http"):
        raise RuntimeError(
            "Incorrect and possibly insecure protocol in url " + url)

    httprequest = Request(url, headers={"Accept": "application/json"})

    with urlopen(httprequest) as response:
        if response.status == 200:
            val = response.read().decode()
            return json.loads(val)['timeseries']['result'][0]['trailingMarketCap'][0]['reportedValue']['raw']
        else:
            raise RuntimeError("Error in request " + url)


def get_capitalization(share, init_time, end_time):
    url = 'https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/' + share + \
        '?lang=en-US&region=US&symbol='+share+'&padTimeSeries=true&type=trailingMarketCap&' + \
        'period1=' + str(init_time)+'&period2='+str(end_time)
    capitalization = request_url(url)
    return capitalization


def get_unix_time():
    init_time = datetime.now() - timedelta(days=DAYS)
    end_time = datetime.now()

    def unix(dt):
        timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
        return int(timestamp)

    init_time = unix(init_time)
    end_time = unix(end_time)
    return init_time, end_time


def bulk_stocks(shares, shares_view):
    init_time, end_time = get_unix_time()
    caps = []
    forecast = []
    share_view2 = []
    views = []
    for i, share in enumerate(shares):
        if i == 0:
            data = load_table(share, init_time, end_time)

        else:
            temp = load_table(share, init_time, end_time)
            data = data.merge(temp, on=['Date'])
        caps.append(get_capitalization(share, init_time, end_time))
        change_12m = (forecast_12months(share) -
                      data.iloc[-1][share])/data.iloc[-1][share]
        if share in [item for sublist in shares_view for item in sublist]:
            share_view2.append(share)
            forecast.append(change_12m)
    # print(np.diag(caps))
    w_caps = np.array(caps)/sum(caps)
    for share1, share2 in shares_view:
        forecast_ii = [forecast[share_view2.index(
            share1)], forecast[share_view2.index(share2)]]
        view = create_views([share1, share2], forecast_ii)
        views = views + view
    # print(weights_cap)
    return data, w_caps, views

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
    b_ = [(0.00, 1.) for i in range(n)]
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
data, w_caps, views = bulk_stocks(shares, shares_view)

# %%
Q, P = create_views_and_link_matrix(shares, views)

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

# para todo el periodo de analisis
mean_returns = (1+mean_returns)**RECORDS - 1
cov_returns = cov_returns * (RECORDS)

# %%
print('Días de análisis : ', DAYS)
print('Cantidad de records analizados: ', RECORDS)
print('Monto total de inversión: {} usd'.format(MONTOUSD))

# %% weights of tangency portfolio with respect to sharpe ratio maximization
print("{} Mean-Variance Optimization (historical) {}".format("#"*10, "#"*10))
weights = solve_weights(mean_returns.copy(), cov_returns.copy(), RISK_FREE)
mean, var = port_mean_var(weights, mean_returns.copy(), cov_returns.copy(),)
std = np.sqrt(var)
for name, fp in zip(names, weights):
    print('{} : {:.2f}% -> {} USD'.format(name, fp*100, round(fp*MONTOUSD, 2)))

print("Portafolio return: {:.4%} -> {} USD".format(mean,
      round(MONTOUSD*mean, 2)))
print("Portafolio standard deviation: {:.4%} -> {} USD".format(
    std, round(MONTOUSD*std, 2)))


# %% print_rendimiento(MONTOUSD, DAYS, mean, var, 4000)
# Black litterman reverse optimization
print("{} Black-litterman reverse optimization {}".format("#"*10, "#"*10))
# Calculate portfolio historical return and variance
mean, var = port_mean_var(w_caps, mean_returns.copy(), cov_returns.copy())

lmb = (mean - RISK_FREE) / var  # Calculate risk aversion
# Calculate equilibrium excess returns
Pi = np.dot(np.dot(lmb, cov_returns.copy()), w_caps)
weights = solve_weights(Pi + RISK_FREE, cov_returns.copy(), RISK_FREE)
mean, var = port_mean_var(weights, mean_returns.copy(), cov_returns.copy(),)
std = np.sqrt(var)
for name, fp in zip(names, weights):
    print('{} : {:.2f}% -> {} USD'.format(name, fp*100, round(fp*MONTOUSD, 2)))

print("Portafolio return: {:.4%} -> {} USD".format(mean,
      round(MONTOUSD*mean, 2)))
print("Portafolio standard deviation: {:.4%} -> {} USD".format(
    std, round(MONTOUSD*std, 2)))
# print_rendimiento(MONTOUSD, DAYS, mean, var, 4000)
print("#"*50)

# %% print_rendimiento(MONTOUSD, DAYS, mean, var, 4000)
# CONFIDENCE = .01
# TAU = 0.02  # 1.0/(CONFIDENCE-1.0)
# RISK_FREE = -0.002
# # Black litterman reverse optimization
print("{} Black-litterman Full Algorithms {}".format("#"*10, "#"*10))
# Calculate portfolio historical return and variance
mean, var = port_mean_var(w_caps, mean_returns.copy(), cov_returns.copy())
#####
lmb = (mean - RISK_FREE) / var  # Calculate risk aversion
# Calculate equilibrium excess returns
Pi = np.dot(np.dot(lmb, cov_returns.copy()), w_caps)

####
# Calculate omega - uncertainty matrix about views
# 0.025 * P * C * transpose(P)
C = cov_returns.copy()
omega = np.dot(np.dot(np.dot(TAU, P), C), np.transpose(P))
# Calculate equilibrium excess returns with views incorporated
sub_a = np.linalg.inv(np.dot(TAU, C))
sub_b = np.dot(np.dot(np.transpose(P), np.linalg.inv(omega)), P)
sub_c = np.dot(np.linalg.inv(np.dot(TAU, C)), Pi)
sub_d = np.dot(np.dot(np.transpose(P), np.linalg.inv(omega)), Q)
Pi_adj = np.dot(np.linalg.inv(sub_a + sub_b), (sub_c + sub_d))

weights = solve_weights(Pi_adj + RISK_FREE, cov_returns.copy(), RISK_FREE)
mean, var = port_mean_var(weights, Pi_adj + RISK_FREE, cov_returns.copy(),)
std = np.sqrt(var)
for name, fp in zip(names, weights):
    print('{} : {:.2f}% -> {} USD'.format(name, fp*100, round(fp*MONTOUSD, 2)))

print("Portafolio return: {:.4%} -> {} USD".format(mean,
      round(MONTOUSD*mean, 2)))
print("Portafolio standard deviation: {:.4%} -> {} USD".format(
    std, round(MONTOUSD*std, 2)))
# print_rendimiento(MONTOUSD, DAYS, mean, var, 4000)
print("#"*50)


# # Aplicamos los pesos de la capitalizacion a la matriz de covarianza
# cov_returns = np.matmul(
#     np.matmul(weights, cov_returns), np.transpose(weights))
# # Aplicamos los pesos de la capitalizacion a la media de retornos
# mean_returns = np.matmul(mean_returns, np.transpose(weights))


# %%
# OPTIMIZACION
options["show_progress"] = False

# %%
n_prices = len(data.columns.tolist())
g1 = np.diag([-1.0]*n_prices)
g2 = np.diag([1.0]*n_prices)

n = mean_returns.shape[0]
P = matrix(cov_returns)
q = matrix(0.0, (n, 1))
G = matrix(np.append(g1, g2, 0))
#h = matrix(0.0, (n, 1))
h = matrix(np.stack([[float(i)] for i in low_up_bound]))
A = matrix(1.0, (1, n))
b = matrix(1.0)


# %%
# Calculamos la solucion
sol = qp(P, q, G, h, A, b)
pesos = sol["x"]
porcent = 100*pesos.T
# %%
print("{} Minimal Variance Optimization {}".format("#"*10, "#"*10))
for name, fp in zip(names, porcent):
    print('{} : {:.2f}% -> {} USD'.format(name, fp, round(fp*MONTOUSD/100, 2)))


# %%
# Calculamos las estadisticas de la posicion optima
min_variance = dot(pesos, P*pesos)
min_std = np.sqrt(min_variance)
min_std_return = dot(matrix(mean_returns), pesos)
print("Portafolio return: {:.4%} -> {} USD".format(min_std_return,
      round(MONTOUSD*min_std_return, 2)))
print("Portafolio standard deviation: {:.4%} -> {} USD".format(
    min_std, round(MONTOUSD*min_std, 2)))
# print_rendimiento(MONTOUSD, DAYS, min_std_return, min_std, 4000)
# %%
# Foronterda eficiente
max_return = np.max(mean_returns)
print("Portafolio max return: {:.4%}".format(max_return))
# dividimos el eje y de la frontera eficiente en 100 puntos
m = 100
returns = np.linspace(min_std_return, max_return, m)
# print(returns[-10:])
# %%
risk_k = []
k_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
for k in k_values:
    if k == 0:
        n = mean_returns.shape[0]
        P = matrix(cov_returns)
        q = matrix(0.0, (n, 1))
        G = matrix(np.diag([-1.0]*n_prices))
        h = matrix(k, (n, 1))
        A1 = matrix(1.0, (1, n))
        A2 = matrix(mean_returns).T
        A = matrix([A1, A2])
    else:
        n = mean_returns.shape[0]
        P = matrix(cov_returns)
        q = matrix(0.0, (n, 1))
        G = matrix(np.diag([1.0]*n_prices))
        h = matrix(k, (n, 1))
        A1 = matrix(1.0, (1, n))
        A2 = matrix(mean_returns).T
        A = matrix([A1, A2])

    risk = []
    pesos_opt = np. zeros([m, n])
    for i in range(m):
        mu = returns[i]
        b = matrix([1.0, mu])
        sol = qp(P, q, G, h, A, b)
        if sol["status"] == "optimal":
            pesos = sol["x"]
            pesos_opt[i, :] = np.array(pesos).flatten()
            port_variance = dot(pesos, P*pesos)
            port_std = np.sqrt(port_variance)
            # port_return = dot(A2, pesos)
            risk.append(port_std)
    risk = risk[: -1]
    risk_k.append(risk)

# %%
plt.figure(figsize=(15, 8))

for ind, k in enumerate(k_values):
    plt.plot(risk_k[ind], returns[: len(risk_k[ind])], label=str(k))
plt.legend(loc='best')
plt.title("Frontera eficiente")
plt.ylabel("Retorno")
plt.xlabel("Riesgo")
plt.grid(True)
plt.show(block=False)
