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
WEEK = False
MONTH = False
MONTOUSD = 5000
shares = ['GOOG', 'AAPL', 'MSFT', 'AMZN',
          'ACN', 'TREX', 'COIN', 'TSLA']  # '0700.HK']
low_up_bound = [-0.01, -0.01, -0.01, -0.01, -0.0, -0.0, -0.0, -0.0] + \
    [0.5, 0.5, 0.5, 0.5, 0.8, 0.5, 0.5, 0.5]
DAYS = 30*6


def str_to_datetime(col):
    col = col.apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d"))
    return col


def load_table(name, init_time, end_time):
    url = 'https://query1.finance.yahoo.com/v7/finance/download/'+name + \
        '?period1=' + str(init_time)+'&period2='+str(end_time) + \
        '&interval=1d&events=history&includeAdjustedClose=true'
    table = pd.read_csv(url, usecols=['Date', 'Close'],)
    table.Date = str_to_datetime(table.Date)
    table.rename(columns={'Close': name.split('.')[0]}, inplace=True)
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
    for i, share in enumerate(shares):
        if i == 0:
            data = load_table(share, init_time, end_time)
        else:
            temp = load_table(share, init_time, end_time)
            data = data.merge(temp, on=['Date'])
    data.dtypes
    return data


data = bulk_stocks(shares)

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
returns.describe()
# print(cov_returns)

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
print('Días de análisis : ', DAYS)
print('Cantidad de records analizados: ', RECORDS)
print('Monto total de inversión: {} usd'.format(MONTOUSD))
# %%
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
print_rendimiento(MONTOUSD, DAYS, min_std_return, min_std, 4000)
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
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
