import random
import pandas as pd

def final_wallet(monto, periodos, mu,sigma):
    for _ in range(periodos):
        val = random.normalvariate(mu, sigma)
        monto += monto*val
    return monto


def print_rendimiento(MONTO, Ts, MU, STD, TOTAL_SIM):
    """# MU = 0.001409
    # STD = 0.01276
    # Ts = 360
    # MONTO = 5000
    # TOTAL_SIM = 1000"""
    distr = [(final_wallet(MONTO, Ts, MU ,STD  )-MONTO)/MONTO*100 for _ in range(TOTAL_SIM)]
    distr = pd.Series(distr)
    print('======== RENDIMIENTO (%) ========')
    print(distr.describe())