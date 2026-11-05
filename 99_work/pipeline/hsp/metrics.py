"""Per-basin KGE (Gupta et al. 2009), NSE and KGE components. NaN observations are masked."""
import numpy as np
import pandas as pd

from config import experiment as E


def kge_parts(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[m], sim[m]
    out = {"n": int(m.sum()), "KGE": np.nan, "NSE": np.nan, "r": np.nan, "alpha": np.nan, "beta": np.nan}
    if len(o) < 2 or np.std(o) == 0:
        return out
    r = np.corrcoef(o, s)[0, 1] if np.std(s) > 0 else np.nan
    alpha = np.std(s) / np.std(o)
    beta = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
    out.update(r=r, alpha=alpha, beta=beta,
               KGE=1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) if np.isfinite(r) else np.nan,
               NSE=1 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2))
    return out


def nse(obs, sim):
    return kge_parts(obs, sim)["NSE"]


def period_metrics(obs: pd.Series, sim: pd.Series) -> list:
    """Rows of {period, KGE, NSE, r, alpha, beta, n} for cal / val / test."""
    rows = []
    for name, (a, b) in E.PERIODS.items():
        rows.append({"period": name, **kge_parts(obs.loc[a:b], sim.reindex(obs.loc[a:b].index))})
    return rows
