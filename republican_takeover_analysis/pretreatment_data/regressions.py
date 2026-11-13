"""Full-sample OLS associations, separate from the forest's overlap target."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

OUTCOMES = ['clean_announcements', 'manufacturing_announcements', 'wind_solar_net_mw']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    out = args.results
    units = pd.read_csv(out / 'expanded/areas.csv')
    baseline = (out / 'baseline/features.txt').read_text().splitlines()
    expanded = (out / 'expanded/features.txt').read_text().splitlines()
    assert len(units) == 196 and units.unit.is_unique and units.state.nunique() == 25
    z = (units[expanded] - units[expanded].mean()) / units[expanded].std()
    rows, fits = [], []
    for model, features in [('baseline', baseline), ('expanded', expanded)]:
        x = pd.concat([units[['gainer']].astype(float), z[features],
                       pd.get_dummies(units.state, dtype=float)], axis=1)
        assert np.linalg.matrix_rank(x) == len(x.columns)
        for outcome in OUTCOMES:
            y = units[outcome + '_delta']
            fit = sm.OLS(y, x, hasconst=True).fit()
            cov = cov_cluster(fit, units.state, use_correction=True)
            # Independently reconstruct both coefficients and the CR1 covariance.
            inv = np.linalg.pinv(x.T @ x)
            scores = x.mul(fit.resid, axis=0).groupby(units.state).sum().to_numpy()
            manual = inv @ (scores.T @ scores) @ inv * 25 / 24 * (len(units)-1) / (len(units)-x.shape[1])
            assert np.allclose(cov, manual, rtol=1e-7, atol=1e-7)
            assert np.allclose(np.linalg.lstsq(x, y, rcond=None)[0], fit.params, rtol=1e-7, atol=1e-7)
            for j, term in enumerate(['gainer'] + features):
                estimate = float(fit.params[term])
                se = float(np.sqrt(cov[j, j]))
                critical = stats.t.ppf(.975, 24)
                rows.append(dict(model=model, outcome=outcome, term=term,
                                 estimate=estimate, se=se,
                                 lo=estimate-critical*se, hi=estimate+critical*se,
                                 p=2*stats.t.sf(abs(estimate/se), 24)))
            fits.append(dict(model=model, outcome=outcome, n=196, states=25,
                             parameters=x.shape[1], residual_df=fit.df_resid,
                             adjusted_r2=fit.rsquared_adj))
    pd.DataFrame(rows).to_csv(out / 'regression_coefficients.csv', index=False)
    pd.DataFrame(fits).to_csv(out / 'regression_fit.csv', index=False)
    print('Reproduced six full-sample OLS association models (25 state clusters).')


if __name__ == '__main__':
    main()
