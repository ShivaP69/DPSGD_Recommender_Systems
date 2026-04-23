import evaluation
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def pareto_distribution(df, item_mapping):
    popular,data = evaluation.PopularItems(df, item_mapping)
    data=list(data.values())
    sorted_data = np.sort(data)
    ranks = np.arange(1, len(sorted_data) + 1)
    plt.loglog(sorted_data, ranks, marker=".", linestyle='none')
    plt.xlabel('Log(Popularity Degree)')
    plt.ylabel('Log(Rank)')
    plt.title('Log-Log Plot of Data')
    plt.show()
    xm = np.min(data)
    alpha = len(data) / np.sum(np.log(data / xm))
    print(f'Estimated parameters: xm = {xm}, alpha = {alpha}')
    D, p_value = stats.kstest(data, 'pareto', args=(alpha, xm))
    print(f'K-S test: D = {D}, p-value = {p_value}')

    # Step 4: Compare with Alternative Distributions
    # For simplicity, comparing with an exponential distribution
    exp_params = stats.expon.fit(data)
    aic_pareto = 2 * 1 - 2 * np.sum(stats.pareto.logpdf(data, alpha, xm))
    aic_expon = 2 * 2 - 2 * np.sum(stats.expon.logpdf(data, *exp_params))
    print(f'AIC for Pareto: {aic_pareto}, AIC for Exponential: {aic_expon}')
