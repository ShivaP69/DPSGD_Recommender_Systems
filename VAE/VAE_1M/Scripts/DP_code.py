import pandas as pd
import numpy as np

np.random.seed(42)
def apply_dp(pos_items, all_items, privacy_budget):
    # the probability that a positive feedback is included in the new DP dataset
    prob_pos = 1 / (np.exp(privacy_budget) + 1) + (np.exp(privacy_budget) - 1) / (np.exp(privacy_budget) + 1)
    pos_from_pos_items = np.random.choice(list(pos_items),
                                          size=np.random.binomial(len(pos_items), p=prob_pos),
                                          replace=False)

    # the probability that a negative or missing feedback is included in the new DP dataset
    prob_neg = 1 / (np.exp(privacy_budget) + 1)
    all_neg_items = list(set(all_items).difference(pos_items))
    neg_items = np.random.choice(all_neg_items, size=min(len(pos_items), len(all_neg_items)), replace=False)
    pos_from_neg_items = np.random.choice(list(neg_items),
                                          size=np.random.binomial(len(neg_items), p=prob_neg),
                                          replace=False)

    # the new DP dataset that is used for model training
    new_pos_items = list(pos_from_pos_items) + list(pos_from_neg_items)
    return new_pos_items


