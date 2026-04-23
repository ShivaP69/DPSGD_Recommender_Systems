


import numpy as np
import torch
def generate_item_candidates(item_mapping, test_items,predicted_scores, user_id,test_user_item_dict,Liked):

        items = []
        item_ids_and_scores = []
        # not_interacted_items = set(all_movieIds) - set(Liked[user_id]) #-liked
        # selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99, replace=False))
        # selected_not_interacted= list(not_interacted_items)
        # test_items = selected_not_interacted + test_user_item_dict[user_id] # instead of all items
        # test_items = [item_id for item_id in test_items if item_id in item_mapping]

        # k=len(test_items)
        # predicted_scores = np.squeeze(predict_score(torch.tensor([user_id]*k),torch.tensor(test_items),model).detach().numpy()) # could be interpreted as score

        item_ids_and_scores.extend(zip(test_items, predicted_scores))
        item_scores_dict = {item_id: score for item_id, score in item_ids_and_scores}


        for item_id in test_items:
            item = item_mapping[item_id]

            item.score = item_scores_dict[item_id]
            items.append(item)

        return items


def weighted_genre_distribution(items):  #interacted_items >>> real id of interacted items
    """Compute the genre distribution for a given list of Items."""
    weighted_distr = {}
    count_total=0
    for item in items:
        for genre,score in item.genres.items():  #score is rate
            genre_freq = weighted_distr.get(genre, 0.)
            weighted_distr[genre] = genre_freq + 1
            count_total+=1


    # we normalize the summed up probability so it sums up to 1
    # and round it to three decimal places, adding more precision
    # doesn't add much value and clutters the output
    for genre, freq in weighted_distr.items():
        genre_total_freq = round(freq / count_total, 3)
        weighted_distr[genre] = genre_total_freq

    return weighted_distr


def compute_genre_distr(items):
    """Compute the genre distribution for a given list of Items."""
    distr = {}
    rate_of_genre={}
    for item in items:
        for genre, score in item.genres.items():
            genre_score = distr.get(genre, 0.)
            distr[genre] = genre_score + score

    # we normalize the summed up probability so it sums up to 1
    # and round it to three decimal places, adding more precision
    # doesn't add much value and clutters the output
    for item, genre_score in distr.items():
        normed_genre_score = round(genre_score / len(items), 3)
        distr[item] = normed_genre_score
    weighted_distr = weighted_genre_distribution(items)
    for item,score in distr.items():
      rate_of_genre[item]=distr[item] * weighted_distr[item]

    return rate_of_genre
def normalize_distribution(distr):
    total = sum(distr.values())
    if total == 0:
        # Avoid division by zero; return the original distribution if it sums to zero
        return distr
    return {genre: score / total for genre, score in distr.items()}


def compute_kl_divergence(interacted_distr, reco_distr, score=0, alpha=0.01):
    """
    KL (p || q), the lower the better.

    alpha is not really a tuning parameter, it's just there to make the
    computation more numerically stable.
    """
    kl_div = 0.
    # Normalize the distributions
    if score ==1:
        interacted_distr = normalize_distribution(interacted_distr)
        reco_distr = normalize_distribution(reco_distr)

    for genre, score in interacted_distr.items():
        reco_score = reco_distr.get(genre, 0.)
        if score == 0 and reco_score == 0:
            kl_div += 0
        else:
            reco_score = (1 - alpha) * (reco_score) + alpha * (score)
        if score == 0 or reco_score == 0:
            kl_div += 0
        else:
            kl_div += score * np.log2(score / reco_score)


    return kl_div

# a point : KLD does not consider a case that we have new genres in recommended list > because it is not relevant to KLD

def compute_utility(reco_items, interacted_distr, lmbda):
    """
    Our objective function for computing the utility score for
    the list of recommended items.

    lmbda : float, 0.0 ~ 1.0, default 0.5
        Lambda term controls the score and calibration tradeoff,
        the higher the lambda the higher the resulting recommendation
        will be calibrated. Lambda is keyword in Python, so it's
        lmbda instead ^^
    """
    reco_distr = compute_genre_distr(reco_items)
    kl_div = compute_kl_divergence(interacted_distr, reco_distr)

    total_score = 0.0
    for item in reco_items:
        total_score += item.score

    # kl divergence is the lower the better, while score is
    # the higher the better so remember to negate it in the calculation
    utility = (1 - lmbda) * total_score - lmbda * kl_div
    return utility

def calib_recommend(items, interacted_distr, topn, lmbda):
    """
    start with an empty recommendation list,
    loop over the topn cardinality, during each iteration
    update the list with the item that maximizes the utility function.
    """
    calib_reco = []
    for _ in range(topn):
        max_utility = -np.inf
        for item in items:  # candidate items
            if item in calib_reco: # ignore duplicate items
                continue

            utility = compute_utility(calib_reco + [item], interacted_distr, lmbda) # add item to best_item if increase utility of calib_reco
            if utility > max_utility:
                max_utility = utility
                best_item = item

        calib_reco.append(best_item)

    return calib_reco