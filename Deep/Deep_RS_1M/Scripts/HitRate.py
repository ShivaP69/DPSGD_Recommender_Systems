
import tqdm

import numpy as np
import torch
def hit_rate(model, test_data, train_data,all_movieIds):
    # User-item pairs for testing
    test_user_item_set = set(zip(test_data['userId'], test_data['movieId']))

    # Dict of all items that are interacted with by each user
    user_interacted_items = train_data.groupby('userId')['movieId'].apply(list).to_dict()
    #binary_predictions = []
    candidate_items={}
    hits = []
    for (u,i) in (test_user_item_set):
        interacted_items = user_interacted_items[u]
        not_interacted_items = set(all_movieIds) - set(interacted_items)
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99, replace=False))
        test_items = selected_not_interacted + [i]
        #k=len(test_items)
        predicted_scores = np.squeeze(model(torch.tensor([u]*100),
                                            torch.tensor(test_items)).detach().numpy())

        #binary_predictions= (((predicted_scores >= 0.5).astype(int)))

        #candidate_items[u]=[test_items[i] for i in np.argsort(predicted_scores)[::-1].tolist()]
        top10_items = [test_items[i] for i in np.argsort(predicted_scores)[::-1][0:10].tolist()]

        if i in top10_items:
            hits.append(1)
        else:
            hits.append(0)
    return hits