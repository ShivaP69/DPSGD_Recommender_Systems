import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import defaultdict
import Preprocess

import torch
import numpy as np
from torch.utils.data import Dataset

def creat_data(user_interacted_items,all_movieIds,batch_size, num_negatives=4):

    data = {
    'userId': [],
    'pos_id': [],
    'neg_id': []
    }

    for user in user_interacted_items.keys():
        for item in user_interacted_items[user]:
            def generate_negative():
                neg = item
                while neg in user_interacted_items[user]:
                    neg = np.random.choice(all_movieIds)
                return neg

            neg_samples= [generate_negative() for i in range(num_negatives)]
            for i in range(num_negatives):
                data['userId'].append(user)
                data['pos_id'].append(item)
                data['neg_id'].append(neg_samples[i])

    data_tensor= TensorDataset(torch.tensor(data['userId']), torch.tensor(data['pos_id']), torch.tensor(data['neg_id']))

    return DataLoader(data_tensor, batch_size, shuffle=True)

