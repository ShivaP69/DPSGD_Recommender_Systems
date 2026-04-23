import torch
import numpy as np
from torch.utils.data import Dataset
from DP_Code import apply_dp


class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds

    """

    def __init__(self, user_interacted_items, all_movieIds):
        self.users, self.items, self.labels = self.get_dataset(user_interacted_items, all_movieIds)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, user_interacted_items, all_movieIds):
        users, items, labels = [], [], []
        num_negatives = 4
        for u in user_interacted_items:
            users.append(u)
            for item in user_interacted_items[u]:
                items.append(item)
                labels.append(1)
                for _ in range(num_negatives):
                    negative_item = np.random.choice(all_movieIds)
                    while  negative_item in user_interacted_items[u]:
                        negative_item = np.random.choice(all_movieIds)
                    users.append(u)
                    items.append(negative_item)
                    labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)
























