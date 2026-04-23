import torch
import numpy as np
from torch.utils.data import Dataset



class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds

    """

    def __init__(self, ratings, all_movieIds):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, user_interacted_items, all_movieIds):
        """
        :param user_interacted_items: the history of the user
        :param all_movieIds: all the movieIds
        :return: tensors (user_id, item, label)
        label:
            could be 1 for positive items or 0 for negative items
        """
        users, items, labels = [], [], []
        #user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()
        """for u in user_interacted_items:
            positive_items=user_interacted_items[u]
            new_positive_samples=apply_dp(positive_items, all_movieIds,epsilon)
            user_interacted_items[u]=new_positive_samples"""

        num_negatives = 4 # we have 4 random negative items per each positive one
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
























