

import torch
import torch.nn as nn


class SVD(nn.Module):
    def __init__(self, num_users, num_items, latent_dimension, weight_decay=1e-5):
        super(SVD, self).__init__()
        self.user_embeddings = torch.nn.Embedding(num_users, latent_dimension)
        self.item_embeddings = torch.nn.Embedding(num_items, latent_dimension)
        self.weight_decay = weight_decay

    def forward(self, user, item):
        user_embed = self.user_embeddings(user)
        item_embed = self.item_embeddings(item)

        return torch.sum(user_embed * item_embed, dim=1)


def predict_score(user_ids, item_ids, model):
    user_embed = model.user_embeddings(user_ids)
    item_embed = model.item_embeddings(item_ids)
    scores = torch.sum(user_embed * item_embed, dim=1)
    return scores
