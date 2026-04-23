

import torch
import torch.nn as nn


class BPR(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, weight_decay=1e-5):
        super(BPR, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, latent_dim)
        self.item_embeddings = nn.Embedding(num_items, latent_dim)
        self.weight_decay = weight_decay  # L2 regularization strength

    def forward(self, user, pos_item, neg_item):
        user_embed = self.user_embeddings(user)
        pos_item_embed = self.item_embeddings(pos_item)
        neg_item_embed = self.item_embeddings(neg_item)
        pos_scores = torch.sum(user_embed * pos_item_embed, dim=1)
        neg_scores = torch.sum(user_embed * neg_item_embed, dim=1)
        return pos_scores, neg_scores


def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

def regularization_loss(model):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param)
    return l2_reg

def predict_score(user_ids,item_ids, model):
    user_embed = model.user_embeddings(user_ids)
    item_embed = model.item_embeddings(item_ids)
    scores = torch.sum(user_embed * item_embed, dim=1)
    return scores
