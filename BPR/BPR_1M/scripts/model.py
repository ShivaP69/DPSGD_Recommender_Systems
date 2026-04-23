import torch
import torch.nn as nn


class BPR(nn.Module):
    """
    Initialize the BPR model.

    Args:
        num_users (int): Number of users in the dataset.
        num_items (int): Number of items in the dataset.
        latent_dim (int): Size of the latent factor vectors for users and items.
        weight_decay (float): L2 regularization term (used externally in the loss if needed).
    """
    def __init__(self, num_users, num_items, latent_dim, weight_decay=1e-5):
        super(BPR, self).__init__()
        # Embedding layer for users: each user is represented by a latent vector
        self.user_embeddings = nn.Embedding(num_users, latent_dim)

        # Embedding layer for items: each item is represented by a latent vector
        self.item_embeddings = nn.Embedding(num_items, latent_dim)
        self.weight_decay = weight_decay  # L2 regularization strength

    def forward(self, user, pos_item, neg_item):
        """
        Perform the forward pass for one batch of triplets (user, positive item, negative item).

        Args:
            user (Tensor): Tensor of user IDs (batch_size,)
            pos_item (Tensor): Tensor of positive item IDs (batch_size,)
            neg_item (Tensor): Tensor of negative item IDs (batch_size,)

        Returns:
            pos_scores (Tensor): Predicted preference score for the positive item (batch_size,)
            neg_scores (Tensor): Predicted preference score for the negative item (batch_size,)
        """
        user_embed = self.user_embeddings(user)
        pos_item_embed = self.item_embeddings(pos_item)
        neg_item_embed = self.item_embeddings(neg_item)
        pos_scores = torch.sum(user_embed * pos_item_embed, dim=1)  # Compute the dot product between user and positive item (preference score)
        neg_scores = torch.sum(user_embed * neg_item_embed, dim=1)  # Compute the dot product between user and negative item

        # Return the scores so BPR loss can be applied externally
        return pos_scores, neg_scores





