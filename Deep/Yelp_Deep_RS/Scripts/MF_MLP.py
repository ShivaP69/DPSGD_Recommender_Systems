
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from DataProcess import MovieLensTrainDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch.nn.init as init


def glorot_uniform(layer):
    fan_in, fan_out = layer.in_features, layer.out_features
    limit = np.sqrt(6. / (fan_in + fan_out))
    layer.weight.data.uniform_(-limit, limit)


class NeuMF(nn.Module):
    def __init__(self, nb_users, nb_items,
                 mf_dim, mlp_layer_sizes, dropout,weight_decay=1e-3):

        if mlp_layer_sizes[0] % 2 != 0:
            raise RuntimeError('u dummy, mlp_layer_sizes[0] % 2 != 0')
        super(NeuMF, self).__init__()
        nb_mlp_layers = len(mlp_layer_sizes)

        self.mf_user_embed = nn.Embedding(nb_users, mf_dim)
        self.mf_item_embed = nn.Embedding(nb_items, mf_dim)

        self.mlp_user_embed = nn.Embedding(nb_users, mlp_layer_sizes[0] // 2) # division by 2 is just a choice for distributing the embedding size,
        self.mlp_item_embed = nn.Embedding(nb_items, mlp_layer_sizes[0] // 2)
        self.dropout = dropout
        self.weight_decay = weight_decay

        # define number of layers in the MLP model
        self.mlp = nn.ModuleList()
        for i in range(1, nb_mlp_layers):
            self.mlp.extend([nn.Linear(mlp_layer_sizes[i - 1], mlp_layer_sizes[i])])  # noqa: E501 input , output of linear layers of the MLP

        self.mf= nn.Linear(mf_dim, mlp_layer_sizes[-1] )

        self.final = nn.Linear(mlp_layer_sizes[-1] + mlp_layer_sizes[-1] , 1)

        #self.mf_user_embed.weight.data.normal_(0., 0.02)
        #self.mf_item_embed.weight.data.normal_(0., 0.02)
        #self.mlp_user_embed.weight.data.normal_(0., 0.02)
        #self.mlp_item_embed.weight.data.normal_(0., 0.02)

        init.normal_(self.mf_user_embed.weight, mean=0, std=0.01)
        init.normal_(self.mf_item_embed.weight, mean=0, std=0.01)
        init.normal_(self.mlp_user_embed.weight, mean=0, std=0.01)
        init.normal_(self.mlp_item_embed.weight, mean=0, std=0.01)

        def glorot_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6. / (fan_in + fan_out))
            layer.weight.data.uniform_(-limit, limit)

        def lecunn_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features  # noqa: F841, E501
            limit = np.sqrt(3. / fan_in)
            layer.weight.data.uniform_(-limit, limit)

        for layer in self.mlp:
            if type(layer) != nn.Linear:
                continue
            glorot_uniform(layer)
        lecunn_uniform(self.final)

    def forward(self, user, item, sigmoid=True):
        xmfu = self.mf_user_embed(user)
        xmfi = self.mf_item_embed(item)
        xmf = xmfu * xmfi
        xmf =self.mf(xmf)
        xmf= nn.functional.relu(xmf)

        xmlpu = self.mlp_user_embed(user)
        xmlpi = self.mlp_item_embed(item)
        xmlp = torch.cat((xmlpu, xmlpi), dim=1) # total size of xmlp is mlp_layer_sizes[i - 1] and because of this we have mlp_layer_sizes[0] // 2

        for i, layer in enumerate(self.mlp):
            xmlp = layer(xmlp) # send xmlp to the layer
            xmlp = nn.functional.relu(xmlp)
            if self.dropout != 0:
                xmlp = nn.functional.dropout(xmlp, p=self.dropout, training=self.training)

        x = torch.cat((xmf, xmlp), dim=1)
        x = self.final(x)
        if sigmoid:
            x = torch.sigmoid(x)
        return x

    """def calculate_loss(self, predicted_labels, labels):
        criterion = nn.BCELoss()
        loss = criterion(predicted_labels, labels.view(-1, 1).float())
        #l2_reg = sum((param.norm(2) ** 2) for param in self.parameters())
        #loss += self.weight_decay * l2_reg
        return loss"""


    def replace_mlp_weights(self, mlp_model):
        # Replace MLP embeddings
        self.mlp_user_embed.weight.data = mlp_model.user_embedding.weight.data
        self.mlp_item_embed.weight.data = mlp_model.item_embedding.weight.data

        # Replace MLP weights
        for i, layer in enumerate(self.mlp):
            if isinstance(layer, nn.Linear):
                layer.weight.data = mlp_model.mlp[i].weight.data
                if layer.bias is not None:
                    init.zeros_(layer.bias.data)


    def replace_mf_weights(self, mf_model):
        self.mf_user_embed.weight.data = mf_model.user_emb.weight.data
        self.mf_item_embed.weight.data = mf_model.item_emb.weight.data

        #self.mf.weight.data = mf_model.output.weight.data
        self.mf.weight.data.copy_(mf_model.output.weight.data)


def calculate_loss(predicted_labels, labels):
    criterion = nn.BCELoss()
    loss = criterion(predicted_labels, labels.view(-1, 1).float())
    return loss






class DeepFM(nn.Module):
    #def __init__(self, num_users, num_items, all_movieIds, ratings, weight_decay=1e-3):
    def __init__(self, num_users, num_items,mlp_layer_sizes, dropout=0.1, weight_decay=1e-3):
        super(DeepFM,self).__init__()
        self.user_embedding = nn.Embedding(num_users, mlp_layer_sizes[0] // 2)
        self.item_embedding = nn.Embedding(num_items, mlp_layer_sizes[0] // 2)
        #self.fc1 = nn.Linear(in_features=32, out_features=16)
        #self.fc2 = nn.Linear(in_features=32, out_features=16)
        #self.output = nn.Linear(in_features=16, out_features=1)
        #self.all_movieIds = all_movieIds
        self.weight_decay = weight_decay
        """init.xavier_uniform_(self.user_embedding.weight)
        init.xavier_uniform_(self.item_embedding.weight)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.output.weight)"""
        self.mlp = nn.ModuleList()
        for i in range(1, len(mlp_layer_sizes)):
            self.mlp.extend([nn.Linear(mlp_layer_sizes[i - 1], mlp_layer_sizes[i])])
        self.dropout = dropout

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                glorot_uniform(layer)

        self.final= nn.Linear(mlp_layer_sizes[-1], 1)
        init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        #init.kaiming_uniform_(self.user_embedding.weight, a=0, mode='fan_in', nonlinearity='relu')
        #init.kaiming_uniform_(self.item_embedding.weight, a=0, mode='fan_in', nonlinearity='relu')
        #init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        #init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        #init.kaiming_uniform_(self.output.weight, a=0, mode='fan_in', nonlinearity='sigmoid')
        #self.ratings = ratings

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        xmlp = torch.cat((user_embedded, item_embedded), dim=1)

        for i, layer in enumerate(self.mlp):
            xmlp = layer(xmlp)
            xmlp = nn.functional.relu(xmlp)
            if self.dropout != 0:
                xmlp = nn.functional.dropout(xmlp, p=self.dropout, training=self.training)

        output= torch.sigmoid(self.final(xmlp))

        return output
        #vector = torch.cat([user_embedded, item_embedded], dim=-1)
        #vector = F.relu(self.fc1(vector))
        #vector = F.relu(self.fc2(vector))
        #pred = torch.sigmoid(self.output(vector))
        #return pred

    def calculate_loss(self, predicted_labels, labels):
        criterion = nn.BCELoss()
        loss = criterion(predicted_labels, labels.view(-1, 1).float())
        #l2_reg = sum((param.norm(2) ** 2) for param in self.parameters())
        #loss += self.weight_decay * l2_reg

        return loss

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = self.calculate_loss(predicted_labels, labels)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MovieLensTrainDataset(self.ratings, self.all_movieIds),
                          batch_size=128, num_workers=2)




class MF(pl.LightningModule):
    def __init__(self, num_users, num_items,mf_dim,weight_decay=1e-3):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim= mf_dim
        self.weight_decay = weight_decay
        #self.ratings = ratings

        self.user_emb = nn.Embedding(self.num_users, self.mf_dim)
        self.item_emb = nn.Embedding(self.num_items, self.mf_dim)
        self.output= nn.Linear(in_features=mf_dim, out_features=1)
        #init.xavier_uniform_(self.user_emb.weight)
        #init.xavier_uniform_(self.item_emb.weight)
        # Replace Xavier initialization with He initialization
        #init.kaiming_uniform_(self.user_emb.weight, a=0, mode='fan_in', nonlinearity='relu')
        #init.kaiming_uniform_(self.item_emb.weight, a=0, mode='fan_in', nonlinearity='relu')
        #init.kaiming_uniform_(self.output.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.normal_(self.user_emb.weight, mean=0, std=0.01)
        init.normal_(self.item_emb.weight, mean=0, std=0.01)
        init.normal_(self.output.weight, mean=0, std=0.01)


    def forward(self, user_input, item_input):
        user_embedded= self.user_emb(user_input)
        item_embedded= self.item_emb(item_input)
        mf_vector= user_embedded * item_embedded
        output=self.output(mf_vector)
        prediction= nn.Sigmoid()(output) # or torch.sigmoid(self.output(mf_vector))
        return prediction

    def calculate_loss(self, predicted_labels, labels):
        criterion = nn.BCELoss()
        loss = criterion(predicted_labels, labels.view(-1, 1).float())
        #L2 regularization for embeddings and hidden layers

        l2_reg = sum((param.norm(2) ** 2) for param in self.parameters())
        loss += self.weight_decay * l2_reg
        return loss



    """def training_step(self, batch, batch_idx):
      user_input, item_input, labels = batch
      predicted_labels = self(user_input, item_input)

      loss = self.calculate_loss(predicted_labels, labels)
      #self.log('train_loss', loss)  # Log the training loss
      return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MovieLensTrainDataset(self.ratings),
                          batch_size=128, num_workers=2)

"""



class MF_MLP_Model(pl.LightningModule):
    def __init__(self, num_users, num_items, ratings, all_movieIds, mf_dim, layers,reg_layers, reg_mf):
        super(MF_MLP_Model, self).__init__()
        assert len(layers) == len(reg_layers)
        num_layer = len(layers)  # Number of layers in the MLP
        self.ratings = ratings
        self.all_movieIds = all_movieIds
        self.reg_layers=reg_layers
        self.reg_mf=reg_mf

        # Embedding layers
        self.MF_Embedding_User = nn.Embedding(num_users, mf_dim)
        self.MF_Embedding_Item = nn.Embedding(num_items, mf_dim)

        self.MLP_Embedding_User = nn.Embedding(num_users, layers[0] // 2)
        self.MLP_Embedding_Item = nn.Embedding(num_items, layers[0] // 2)

        # MF part
        self.flatten = nn.Flatten()

        # MLP part
        self.mlp_vector = nn.Sequential(
            nn.Flatten(),
        )
        for idx in range(1, num_layer):
            layer = nn.Linear(layers[idx - 1], layers[idx])
            self.mlp_vector.add_module("layer%d" % idx, layer)
            self.mlp_vector.add_module("activation%d" % idx, nn.ReLU())

        # Final prediction layer
        self.prediction = nn.Linear(layers[-1] + mf_dim, 1)

    def get_regularization_loss(self, user_embed, item_embed, mlp_layers):
      # Calculate L2 regularization loss for MF embeddings
      mf_reg = (user_embed.norm(2) ** 2 + item_embed.norm(2) ** 2) * self.reg_mf

      # Calculate L2 regularization loss for MLP layers
      layers_reg = sum((layer.norm(2) ** 2) for layer in mlp_layers.parameters()) * sum(self.reg_layers)

      return mf_reg + layers_reg

    def forward(self, user_input, item_input):
        # MF embeddings
        mf_user_latent = self.flatten(self.MF_Embedding_User(user_input))
        mf_item_latent = self.flatten(self.MF_Embedding_Item(item_input))

        # Element-wise multiplication
        mf_vector = mf_user_latent * mf_item_latent

        # MLP embeddings
        mlp_user_latent = self.flatten(self.MLP_Embedding_User(user_input))
        mlp_item_latent = self.flatten(self.MLP_Embedding_Item(item_input))
        mlp_vector = self.mlp_vector(torch.cat([mlp_user_latent, mlp_item_latent], dim=1))

        # Concatenate MF and MLP parts
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=1)

        # Final prediction layer
        prediction = torch.sigmoid(self.prediction(predict_vector))

        return prediction

    def calculate_loss(self, predicted_labels, labels):
        # Calculate binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(predicted_labels, labels.view(-1, 1).float())

        # Calculate regularization loss
        regularization_loss = self.get_regularization_loss(
            self.MF_Embedding_User.weight, self.MF_Embedding_Item.weight, self.mlp_vector
        )

        return bce_loss + regularization_loss

    def training_step(self, batch, batch_idx):
      user_input, item_input, labels = batch
      predicted_labels = self(user_input, item_input)

      loss = self.calculate_loss(predicted_labels, labels)
      self.log('train_loss', loss)  # Log the training loss
      return loss

    """def validation_step(self, batch, batch_idx):
      user_input, item_input, labels = batch
      predicted_labels = self(user_input, item_input)
      loss = self.criterion(predicted_labels, labels.view(-1, 1).float())
      self.log('val_loss', loss)  # Log the validation loss
      return loss """
    def validation_step(self, batch, batch_idx):
      user_input, item_input, labels = batch
      predicted_labels = self(user_input, item_input)

      # Calculate binary cross-entropy loss without regularization
      bce_loss = F.binary_cross_entropy(predicted_labels, labels.view(-1, 1).float())

      # Calculate evaluation metrics
      accuracy = accuracy_score(labels.cpu(), predicted_labels.cpu().round())
      precision = precision_score(labels.cpu(), predicted_labels.cpu().round())
      recall = recall_score(labels.cpu(), predicted_labels.cpu().round())
      f1 = f1_score(labels.cpu(), predicted_labels.cpu().round())

      # Log the validation metrics
      self.log('val_loss', bce_loss, on_step=False, on_epoch=True)
      self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)
      self.log('val_precision', precision, on_step=False, on_epoch=True)
      self.log('val_recall', recall, on_step=False, on_epoch=True)
      self.log('val_f1', f1, on_step=False, on_epoch=True)

      return bce_loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())



    def train_dataloader(self):
        return DataLoader(MovieLensTrainDataset(self.ratings),
                          batch_size=64, num_workers=2)




class ImplicitRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=8, hidden_dim=16):
        super(ImplicitRecommender, self).__init__()  # Add parentheses to super

        # User and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        interaction = torch.cat((user_embeds, item_embeds), dim=1)

        # Pass through the MLP layers
        output = self.mlp(interaction)

        return torch.sigmoid(output)













#https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/4
#https://arxiv.org/pdf/1708.05031.pdf
#https://arxiv.org/abs/1703.04247
#

