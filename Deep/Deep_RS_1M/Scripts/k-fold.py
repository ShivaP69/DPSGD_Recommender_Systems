# This is a sample Python script.


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from sklearn.metrics import recall_score
from ansible.plugins.loader import test_loader
import numpy as np
from MF_MLP import MF_MLP_Model
from MF_MLP import NeuMF
from DataProcess import MovieLensTrainDataset
from train_test_split import one_leave_out
import torch
import torch.nn as nn
import opacus
from opacus import PrivacyEngine
import pytorch_lightning as pl
from Preprocess import preprocess
from train_test_split import train_test_split_version1
from train_test_split import read_data_ml100k
from train_test_split import split_and_load_ml100k
from torch.utils.data import DataLoader, RandomSampler
from HitRate import hit_rate
from evaluation import evaluation_metrics
from ItemMapping import create_item_mapping
import json
from csv import writer
import calibration_Func
import evaluation
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import os
import matplotlib.pyplot as plt
from MF_MLP import ImplicitRecommender
from Recall import calculate_total_average_recall
# Press the green button in the gutter to run the script.
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == '__main__':
    # 100 k
    # Load configuration from the JSON file
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    num_epochs = config.get('num_epochs')
    noise_multiplier = config.get('noise_multiplier')
    max_grad_norm = config.get('max_grad_norm')
    delta = config.get('delta')
    top_k=config.get('k')
    calibration=config.get('calibration')
    learning_rate = config.get('learning_rate')

    """header = ['dataset', 'num_epochs', 'max_grad_norm', 'noise_multiplier', 'ndcg_average', 'average_recall', 'epsilon',
              'delta', 'Train_loss', 'Test_loss']"""

    #read data
    file_path = 'u.data'
    """np.random.seed(123)
    df = pd.read_csv("rating.csv", parse_dates=['timestamp'])
    print('data dimension: \n', df.shape)
    print(df.head)
    dataset=1M"""
    """ratings=pd.read_csv("rating.csv", parse_dates=['timestamp'])

    # just use 30 percent of data
    rand_userIds = np.random.choice(ratings['userId'].unique(),
                                size=int(len(ratings['userId'].unique())*0.3),
                                replace=False)
    df = ratings.loc[ratings['userId'].isin(rand_userIds)]"""


    average_recalls = []
    KLD_list = []
    privacy_epsilons = []

    test_ndcg=[]
    hit=[]
    relevance_metric=[]
    KLD_list_train=[]
    average_recalls_train=[]
    ndcg_train=[]

    """names = ['userId', 'movieId', 'rating', 'timestamp']
    df = pd.read_csv(file_path, sep='\t', names=names)"""
    #print('data dimension: \n', df.shape)
    #dataset="100k"
    #print(df.head())

    title_col = 'title'
    genre_col = 'genres'
    item_col = 'movieId'

    df_item = pd.read_csv('movie.csv')
    df_item = df_item[
        df_item[genre_col] != '(no genres listed)']  # eliminate movies that had no genre information attached
    print('dimension: ', df_item.shape)

    df,_,_= read_data_ml100k()

    #num_users = df['userId'].max() + 1
    #num_items = df['movieId'].max() + 1
    df = preprocess(df)
    #all_movieIds = df['movieId'].unique()

    df['userId'] = pd.factorize(df['userId'])[0]
    df['movieId'] = pd.factorize(df['movieId'])[0]
    all_movieIds = df['movieId'].unique()

    num_users = df['userId'].max() + 1
    num_items = df['movieId'].max() + 1

    n_splits = 5
    kf = GroupKFold(n_splits=n_splits)

    # Store performance metrics for each fold
    fold_results_test = []
    fold_results_train = []
    for train_index, test_index in kf.split(df, groups=df['userId']):
        train_data, test_data = df.iloc[train_index], df.iloc[test_index]
        train_dataset = MovieLensTrainDataset(train_data, all_movieIds)
        train_dataloader = DataLoader(train_dataset, batch_size=64)
        test_dataset = MovieLensTrainDataset(test_data, all_movieIds)
        val_dataloader = DataLoader(test_dataset, batch_size=64)
        #model = NeuMF(num_users, num_items, mf_dim=2, mlp_layer_sizes=[32, 16], dropout=0)
        model = NeuMF(num_users, num_items, mf_dim=5, mlp_layer_sizes=[32, 16], dropout=0)
        """model = MF_MLP_Model(num_users, num_items, train_data, all_movieIds, mf_dim=5, layers=[16, 8],
                             reg_mf=0.01,
                             reg_layers=[0.02, 0.02])"""
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        test_losses = []
        train_losses=[]
        # Define your loss function (Binary Cross-Entropy)
        criterion = nn.BCELoss()

        for epoch in range(num_epochs):
            model.train()  # Set the model in training mode
            total_loss_train = []

            for batch in train_dataloader:
                user_input, item_input, labels = batch

                optimizer.zero_grad()  # Zero the gradients
                predicted_labels = model(user_input, item_input)

                loss = criterion(predicted_labels, labels.view(-1, 1).float())
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the model parameters

                total_loss_train.append(loss.item())
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {np.mean(total_loss_train):}')

            # Validation
            model.eval()
            all_predictions = []
            all_labels = []
            total_loss_test = []
            with torch.no_grad():
                for batch in val_dataloader:
                    user_input, item_input, labels = batch
                    predicted_labels = model(user_input, item_input).squeeze()
                    all_predictions.extend(predicted_labels.tolist())
                    all_labels.extend(labels.tolist())
                    loss = nn.BCELoss()(predicted_labels, labels.float())

                    total_loss_test.append(loss.item())
            print(f'Loss test: {np.mean(total_loss_test):.4f}')

            accuracy = accuracy_score(all_labels, [1 if p >= 0.5 else 0 for p in all_predictions])
            # auc = roc_auc_score(all_labels, all_predictions)

            # print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {np.mean(total_loss_test):.4f}')
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}')
            # print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}, AUC: {auc:.4f}')
            test_losses.append(np.mean(total_loss_test))
            train_losses.append(np.mean(total_loss_train))

        fold_results_test.append(np.mean(test_losses))
        fold_results_train.append(np.mean(train_losses))

        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        axs.plot(range(1, num_epochs + 1), test_losses, label='test-Deep_RS', marker='o')
        axs.plot(range(1, num_epochs + 1), train_losses, label='train-Deep_RS', marker='x')
        # axs.plot(range(1, num_epochs + 1), KLD_cal, label='calibarted-Deep-RS', marker='+')
        axs.set_xlabel('Epochs')
        axs.set_ylabel('Loss')
        # axs.set_title('Comparison of Model Performance with two number of different layers Based on Relevance')
        axs.legend()
        axs.grid(True)

        plt.show()


    print(fold_results_test)







