import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import defaultdict
from preprocess import preprocess
import os

def create_data_loader(data, test_neg_num=4,batch_size=256,save_train_path='train', save_data_path='data', save_test_path='test', save_train_data_path='train_data',save_test_data_path='test_data'):
    """
    Prepares training and testing dataloaders for BPR-based models using implicit feedback data.

    Args:
        data (pd.DataFrame): Raw user-item interaction data with at least userId, movieId, timestamp, rating.
        test_neg_num (int): Number of negative samples per test item.
        batch_size (int): Batch size for PyTorch DataLoader.
        save_train_path (str): Path to save train interactions (full).
        save_data_path (str): Path to save preprocessed data (full).
        save_test_path (str): Path to save test interactions (full).
        save_train_data_path (str): Path to save train triplets (user, pos, neg).
        save_test_data_path (str): Path to save test triplets (user, pos, neg).

    Returns:
        Tuple of train and test DataLoaders, train DataFrame, test DataFrame, and list of all item IDs.
    """

    """num_items = data.groupby('userId').size()
    data = data[data['userId'].isin(num_items[num_items >filter_user].index)]
    item_supports = data.groupby('movieId').size()
    data = data[data['movieId'].isin(item_supports[item_supports > filter_item].index)]
    self.data= self.data.loc[self.data['rating']>=3,'rating']=1
    self.data= self.data.loc[self.data['rating']<3,'rating']=0
    data = data[data.rating>=thereshold]"""
    if save_data_path and os.path.exists(save_data_path):
        data=pd.read_csv(save_data_path)
    else:

        data = preprocess(data)
        data = data.sort_values(by=['userId', 'timestamp']) # Sort for temporal order
        if save_data_path:
            data.to_csv(save_data_path, index=False)

    all_movieIDs = data['movieId'].unique()

    print("****** NEGATIVE AND POSITIVE SAMPLING IS STARTING *****")

    #user_item_dict = data.groupby('userId')['movieId'].apply(list).to_dict()
    item_id_max = data['movieId'].unique()

    if save_train_data_path and os.path.exists(save_train_data_path):
        train_data=pd.read_csv(save_train_data_path)
        test_data=pd.read_csv(save_test_data_path)
        train = pd.read_csv(save_train_path)
        test = pd.read_csv(save_test_path)

    else:
        train_data, test_data = defaultdict(list), defaultdict(list)
        test, train = pd.DataFrame(), pd.DataFrame()
        for user_id, df in tqdm(data[['userId', 'movieId']].groupby('userId')):
          pos_list = df['movieId'].tolist()

          def gen_neg():
              neg = pos_list[0]
              while neg in pos_list:
                  neg = np.random.choice(item_id_max)
              return neg

          # Generate enough negative samples for this user
          neg_list = [gen_neg() for i in range(len(pos_list) + test_neg_num-1)]
          # 0.8 for train, 0.2 for test
          test_range =len(pos_list) *0.2  # Last 20% interactions are used for test

          # Iterate through the interaction sequence
          for i in range(1, len(pos_list)):
            if i >= (len(pos_list) - test_range):
              # Assign interactions to test set
              j=0
              while(j<len(neg_list[i:])):

                test_data['userId'].append(user_id)
                test_data['pos_id'].append(pos_list[i])
                test_data['neg_id'].append((neg_list[i+j]))
                j+=1

              # append to test
              row =data.loc[(data['userId']==user_id) & (data['movieId']==pos_list[i])]
              test= pd.concat([test, row])

            #elif i == len(pos_list) - 2:
                #val_data['userId'].append(user_id)
                #val_data['pos_id'].append(pos_list[i])
                #val_data['neg_id'].append(neg_list[i])
            else:
                # Assign interactions to training set
                train_data['userId'].append(user_id)
                train_data['pos_id'].append(pos_list[i])
                train_data['neg_id'].append(neg_list[i])
                row= data.loc[(data['userId']==user_id) & (data['movieId']==pos_list[i])]
                #print(f"row:{row}")
                train= pd.concat([train, row])


        if save_train_path and save_test_path:
            train.to_csv(save_train_path, index=False)  # Save train data to CSV file
            test.to_csv(save_test_path, index=False)
        if save_train_data_path and not os.path.exists(save_train_data_path):
            train_data_df= pd.DataFrame(train_data)
            train_data_df.to_csv(save_train_data_path, index=False)
            test_data_df=pd.DataFrame(test_data)
            test_data_df.to_csv(save_test_data_path, index=False)

    # Convert data to PyTorch tensors
    train_tensor = TensorDataset(torch.tensor(train_data['userId']),
                                  torch.tensor(train_data['pos_id']),
                                  torch.tensor(train_data['neg_id']))
    """val_tensor = TensorDataset(torch.tensor(val_data['userId']),
                                torch.tensor(val_data['pos_id']),
                                torch.tensor(val_data['neg_id']))"""
    test_tensor = TensorDataset(torch.tensor(test_data['userId']),
                                torch.tensor(test_data['pos_id']),
                                torch.tensor(test_data['neg_id']))


    return DataLoader(train_tensor, batch_size=batch_size, shuffle=True), DataLoader(test_tensor, batch_size=batch_size, shuffle=True), train, test,all_movieIDs
