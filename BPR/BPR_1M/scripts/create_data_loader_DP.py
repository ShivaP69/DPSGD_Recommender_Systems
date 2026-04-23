import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import defaultdict
from preprocess import preprocess
from DP_Code import apply_dp
import os

def create_data_loader(data, privacy_budget,test_neg_num=4,batch_size=256,save_train_path='train', save_data_path='data', save_test_path='test', save_train_data_path='train_data',save_test_data_path='test_data'):

    if save_data_path and os.path.exists(save_data_path):
        data=pd.read_csv(save_data_path)
    else:
        data = preprocess(data)
        data = data.sort_values(by=['userId', 'timestamp'])
        if save_data_path:
            data.to_csv(save_data_path, index=False)

    all_movieIDs = data['movieId'].unique()

    print("****** NEGATIVE AND POSITIVE SAMPLING IS STARTING *****")

    item_id_max = data['movieId'].unique()
    train_data, test_data = defaultdict(list), defaultdict(list)
    test, train = pd.DataFrame(), pd.DataFrame()

    existing_combinations = set(zip(data['userId'], data['movieId'])) # all unique user and movie combinations : eg. (user1,item1), (user1,item4), ...

    for user_id, df in tqdm(data[['userId', 'movieId']].groupby('userId')):
      pos_list = df['movieId'].tolist() # all positive items for specific user (user_id)
      pos_list= apply_dp(pos_list,all_movieIDs,privacy_budget) # Apply LDP (Adding a new positive set (DP-positive set))
      new_rows = []

      # the main purpose of this loop: finding and adding new combinations of (user, item) after applying LDP
      for item in pos_list:
          new_data={'userId':user_id,'movieId':item,'rating':1} # rating is 1 (it is a positive interaction)
          new_combination = (new_data['userId'], new_data['movieId']) # eg. (user1,item5)
          if new_combination not in existing_combinations:
              new_rows.append(new_data)
              existing_combinations.add(new_combination)  # Add the new combination to the already existing one

      new_data_df = pd.DataFrame(new_rows)
      if not new_data_df.empty:
          data = pd.concat([data, new_data_df], ignore_index=True) # adding new positive rows to the data

      def gen_neg():
          neg = pos_list[0]
          while neg in pos_list:
              neg = np.random.choice(item_id_max)
          return neg

      neg_list = [gen_neg() for i in range(len(pos_list) + test_neg_num-1)]
      # 0.8 for train, 0.2 for test
      test_range =len(pos_list) *0.2
      for i in range(1, len(pos_list)):

        if i >= (len(pos_list) - test_range):
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
            train_data['userId'].append(user_id)
            train_data['pos_id'].append(pos_list[i])
            train_data['neg_id'].append(neg_list[i])
            row= data.loc[(data['userId']==user_id) & (data['movieId']==pos_list[i])]
            #print(f"row:{row}")
            train= pd.concat([train, row])

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

