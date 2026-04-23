
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






from torch.utils.data import DataLoader, TensorDataset
def creat_data(df,all_movieIds,batch_size, num_negatives=4):

    user_col = 'user_idx'
    item_col = 'business_idx'
    rating = 'stars'
    data = {
    'userId': [],
    'pos_id': [],
    'neg_id': []
    }

    positives =df.groupby(user_col)[item_col].apply(list).to_dict()
    for user in positives.keys():
        for item in positives[user]:
            def generate_negative():
                neg = item
                while neg in positives[user]:
                    neg = np.random.choice(all_movieIds)
                return neg

            neg_samples= [generate_negative() for i in range(num_negatives)]
            for i in range(num_negatives):
                data['userId'].append(user)
                data['pos_id'].append(item)
                data['neg_id'].append(neg_samples[i])

    data_tensor= TensorDataset(torch.tensor(data['userId']), torch.tensor(data['pos_id']), torch.tensor(data['neg_id']))

    return DataLoader(data_tensor, batch_size, shuffle=True)







def create_data_loader(data, test_neg_num=20,batch_size=128):

    user_col = 'user_idx'
    item_col = 'business_idx'
    value_col = 'stars'
    timestamp='date'
    #num_items = data.groupby('userId').size()
    #data = data[data['userId'].isin(num_items[num_items >filter_user].index)]
    #item_supports = data.groupby('movieId').size()
    #data = data[data['movieId'].isin(item_supports[item_supports > filter_item].index)]
    #self.data= self.data.loc[self.data['rating']>=3,'rating']=1
    #self.data= self.data.loc[self.data['rating']<3,'rating']=0
    #data = data[data.rating>=thereshold]
    print("starting")
    data = Preprocess.preprocess(data)
    print("End of preprocessing")
    data = data.sort_values(by=[user_col, timestamp])


    all_movieIDs = data[item_col].unique()
    print("****** NEGATIVE AND POSITIVE SAMPLING IS STARTING *****")

    #user_item_dict = data.groupby('userId')['movieId'].apply(list).to_dict()
    item_id_max = data[item_col].unique()
    train_data, test_data = defaultdict(list), defaultdict(list)

    test, train = pd.DataFrame(), pd.DataFrame()
    for user_id, df in tqdm(data[[user_col, item_col]].groupby(user_col)):
      pos_list = df[item_col].tolist()

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
          row =data.loc[(data[user_col]==user_id) & (data[item_col]==pos_list[i])]
          test= pd.concat([test, row])



        #elif i == len(pos_list) - 2:
            #val_data['userId'].append(user_id)
            #val_data['pos_id'].append(pos_list[i])
            #val_data['neg_id'].append(neg_list[i])
        else:
            train_data['userId'].append(user_id)
            train_data['pos_id'].append(pos_list[i])
            train_data['neg_id'].append(neg_list[i])
            row= data.loc[(data[user_col]==user_id) & (data[item_col]==pos_list[i])]
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


