# create train test split
from Preprocess import preprocess
import  pandas as pd
from sklearn.utils import shuffle
import torch
#u.data
# create train test split
#First approach
# Each user that is in the test should be in the train too
from sklearn.model_selection import train_test_split
import numpy as np

"""def train_test_split_version1(ratings):
  user_col = 'userId'
  item_col = 'movieId'
  value_col = 'rating'
  time_col = 'timestamp'
  # Split the data into training and test sets
  ratings = preprocess(ratings)
  train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

  # Identify users in the test set that are missing in the training set
  missing_users = set(test_data[user_col]) - set(train_data[user_col])

  # Adjust the split by adding interactions for missing items to the training set
  lst=[]
  for user_id in missing_users:

    sampled_rows =(test_data[test_data[user_col]==user_id].sample(frac=0.2))
    lst.append(sampled_rows)

    test_data=test_data[~test_data.index.isin(sampled_rows.index)]

    # Concatenate the list of DataFrames into a single DataFrame
  df_extended = pd.concat(lst)
  train_data = pd.concat([train_data, df_extended])

  return train_data, test_data"""




# create train test split
#First approach
# Each user that is in the test should be in the train too


#change


# create train test split
#First approach
# Each user that is in the test should be in the train too


def train_test_split_version1(ratings):
    """
    Split the data into train and test sets in a way that for each sample in the test set we have a history in the train set (this is essential for calculating KLD and also for making a calibrated list)
    """
    user_col = 'userId'
    ratings = shuffle(ratings, random_state=42)

    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

    missing_users = set(test_data[user_col]) - set(train_data[user_col])
    if len(missing_users)>1:
      sampled_rows_list=[]
      for user_id in missing_users:
          sampled_rows = test_data[test_data[user_col] == user_id].sample(frac=0.2)
          sampled_rows_list.append(sampled_rows)

      train_data = pd.concat([train_data] + sampled_rows_list)
      test_data = test_data[~test_data.index.isin(pd.concat(sampled_rows_list).index)]

    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)


# Split Train & Test function

def read_data_ml100k():
  file_path = 'u.data'
  names = ['userId', 'movieId', 'rating', 'timestamp']
  data = pd.read_csv(file_path, sep='\t', names=names)
  num_users = data.userId.unique().shape[0]
  num_items = data.movieId.unique().shape[0]
  #data['userId'] = data['userId'] - 1
  #data['movieId'] = data['movieId'] - 1
  return data, num_users, num_items



def read_1M():
  names = ['userId', 'movieId', 'rating', 'timestamp']
  #data = pd.read_csv("rating.csv", parse_dates=['timestamp'])
  np.random.seed(123)
  ratings = pd.read_csv("rating.csv", parse_dates=['timestamp'])
  #num_users = ratings.userId.unique().shape[0]
  #num_items = ratings.movieId.unique().shape[0]
  num_users, num_items = ratings['userId'].max() + 1, ratings['movieId'].max() + 1
  # just use 30 percent of data
  rand_userIds = np.random.choice(ratings['userId'].unique(),
                                  size=int(len(ratings['userId'].unique()) * 0.2),
                                  replace=False)
  data = ratings.loc[ratings['userId'].isin(rand_userIds)]
  print('data dimension: \n', data.shape)

  #data['userId'] = data['userId'] - 1
  #data['movieId'] = data['movieId'] - 1
  return data,num_users,num_items




def split_data_ml100k(data, num_users, num_items, split_mode='seq-aware', test_ratio=0.2):
  column_names = ['userId', 'movieId', 'rating', 'timestamp']  # Desired column names

  if split_mode == 'seq-aware':
    train_items, test_items, train_list = {}, {}, []
    for line in data.itertuples():
      u, i, rating, time = line[1], line[2], line[3], line[4]
      train_items.setdefault(u, []).append((u, i, rating, time))
      if u not in test_items or test_items[u][-1] < time:
        test_items[u] = (i, rating, time)
    for u in test_items:
      train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
    test_data = [(key, *value) for key, value in test_items.items()]
    train_data = [item for item in train_list if item not in test_data]
    train_data = pd.DataFrame(train_data,columns=column_names)
    test_data = pd.DataFrame(test_data,columns=column_names)
  else:
    mask = [True if x == 1 else False for x in np.random.uniform(
      0, 1, (len(data))) < 1 - test_ratio]
    neg_mask = [not x for x in mask]
    train_data, test_data = data[mask], data[neg_mask]
  return train_data, test_data


# Load Data for model training function
def load_data_ml100k(data, num_users, num_items, feedback='implicit'):

  users, items, scores = [], [], []
  inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
  for line in data.itertuples():
    user_index, item_index = line[1], line[2]

    score = float(line[3])
    users.append(user_index)
    items.append(item_index)
    scores.append(score)
    if feedback == 'implicit':
      inter.setdefault(user_index, []).append(item_index)
    else:
      inter[item_index, user_index] = score
  return users, items, scores, inter


# Read & Split & Load & Get on Torch DataLoader function
def split_and_load_ml100k(split_mode='seq-aware', feedback='implicit',
                          test_ratio=0.1, batch_size=256):
  # read data
  data, num_users, num_items = read_1M()
  #all_movieIds = data['movieId'].unique()
  # split data
  if feedback=='implicit':
    data = preprocess(data)
  all_movieIds = data['movieId'].unique()
  print("***** data preprocess has finished ****")
  train_data_main, test_data_main = split_data_ml100k(
    data, num_users, num_items, split_mode, test_ratio)

  # load data with proper form
  train_u, train_i, train_r, _ = load_data_ml100k(
    train_data_main, num_users, num_items, feedback)
  test_u, test_i, test_r, _ = load_data_ml100k(
    test_data_main, num_users, num_items, feedback)


  # Get on TensorDataset
  train_set = torch.utils.data.TensorDataset(
    torch.tensor(train_u), torch.tensor(train_i), torch.tensor(train_r))
  test_set = torch.utils.data.TensorDataset(
    torch.tensor(test_u), torch.tensor(test_i), torch.tensor(test_r))
  # Get on DataLoader
  train_iter = torch.utils.data.DataLoader(
    train_set, shuffle=True, batch_size=batch_size)
  test_iter = torch.utils.data.DataLoader(
    test_set, shuffle=True, batch_size=batch_size)

  return num_users, num_items,all_movieIds, train_data_main,test_data_main, train_iter, test_iter



def one_leave_out():
  # warning !!!! This code needs an item mapping
  df,num_users,num_items= read_data_ml100k()
  df = preprocess(df)
  #all_movieIds = df['movieId'].unique()
  df['userId'] = pd.factorize(df['userId'])[0]
  df['movieId'] = pd.factorize(df['movieId'])[0]
  all_movieIds = df['movieId'].unique()

  user_cardinality = df['userId'].max() + 1
  item_cardinality = df['movieId'].max() + 1
  df.sort_values(by='timestamp', inplace=True)
  del df['rating'], df['timestamp']
  grouped_sorted = df.groupby('userId', group_keys=False)
  test_data = grouped_sorted.tail(2).sort_values(by='userId')
  # Train set is all interactions but the last one
  train_data = grouped_sorted.apply(lambda x: x.iloc[:-2])


  return train_data, test_data,user_cardinality,item_cardinality,all_movieIds



