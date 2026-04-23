import os
import pandas as pd
from scipy import sparse
import numpy as np
from DP_code import apply_dp
import calibration_Func
from torch.utils.data import Dataset
import torch

class DataLoader_classic():
    '''
    Load Movielens-20m dataset
    '''
    def __init__(self, path,show2id,index_to_item_global,profile2id,item_mapping):
        self.pro_dir = os.path.join(path, 'pro_sg')
        print("address",self.pro_dir)
        assert os.path.exists(self.pro_dir), "Preprocessed files does not exist. Run data.py"
        self.train_users = pd.read_csv(os.path.join(self.pro_dir, 'train.csv'))['userId'].unique()
        self.val_users = pd.read_csv(os.path.join(self.pro_dir, 'validation_tr.csv'))['userId'].unique()
        self.test_users = pd.read_csv(os.path.join(self.pro_dir, 'test_tr.csv'))['userId'].unique()
        # Combine users across all datasets (train, val, test)
        #raw_data= pd.read_csv(os.path.join(self.pro_dir, 'raw_data.csv'))
        self.item_mapping=item_mapping
        self.profile2id = profile2id
        self.n_items = len(show2id)
        self.show2id = show2id
        self.item_mapping = item_mapping
        self.index_to_item_global=index_to_item_global

    def load_data(self ,datatype='train'):
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")
        
    def load_n_items(self):
        unique_sid = list()
        with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items
    
    def _load_train_data(self):
        path = os.path.join(self.pro_dir, 'train.csv')
        tp = pd.read_csv(path)
        tp = numerize(tp, self.profile2id, self.show2id)

        n_users = tp['uid'].max() + 1
        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, self.n_items))
        return data
    
    def _load_tr_te_data(self, datatype='test',debug=False):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))
        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)
        #pure_tp_tr=tp_tr.copy()
        #popular_id = evaluation.popularity_id(tp_tr, self.item_mapping)
        #interacted_distr={}
        grouped_tp_tr = tp_tr.groupby('userId')
        # Iterate over grouped users
        #print("profile2id",self.profile2id)
        interacted_distr={}
        if debug:
            for user_id, group in grouped_tp_tr:
                # Ensure that user_id is in the same format as profile2id keys (integer)
                user_id = int(user_id)  # Convert to int if necessary
                # Check if this user exists in profile2id
                if user_id in self.profile2id:
                    interacted = group['movieId'].tolist()
                    interacted = [item_id for item_id in interacted if item_id in self.item_mapping]
                    real_interacted_items = [self.item_mapping[item_id] for item_id in interacted]
                    # Compute genre distribution and assign to interacted_distr
                    interacted_distr[user_id] = calibration_Func.compute_genre_distr(real_interacted_items)
                else:
                    print(f"User ID {user_id} not found in profile2id")

            print("Sample user IDs in tp_tr:", tp_tr['userId'].unique()[:10])
            print("Sample keys in interacted_distr:", list(interacted_distr.keys())[:10])
            missing_in_profile2id = [uid for uid in tp_tr['userId'].unique() if uid not in self.profile2id]
            if missing_in_profile2id:
                print(f"User IDs in tp_tr but missing in profile2id: {missing_in_profile2id}")
            else:
                print("All user IDs in tp_tr are present in profile2id")

            missing_in_interacted_distr = [uid for uid in tp_tr['userId'].unique() if uid not in interacted_distr]
            if missing_in_interacted_distr:
                print(f"User IDs in tp_tr but missing in interacted_distr: {missing_in_interacted_distr}")
            else:
                print("All user IDs in tp_tr are present in interacted_distr")
        # Generally all tp_tr are represented in profile2id
        # And all tp_tr are represented in intercated_dist
        # But there are items in profile2id which are not represented in interacted_distr
        tp_tr = numerize(tp_tr, self.profile2id, self.show2id)
        tp_te = numerize(tp_te, self.profile2id, self.show2id)
        grouped_tp_tr = tp_tr.groupby('uid')  # Group by numerized user ID ('uid')
        # grouped_tp_tr['uid']
        users=[]
        for user_id, group in grouped_tp_tr:
            # Ensure that user_id is now numerized (as 'uid')
            users.append(user_id)
            interacted = group['sid'].tolist()  # Use 'sid' for the numerized item IDs
            real_items=[self.index_to_item_global[item] for item in interacted if item in self.index_to_item_global]
            interacted = [item_id for item_id in real_items if item_id in self.item_mapping]
            real_interacted_items = [self.item_mapping[item_id] for item_id in interacted]
            # Populate interacted_distr with numerized user_id (as 'uid')
            interacted_distr[user_id] = calibration_Func.compute_genre_distr(real_interacted_items)

        #print("users",users)
        #print("users in tp_tr",tp_tr['uid'].unique())
        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                    (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                    (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))

        #print("Shape of data_tr:", data_tr.shape)
        unique_uids_in_tp_tr = tp_tr['uid'].nunique()
        #print("Number of unique uids in tp_tr:", unique_uids_in_tp_tr)
        assert data_tr.shape[
                   0] == unique_uids_in_tp_tr, "Mismatch in number of rows in data_tr and unique users in tp_tr"
        return tp_tr,tp_te,data_tr, data_te, interacted_distr#,popular_id

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, min_uc=5, min_sc=5):
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount['size'] > min_sc])]
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount['size'] > min_uc])]
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount

def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()
    np.random.seed(98765)
    for _, group in data_grouped_by_user: # for each user
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True  # pick test_prop * n_items_u of items for user u
            tr_list.append(group[np.logical_not(idx)])  # tr and te are for same user but with different items
            te_list.append(group[idx])
        else:
            tr_list.append(group)
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    return data_tr, data_te

def numerize(tp, profile2id, show2id):
    uid = tp['userId'].apply(lambda x: profile2id[x])
    sid = tp['movieId'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])



class InteractionDataset(Dataset):
    def __init__(self, sparse_data):
        """
        sparse_data: scipy.sparse.csr_matrix representing user-item interactions
        """
        self.sparse_data = sparse_data

    def __len__(self):
        # The number of users is the number of rows in the sparse matrix
        return self.sparse_data.shape[0]

    def __getitem__(self, index):
        # Convert each row (user's interaction) to a dense tensor
        user_interactions = self.sparse_data[index].toarray().squeeze()
        return torch.FloatTensor(user_interactions)

def run_data(args):
    print("***running data preparation***")
    item_col = 'movieId'
    print("Load and Preprocess Movielens-20m dataset")
    # Load Data
    #DATA_DIR = 'ml-20m/'
    names = ['userId', 'movieId', 'rating', 'timestamp']
    file_path = 'ratings.dat'
    raw_data = pd.read_table(file_path, names=names, sep="::", engine='python')
    #print("total size:", raw_data.shape)

    #raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)
    raw_data = raw_data[raw_data['rating'] >= 3]

    # Filter Data
    raw_data, user_activity, item_popularity = filter_triplets(raw_data)
    print("raw data movies ",raw_data['movieId'].nunique())
    # Shuffle User Indices
    #unique_uid = user_activity.index
    unique_uid = user_activity['userId'].values
    unique_item=item_popularity['movieId'].values
    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]
    n_users = unique_uid.size
    n_items=unique_item.size
    print("number of total users : ",n_users) #6031
    print("number of total items : ",n_items)
    n_heldout_users = 603 # why? # 20% of 6031 is 1206 which half of that will be 603

    # Split Train/Validation/Test User Indices
    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]
    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]   # it could be greater than tr_users because there it is just unique, here is all
    #print(len(train_plays['userId'].unique()))
    unique_sid = pd.unique(train_plays['movieId'])
    if args.DP=="True":
        print("***Prepare data in DP mode***")
        interacted_items=train_plays.groupby('userId')['movieId'].apply(list).to_dict()
        for u in interacted_items:
            positive_items = interacted_items[u]
            new_positive_samples = apply_dp(positive_items, unique_sid, 0.1)
            interacted_items[u] = new_positive_samples
        new_rows=[]
        for user,movies in interacted_items.items():
            for item in movies:
                new_rows.append({"userId": user, 'movieId': item})
        updated_train_plays=pd.DataFrame(new_rows)
        train_plays=updated_train_plays.copy()
    pro_dir = os.path.join('pro_sg')
    if not os.path.exists(pro_dir):
        print("making direction")
        os.makedirs(pro_dir)
    else:
        print("pro_sg already exist")

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    with open(os.path.join(pro_dir, 'unique_uid.txt'), 'w') as f:
        for uid in unique_uid:
            f.write('%s\n' % uid)

    # check if all user IDS inraw data and unique_ids are identical
    raw_data_user_ids = set(raw_data['userId'].unique())
    # Step 2: Convert unique_uid to a set if it's not already
    unique_uid_set = set(unique_uid)
    # Step 3: Check if both sets are identical
    """if raw_data_user_ids == unique_uid_set:
        print("All unique_ids and user_ids in raw_data are identical!")
        print(f"raw_data_user_ids:{raw_data_user_ids}")
        print(f"unique_uid_set:{unique_uid_set}")

    else:
        # If they're not identical, find the differences
        missing_in_unique_uid = raw_data_user_ids - unique_uid_set
        missing_in_raw_data = unique_uid_set - raw_data_user_ids

        print(f"IDs in raw_data but not in unique_uid: {missing_in_unique_uid}")
        print(f"IDs in unique_uid but not in raw_data: {missing_in_raw_data}")"""
    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]
    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)
    test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]
    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

    raw_data.to_csv(os.path.join(pro_dir, 'raw_data.csv'), index=False)
    #train_data = numerize(train_plays, profile2id, show2id)
    #train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
    train_plays[item_col] = train_plays[item_col].astype(int)
    train_plays.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
    #vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    #vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
    vad_plays_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
    #vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    #vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
    vad_plays_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
    #test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    #test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
    test_plays_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
   # test_data_te = numerize(test_plays_te, profile2id, show2id)
    #test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
    test_plays_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

    print("Done!")
