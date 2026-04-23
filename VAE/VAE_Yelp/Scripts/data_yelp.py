import os
import pandas as pd
from scipy import sparse
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
import ast
from DP_code import apply_dp
import calibration_Func
from torch.utils.data import Dataset
import torch
from collections import Counter
user_col = 'user_idx'
item_col = 'business_idx'
value_col = 'stars'
timestamp = 'data'
class DataLoader_classic():
    '''
    Load Movielens-20m dataset
    '''

    def __init__(self,  path,show2id,index_to_item_global,profile2id,item_mapping):
        self.pro_dir = os.path.join(path, 'pro_sg')
        assert os.path.exists(self.pro_dir), "Preprocessed files does not exist. Run data.py"
        self.train_users = pd.read_csv(os.path.join(self.pro_dir, 'train.csv'))['user_idx'].unique()
        self.val_users = pd.read_csv(os.path.join(self.pro_dir, 'validation_tr.csv'))['user_idx'].unique()
        self.test_users = pd.read_csv(os.path.join(self.pro_dir, 'test_tr.csv'))['user_idx'].unique()
        # Combine users across all datasets (train, val, test)

        self.profile2id = profile2id
        self.n_items = len(show2id)
        self.show2id = show2id
        self.item_mapping = item_mapping
        self.index_to_item_global = index_to_item_global
        self.item_mapping = item_mapping


    def load_data(self, datatype='train'):
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
        tp=tp.dropna(subset=[item_col])
        tp = numerize(tp, self.profile2id, self.show2id)

        n_users = tp['uid'].max() + 1
        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float64',
                                 shape=(n_users, self.n_items))
        return data

    def _load_tr_te_data(self, datatype='test',debug=True):
        user_col = 'user_idx'
        item_col = 'business_idx'
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)
        #pure_tp_tr = tp_tr.copy()
        print(f"tp_te:{len(np.unique(tp_te[user_col]))}")
        print(f"tp_tr:{len(np.unique(tp_tr[user_col]))}")
        #popular_id = evaluation.popularity_id(tp_tr, self.item_mapping)
        interacted_distr = {}
        grouped_tp_tr = tp_tr.groupby(user_col)
        if debug:
            for user_id, group in grouped_tp_tr:
                # Ensure that user_id is in the same format as profile2id keys (integer)
                user_id = int(user_id)  # Convert to int if necessary
                # Check if this user exists in profile2id
                if user_id in self.profile2id:
                    interacted = group[item_col].tolist()
                    interacted = [item_id for item_id in interacted if item_id in self.item_mapping]
                    real_interacted_items = [self.item_mapping[item_id] for item_id in interacted]
                    # Compute genre distribution and assign to interacted_distr
                    interacted_distr[user_id] = calibration_Func.compute_genre_distr(real_interacted_items)
                else:
                    print(f"User ID {user_id} not found in profile2id")

            #missing_in_profile2id = [uid for uid in tp_tr[user_col].unique() if uid not in self.profile2id]
            """if missing_in_profile2id:
                print(f"User IDs in tp_tr but missing in profile2id: {missing_in_profile2id}")
            else:
                print("All user IDs in tp_tr are present in profile2id")

            missing_in_interacted_distr = [uid for uid in tp_tr[user_col].unique() if uid not in interacted_distr]"""
            """if missing_in_interacted_distr:
                print(f"User IDs in tp_tr but missing in interacted_distr: {missing_in_interacted_distr}")
            else:
                print("All user IDs in tp_tr are present in interacted_distr")"""

        tp_tr = numerize(tp_tr, self.profile2id, self.show2id )
        tp_te = numerize(tp_te, self.profile2id, self.show2id)

        grouped_tp_tr = tp_tr.groupby('uid')  # Group by numerized user ID ('uid')
        # grouped_tp_tr['uid']
        users = []
        interacted_distr={}

        for user_id, group in grouped_tp_tr:
            # Ensure that user_id is now numerized (as 'uid')
            users.append(user_id)
            interacted = group['sid'].tolist()  # Use 'sid' for the numerized item IDs
            real_items = [self.index_to_item_global[item] for item in interacted if item in self.index_to_item_global]
            interacted = [item_id for item_id in real_items if item_id in self.item_mapping]
            real_interacted_items = [self.item_mapping[item_id] for item_id in interacted]
            # Populate interacted_distr with numerized user_id (as 'uid')
            interacted_distr[user_id] = calibration_Func.compute_genre_distr(real_interacted_items)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                     (rows_tr, cols_tr)), dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                     (rows_te, cols_te)), dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))

        #print("Shape of data_tr:", data_tr.shape)
        unique_uids_in_tp_tr = tp_tr['uid'].nunique()
        #print("Number of unique uids in tp_tr:", unique_uids_in_tp_tr)
        assert data_tr.shape[
                   0] == unique_uids_in_tp_tr, "Mismatch in number of rows in data_tr and unique users in tp_tr"
        return tp_tr,tp_te,data_tr, data_te, interacted_distr

"""def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count"""


def filter_triplets(tp, min_uc=5, min_sc=5):
    user_col = 'user_idx'
    item_col = 'business_idx'
    value_col = 'stars'
    if min_sc > 0:
        itemcount  = tp.groupby(item_col).size()
        tp = tp[tp[item_col].isin(itemcount[itemcount > min_sc].index)]
    if min_uc > 0:
        usercount = tp.groupby(user_col).size()
        tp = tp[tp[user_col].isin(usercount[usercount > min_uc].index)]
        #tp = tp[tp[user_col].isin(usercount.index[usercount['size'] >= min_uc])]
    user_interaction_counts = tp.groupby(user_col).size()
    # Check if all users meet the minimum user condition (min_uc)
    #users_below_min_uc = user_interaction_counts[user_interaction_counts < min_uc]
    """if len(users_below_min_uc) == 0:
        print("All users have at least min_uc interactions.")
    else:
        print("Some users have less than min_uc interactions:")
        print(users_below_min_uc)"""
    # Assert to ensure no users violate the condition
    assert all(user_interaction_counts > min_uc), "Some users have less than min_uc interactions!"
    usercount = tp.groupby(user_col).size().reset_index(name='size')
    itemcount = tp.groupby(item_col).size().reset_index(name='size')
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby(user_col)
    tr_list, te_list = list(), list()
    np.random.seed(98765)
    for _, group in data_grouped_by_user:  # for each user
        n_items_u = len(group)
        #print(f"n_items:{n_items_u}")
        if n_items_u >= 4:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype(
                'int64')] = True  # pick test_prop * n_items_u of items for user u
            tr_list.append(group[np.logical_not(idx)])  # tr and te are for same user but with different items
            te_list.append(group[idx])
        else:
            tr_list.append(group)
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    user_col = 'user_idx'
    item_col = 'business_idx'
    uid = tp[user_col].apply(lambda x: profile2id[x])
    sid = tp[item_col].apply(lambda x: show2id[x])
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

    file_name = 'df_item.csv'
    business_id_file = 'df_business_id.csv'
    result_dict_json = 'result.json'
    if os.path.isfile('result.json'):
        with open('result.json', 'r') as json_file:
            result_dict = json.load(json_file)
    Fifi=False
    if os.path.isfile(file_name) and os.path.isfile(result_dict_json) and os.path.isfile(business_id_file) and Fifi==True:
        df_item = pd.read_csv(file_name)
        df_item.loc[:,'categories'] = df_item['categories'].apply(lambda x: ast.literal_eval(x))
        df_business_id = pd.read_csv(business_id_file)

        if os.path.isfile('result.json'):
            with open('result.json', 'r') as json_file:
                result_dict = json.load(json_file)
        else:

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            URL = "https://blog.yelp.com/businesses/yelp_category_list/"
            page = requests.get(URL, headers=headers, timeout=5)
            soup = BeautifulSoup(page.text, "html.parser")
            sections = (soup.find_all('section'))
            filtered_sections = [section for section in sections if "b-lead-form" not in section.get("class", [])]
            result_dict = {}
            for section in filtered_sections:
                section_id = section.get('id')
                if section_id is not None:
                    section_id = section_id[2:]

                    # Find all paragraphs within the content
                    paragraphs = section.find_all('p')
                    paragraph_texts = []
                    for paragraph in paragraphs:
                        # Use the stripped_strings generator to handle line breaks
                        paragraph_texts.extend(paragraph.stripped_strings)
                    lists = section.find_all('ul')
                    list_items = []
                    for ul in lists:
                        items = ul.find_all('li')
                        list_items.extend([item.text.strip() for item in items])
                    combined_list = paragraph_texts + list_items
                    result_dict[section_id] = combined_list

            for key, value in result_dict.items():
                for i in range(len(value)):
                    if '\n' in value[i]:
                        result_dict[key][i] = value[i].split('\n')
                    else:
                        result_dict[key][i] = [value[i]]
                result_dict[key] = [item for sublist in result_dict[key] for item in sublist]

            for key, value in result_dict.items():
                result_dict[key] = [item for item in value if item.strip()]

            with open('result.json', 'w') as json_file:
                json.dump(result_dict, json_file, indent=4)

        # remove rows without categories
        # map each business id to a numerical value
        """for i, categs in enumerate(full_data_bussiness['categories']):
            new_categs = []
            for item in categs:

                # Split categories containing "/" into separate items
                subcategories = item.split("/")

                for subcategory in subcategories:
                    for key, value in result_dict.items():
                        if subcategory in value or subcategory.lower() == key:
                            new_categs.append(key)
            full_data_bussiness['categories'][i] = list(set(new_categs))
        # remove rows without categories
        df_item = full_data_bussiness[full_data_bussiness['categories'].apply(lambda x: len(x) > 0)]
        """
    else:
        print("No businesses found.")

        #url = 'https://raw.githubusercontent.com/melqkiades/yelp/master/notebooks/yelp_academic_dataset_business.json'
        url = 'yelp_academic_dataset_business.json'
        df = pd.read_json(url, lines=True)
        df = df[df['categories'].apply(lambda x: x is not None and len(x) > 0)]
        print(f"dataset business: {df.shape}")
        print("categorie before processing : ", df['categories'])

        def process_categories(categs):
            list_categ = []
            for item in categs:
                subcategories = item.split("/")
                list_categ.append(subcategories)
            return list_categ

        #df['categories'] = df['categories'].apply(process_categories) for url

        def flatten_list_optimized(nested_list):
            return [item for sublist in nested_list for item in
                    (flatten_list_optimized(sublist) if isinstance(sublist, list) else [sublist])]

        # Apply the optimized flattening function to the specified column
        # df['categories'] = df['categories'].apply(flatten_list_optimized) # for url
        df['categories'] = df['categories'].apply(lambda x: [cat.strip() for cat in x.split(',')])  # for json not url


        def filter_categories(categories):
            return [category for category in categories if category in result_dict['restaurants']]

        # Apply the filtering function to the 'Categories' column
        df = df[df['categories'].map(len) > 0]

        if args.filter_restaurants:
            print("Applying restaurants filter")
            print(f"Before being limited to restaurants :{df.shape}")
            df['categories'] = df['categories'].apply(filter_categories)
            print(f"After being limited to restaurants :{df.shape}")

        if args.state_filter:
            print("Applying state filter")
            print(f"Before removing states:{df.shape}")
            df = df[df['state'].isin(['AZ'])]
            print(f"After removing states:{df.shape}")
            df = df[df['categories'].map(len) > 0]

            # Get the 20 most common categories
        if args.filter_20:
            print("Applying 20 popular item filtering")
            print(f"Before applying 20 most common categories:{df.shape}")
            print("categories: ", df.categories)
            all_categories = df['categories'].explode()
            category_counts = Counter(all_categories)
            top20 = set([cat for (cat, count) in category_counts.most_common(20)])
            print("Top 20 categories:", top20)
            df = df.copy()
            df['categories'] = df['categories'].apply(lambda cats: [c for c in cats if c in top20])
            print(f"After applying 20 most common categories:{df.shape}")

        """full_data_bussiness = full_data_bussiness.drop('categories', axis=1)
        full_data_bussiness= full_data_bussiness.rename(columns={'new_categories':'categories'}, inplace=True)
        """
        #print(f"state: {df['state'].unique()}")
        df = df[df['categories'].map(len) > 0]
        business_mapping = {business_id: idx for idx, business_id in enumerate(df['business_id'])}
        df_business_id = pd.DataFrame(list(business_mapping.items()), columns=['business_id', 'business_idx'])
        # df_item
        full_data_bussiness = pd.merge(df, df_business_id, on='business_id', how='inner')
        df_item = full_data_bussiness[full_data_bussiness['categories'].apply(lambda x: len(x) > 0)]

        # Remove rows that end up with an empty category list
        df_item = df_item[df_item['categories'].map(len) > 0]
        df_item.to_csv('df_item.csv', index=False)
        # df_business_id = df_business_id[df_business_id['business_id'].isin(df_item['business_id'])]
        df_business_id.to_csv('df_business_id.csv', index=False)

        remaining_cats = set(cat for cats in df_item['categories'] for cat in cats)
        print(f"Unique genres: {len(remaining_cats)}" )
        #print(" Categories after filtering:", remaining_cats)


        if args.filter_20:
            assert remaining_cats.issubset(top20), " Found categories outside of top 20!"

    data_file_name = 'data.csv'

    if os.path.isfile(data_file_name) and Fifi==True:
        data = pd.read_csv(data_file_name)

    else:

        #url = 'https://raw.githubusercontent.com/knowitall/yelp-dataset-challenge/master/data/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json'
        #user_df = pd.read_json(url, lines=True)
        user_df= pd.read_json("yelp_academic_dataset_review.json", lines=True)
        num_interactions = user_df.shape[0]
        print(f"Number of user-business interactions (before merging): {num_interactions}")
        number_users=len(user_df['user_id'].unique())
        num_items=len(user_df['business_id'].unique())
        print(f"number of items:{num_items}")
        print(f"number of users:{number_users}")
        print("percentage of interactions:", (num_interactions/(num_items*number_users))*100)
        sorted_user_ids = sorted(user_df['user_id'].unique())  # it should be unique
        # map each user id to a numerical value
        users_mapping = {user_id: idx for idx, user_id in enumerate(sorted_user_ids)}
        df_users_id = pd.DataFrame(list(users_mapping.items()), columns=['user_id', 'user_idx'])
        # print(df_users_id)
        user_df = pd.merge(user_df, df_users_id, on='user_id', how='inner')
        data = pd.merge(user_df, df_business_id, on='business_id', how='inner')
        assert not data['business_idx'].isna().any(), "Found NaNs in business_idx after merging!"
        print("Number of user-business interactions after merging:", data.shape[0])
        data.to_csv(data_file_name, index=False)

    # raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)

    user_col = 'user_idx'
    item_col = 'business_idx'
    value_col = 'stars'
    # Filter Data
    print("before preprocessing: ", data.shape)
    print("data cols ",data.columns)
    print(f"before the second step of preprocessing (remove those rates that are less than 3): {data.shape}")
    #filter ratings

    raw_data = data[data[value_col] >=3] #filtering
    raw_data, user_activity, item_popularity = filter_triplets(raw_data)
    raw_data[value_col] = 1 # all interactions are positive interactions
    print(f"the final size of the dataset after all filters: {raw_data.shape}")
    unique_uid= raw_data[user_col].unique()
    n_users = unique_uid.size
    unique_item = item_popularity['business_idx'].values
    n_items = unique_item.size
    #raw data info
    print("number of total items : ", n_items) #
    print("number of total users: ", n_users) #
    num_interactions = raw_data.shape[0]  # it includes non-zero interactions (zero interactions are already removed)
    print("Number of interactions in raw_data:", num_interactions)
    print(f"percentage of interactions in raw_data:{(num_interactions/(n_items*n_users))*100}%")

    # split data to train/test/validation
    #print("n_users: ", n_users)
    n_heldout_users =int((n_users * 0.2)/2) # 20% for test and validation and 80% for train (if 60/20/20,the code should be int((n_users * 0.4)/2))
    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]
    #print("len te_users",len(te_users))
    train_plays = raw_data.loc[raw_data[user_col].isin(tr_users)]
    print("len train plays:", train_plays.shape)
    unique_sid = pd.unique(train_plays[item_col])

    if args.DP =="True":
        print("***Prepare data in DP mode***")
        interacted_items=train_plays.groupby(user_col)[item_col].apply(list).to_dict()
        for u in interacted_items:
            positive_items = interacted_items[u]
            new_positive_samples = apply_dp(positive_items, unique_sid, 0.1)
            interacted_items[u] = new_positive_samples

        new_rows=[]
        for user,movies in interacted_items.items():
            for item in movies:
                new_rows.append({user_col: user, item_col: item})
        updated_train_plays=pd.DataFrame(new_rows)
        train_plays=updated_train_plays.copy()

    pro_dir = os.path.join('pro_sg')

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    with open(os.path.join(pro_dir, 'unique_uid.txt'), 'w') as f:
        for uid in unique_uid:
            f.write('%s\n' % uid)
    # validation
    vad_plays = raw_data.loc[raw_data[user_col].isin(vd_users)]
    #print(f"len vad plays: {vad_plays.shape}")
    vad_plays = vad_plays.loc[vad_plays[item_col].isin(unique_sid)] # all items in val should also be in main train dataset
    #print(f"vad_plays: {vad_plays.shape}")
    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)
    #  test
    test_plays = raw_data.loc[raw_data[user_col].isin(te_users)]
    test_plays = test_plays.loc[test_plays[item_col].isin(unique_sid)] #  # all items in test should also be in main train dataset
    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

    raw_data.to_csv(os.path.join(pro_dir, 'raw_data.csv'), index=False)
    # train_data = numerize(train_plays, profile2id, show2id)
    # train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
    train_plays.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    # vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    # vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
    vad_plays_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

    # vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    # vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)
    vad_plays_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

    # test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    # test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)
    test_plays_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

    # test_data_te = numerize(test_plays_te, profile2id, show2id)
    # test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
    test_plays_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)


    print("Done!")
