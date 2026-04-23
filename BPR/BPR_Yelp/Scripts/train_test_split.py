


from sklearn.model_selection import train_test_split
from Preprocess import preprocess
import  pandas as pd
from sklearn.utils import shuffle


def train_test_split_version1(ratings):
    user_col = 'user_idx'
    ratings = shuffle(ratings, random_state=42)

    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    # print("columns", test_data.columns)
    missing_users = set(test_data[user_col]) - set(train_data[user_col])

    if len(missing_users) > 1:
        sampled_rows_list = []

        for user_id in missing_users:
            sampled_rows = test_data[test_data[user_col] == user_id].sample(frac=0.2)

            sampled_rows_list.append(sampled_rows)

        train_data = pd.concat([train_data] + sampled_rows_list)
        test_data = test_data[~test_data.index.isin(pd.concat(sampled_rows_list).index)]

    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)




