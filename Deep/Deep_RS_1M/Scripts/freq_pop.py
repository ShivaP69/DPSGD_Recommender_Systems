
import os

from ItemMapping import create_item_mapping
import json
from csv import writer
import calibration_Func
import evaluation

from numpy import percentile
import json
import pandas as pd
import numpy as np
from Preprocess import preprocess

#writer = SummaryWriter(log_dir='logs')
if __name__ == '__main__':
    # 100 k

    title_col = 'title'
    genre_col = 'genres'
    item_col = 'movieId'
    df_item = pd.read_csv('movie.csv')
    df_item = df_item[ df_item[genre_col] != '(no genres listed)']  # eliminate movies that had no genre information attached
    print('dimension: ', df_item.shape)
    item_mapping = create_item_mapping(df_item, item_col, title_col, genre_col)

    names = ['userId', 'movieId', 'rating', 'timestamp']
    file_path = 'ratings.dat'
    df = pd.read_table(file_path, names=names, sep="::", engine='python')
    print(f"pre_preprocess:{df.shape}")
    df = preprocess(df)
    print(f"post_preprocess:{df.shape}")
    print(len(np.unique(list(df['userId']))))
    print(len(np.unique(list(df['movieId']))))
    p= NDCG.popularity_id(df, item_mapping)
    print(p)


    # Assuming p is your dictionary


    print("Current Directory:", os.getcwd())

    # You can specify a full path to control where to save the file
    #filepath = os.path.join(os.getcwd(), 'popularity.json')
    #filepath= os.path.join('/home/parsar0000/PycharmProjects/Deep_RS/', 'popularity.json')
    #new_p = {str(key): value for key, value in p.items()}
    """filepath='popularity.json'
    with open(filepath, 'w') as f:
        json.dump(new_p, f, indent=4)"""

    with open("popularity.json", "r") as file:
        data = json.load(file)

    # Extract popularity values for histogram
    popularity_values = list(data.values())
    quartiles = percentile(popularity_values, [50, 90])
    # calculate min/max
    data_min, data_max = min(popularity_values), max(popularity_values)

    # print 5-number summary
    print('Min:', data_min)
    print('Median: %.4f' % quartiles[0])
    print('99 percentile: %.4f' % quartiles[1])
    # print('Q3: %.3f' % quartiles[2])
    print('Max: %.4f' % data_max)
    import matplotlib.pyplot as plt

    # Create histogram for popularity values
    plt.figure(figsize=(10, 6))
    plt.hist(popularity_values, bins=30, color='blue', alpha=0.7)
    plt.title('Frequency Distribution of Movie Popularity (1M)')
    plt.xlabel('Popularity Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()