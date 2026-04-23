

def preprocess(data,thereshold=3, filter_user=5,filter_item=5 ):
    user_col = 'userId'
    item_col = 'movieId'
    value_col = 'rating'
    time_col = 'timestamp'
    data = data[data.rating>=thereshold]
    num_items = data.groupby('userId').size()
    data = data[data['userId'].isin(num_items[num_items >filter_user].index)]
    item_supports = data.groupby('movieId').size()
    data = data[data['movieId'].isin(item_supports[item_supports > filter_item].index)]

    #self.data= self.data.loc[self.data['rating']>=3,'rating']=1
    #self.data= self.data.loc[self.data['rating']<3,'rating']=0
    #data = data[data.rating>=thereshold]
    #data['rating']=1
    #ratings = df.merge(df_item, on='movieId')

    return data
