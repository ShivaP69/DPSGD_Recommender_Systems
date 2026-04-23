
def preprocess(data,thereshold=3, filter_user=5,filter_item=5 ):
    user_col = 'user_idx'
    item_col = 'business_idx'
    value_col = 'stars'

    data = data[data[value_col]>=thereshold]

    # order of the following code is important

    item_supports = data.groupby(item_col).size()
    data = data[data[item_col].isin(item_supports[item_supports > filter_item].index)]   # it is worth to mention that after this step because of the next lines, it is possible to have less than 5 item_col

    num_items = data.groupby(user_col).size()
    data = data[data[user_col].isin(num_items[num_items >filter_user].index)]

    #self.data= self.data.loc[self.data['rating']>=3,'rating']=1
    #self.data= self.data.loc[self.data['rating']<3,'rating']=0
    #data = data[data.rating>=thereshold]
    #data['rating']=1
    #ratings = df.merge(df_item, on='movieId')

    return data


