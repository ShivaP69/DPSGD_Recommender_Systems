
def preprocess(data,thereshold=3, filter_user=5,filter_item=5 ):
    """
    To filter the movies that their rating is less than 3 and  their number of ratings is less than filter_item, and then the users that have less than filter_user
    """
    data = data[data.rating>=thereshold] #beacuse we just need positive feedbacks
    # the order of the following code is important
    item_supports = data.groupby('movieId').size()
    data = data[data['movieId'].isin(item_supports[item_supports > filter_item].index)]

    num_items = data.groupby('userId').size()
    data = data[data['userId'].isin(num_items[num_items >filter_user].index)]

    #self.data= self.data.loc[self.data['rating']>=3,'rating']=1
    #self.data= self.data.loc[self.data['rating']<3,'rating']=0

    data['rating']=1 # then at the end all ratings can be considered as one (because we just kept the items that got the high rating)
    #ratings = df.merge(df_item, on='movieId')

    return data
