import numpy as np
from scipy.spatial import distance
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from collections import defaultdict
import Calibration
def dcg_at_k(scores, k):
    positions = np.arange(1, min(k, len(scores)) + 1)
    return np.sum((2**np.array(scores) - 1) / np.log2(positions + 1))

def ndcg_at_k(scores, k):
    sorted_scores = sorted(scores, reverse=True)
    dcg = dcg_at_k(scores, k)
    idcg = dcg_at_k(sorted_scores, k)
    if idcg == 0:
        return 0.0  # Avoid division by zero
    return dcg / idcg


"""
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def difference(lst1,lst2):
  return list(set(lst1) - set(lst2))"""

def calculate_recall(recommended_items, interacted_items):
  # Calculate the number of true positives
  true_positives = len(set(recommended_items) & set(interacted_items))

  # Calculate the number of false negatives
  false_negatives = len(set(interacted_items) - set(recommended_items))
  # Calculate recall
  recall = (true_positives / (true_positives + false_negatives)) if true_positives + false_negatives>0 else 0
  #recall= true_positives/len(recommended_items)
  return recall


def calculate_total_average_recall(interacted,recommendations):


  keys=recommendations.keys()
  num_users=len(keys)
  total_recall=0
  for user_id in keys:

    Likes=interacted.get(user_id,[])

    recommended_items = recommendations.get(user_id, [])
    #print(Likes)

    recall =calculate_recall(recommended_items, Likes)
    total_recall += recall

    # Calculate the average recall across all users
  average_recall = total_recall / num_users

  return average_recall



def popularity_id(user_train, item_mapping):
    # Get the frequency of movieIds
    frequency = user_train['movieId'].value_counts(normalize=True)

    # Create a dictionary mapping movieIds to titles
    popularity = {}
    for movie_id, count in frequency.items():
        if movie_id in item_mapping:
          title = item_mapping[movie_id]
          popularity[title] = count

    return popularity

def PL(train_data, item_mapping, recommendations,users, model):
  user_interacted_items = train_data.groupby('userId')['movieId'].apply(list).to_dict() ########## training_data
  interacted_items={}
  for u in user_interacted_items:
      interacted = user_interacted_items[u]
      interacted = [item_id for item_id in interacted if item_id in item_mapping]
      interacted_items[u] = [item_mapping[item_id] for item_id in interacted]

  popular=popularity_id(train_data, item_mapping)
  gap=[]
  history=[]
  model.eval()  # Set the model to evaluation mode

  with torch.no_grad():
    for u in users:
        x = sum(popular.get(rec, 0) for rec in recommendations[u])
        gap.append(x/len(recommendations[u]))

        y=sum(popular.get(prof , 0) for prof in interacted_items[u])
        history.append(y/len(interacted_items[u]))
    GAP_Q_G=np.mean(gap)
    GAP_P_G= np.mean(history)
    PL= (GAP_Q_G - GAP_P_G)/GAP_P_G

  return PL


#p(i|R)

#𝑝(𝑖|𝑅) ~ 𝑝(𝑖) |ℐ|⁄|𝑅| for 𝑖 ∈ 𝑅

#𝑛𝑜𝑣𝑒𝑙𝑡𝑦(𝑅) = −∑𝑝(𝑖|𝑅) log2 𝑝(𝑖)
import math
def p_i_R(p_i,J,R):
  return p_i*J/R

def novelty_R_total(list_recommendation,training_data,item_mapping):
  popular=popularity_id(training_data,item_mapping)

  novelty_users={}
  total=[]

  for user_id in list_recommendation.keys():
     nov_total=0
     R=list_recommendation[user_id]
     for i in R:   # i is title of a movie
        if i in popular:
          pop_i=popular[i]
          pir= p_i_R(pop_i,training_data.shape[1],len(R))
          if pop_i!=0:
            nov_total+=pir*np.log(pop_i)

     novelty_users[user_id]=(-nov_total)
     total.append(novelty_users[user_id])
  return np.mean(total)




def Diversity(recommendation_list, df):
    diversity_user = {}
    total_diversity = 0

    for user_id, movie_list in recommendation_list.items():
        jsd_sum = 0
        num_pairs = 0

        for i in range(len(movie_list)):
            for j in range(i + 1, len(movie_list)):
                movie1 = movie_list[i]
                movie2 = movie_list[j]

                # Filter the DataFrame to get the data for the two movies
                movie1_data = df[df['title'] == str(movie1)].drop([ 'title'], axis=1).values.flatten().tolist()
                movie2_data = df[df['title'] == str(movie2)].drop(['title'], axis=1).values.flatten().tolist()
                matrix1 = np.asarray(movie1_data)
                matrix2 = np.asarray(movie2_data)
                jsd_sum += distance.jensenshannon(movie1_data, movie2_data)
                num_pairs += 1

        if num_pairs > 0:
            diversity_user[user_id] = jsd_sum / num_pairs
            total_diversity += diversity_user[user_id]

    avg_diversity = total_diversity / len(recommendation_list)
    return  avg_diversity


# This code could be used both for all users and or sub group of users




def serendepity_group(user_group, test_user_item_dict,item_mapping,recommendations, genres_df ):
    serendepity=[]
    interacted_items_test={}

    for u in user_group:
      interacted_items= test_user_item_dict[u]
      interacted_items= [index for index in interacted_items if index in item_mapping]
      interacted_items_test[u] =[item_mapping[item_id] for item_id in interacted_items]


      R= len(recommendations[u])
      H= len(interacted_items_test[u])

      cosin_sim_sum=0
      for rec_item in recommendations[u]:

        for interacted_item in interacted_items_test[u]:

          df1 = genres_df[genres_df['title'] == str(rec_item)].drop(columns=['title'])
          df2 = genres_df[genres_df['title'] == str(interacted_item)].drop(columns=['title'])
          matrix1 = np.asarray(df1.values)
          matrix2 = np.asarray(df2.values)
          cosin_sim = cosine_similarity(matrix1, matrix2)
          cosin_sim_sum+=(cosin_sim/R)
      serendepity.append(cosin_sim_sum/H)
    return (np.mean(serendepity))

def MRR(reco_items,test_user):
  rr_users=0
  for user in reco_items.keys():
    rr=0
    for x in test_user[user]:
      if x in reco_items[user]:
        rr+=1/(reco_items[user].index(x) +1)
    rr_users+=rr/len(reco_items[user])
  return rr_users/len(reco_items.keys())





def PopularItems(user_train,item_mapping):
    popularity_lst = popularity_id(user_train, item_mapping)  # Include popularity of items in train data

    # Sort items based on their popularity in descending order
    sorted_popularity = dict(sorted(popularity_lst.items(), key=lambda item: item[1], reverse=True))

    # Calculate the number of items to select (top 20%)
    n = int(len(sorted_popularity) * 0.2)

    # Select the top N items using slicing
    popular_items = list(sorted_popularity.keys())[:n]

    return popular_items



def type_of_user_total(item_mapping, popular_items,ratings):
    niche_user=[]
    blockbuster=[]
    new=[]
    diverse=[]
    x={}
    users=ratings['userId'].tolist()
    users= np.unique(users)
    all_users =ratings.groupby('userId')['movieId'].apply(list).to_dict()
    for user_id in all_users :
      size_profile= len(all_users[user_id])
      interacted =[item_id for item_id in all_users[user_id] if item_id in item_mapping]
      interacted_items= [item_mapping[item_id] for item_id in interacted]

      if size_profile>0:

        y=len(set(interacted_items) & set(popular_items))/size_profile

        if  y>0.85:
          #print("one blockbuster!")
          blockbuster.append(user_id)
        elif y<0.5:
           niche_user.append(user_id)

        else:
           diverse.append(user_id)

      else: new.append(user_id)
    return niche_user,blockbuster,diverse,new

def genres_featues(df_item):
    title_col = 'title'
    genre_col = 'genres'
    item_col = 'movieId'
    all_genres = set()

    for genres in df_item['genres']:
        all_genres.update(genres.split('|'))

    genres_df = pd.DataFrame(df_item['title'].unique(), columns=['title'])
    for genre in all_genres:
        genres_df[genre] = 0.0

    for row in df_item.itertuples():
        item_id = getattr(row, item_col)
        item_title = getattr(row, title_col)
        item_genre = getattr(row, genre_col)

        splitted = item_genre.split('|')
        genre_ratio = 1. / len(splitted)
        # item_genre = {genre: genre_ratio for genre in splitted}

        for genre in splitted:
            genres_df.loc[genres_df['title'] == item_title, genre] = genre_ratio
    return genres_df


#  category_mapping: {(i1:I1),(i2:I1),.....}
def PL_items(train_data, item_mapping,recommendations, users, model, category_mapping):
    user_interacted_items = train_data.groupby('userId')['movieId'].apply(list).to_dict()  ########## training_data
    interacted_items = {}
    for u in user_interacted_items:
        interacted = user_interacted_items[u]
        interacted = [item_id for item_id in interacted if item_id in item_mapping]
        interacted_items[u] = [item_mapping[item_id] for item_id in interacted]

    popular = popularity_id(train_data, item_mapping)
    categories = {cat: [] for cat in set(category_mapping.values())}
    for item, cat in category_mapping.items():
        categories[cat].append(item)
    PL_results = {}
    model.eval()
    with torch.no_grad():
        for category, items_in_category in categories.items():
            gap=[]
            history=[]

            for u in users:
                if u not in recommendations:
                    continue

                category_recommendations=[rec for rec in recommendations[u] if rec in items_in_category]
                category_interacted = [
                    item for item in interacted_items.get(u, []) if item in items_in_category
                ]
                #print(f"User {u}: category_interacted: {category_interacted}")
                #print(f"User {u}: category_recommendations: {category_recommendations}")
                if not category_recommendations or not category_interacted:
                    #print(f"No category_interacted or category_recommendations for category {category} and user {u}. Skipping.")
                    continue
                x = sum(popular.get(rec, 0) for rec in category_recommendations)
                gap.append(x / len(category_recommendations))

                # Calculate GAP_P_G for user history in this category
                y = sum(popular.get(item, 0) for item in category_interacted)
                history.append(y / len(category_interacted))

            # Calculate PL for this category
            GAP_Q_G = np.mean(gap) if gap else 0
            GAP_P_G = np.mean(history) if history else 0
            PL = (GAP_Q_G - GAP_P_G) / GAP_P_G if GAP_P_G != 0 else 0

            PL_results[category] = PL

        return PL_results

def categories(train_data,item_mapping):
    category_mapping = {}
    items = train_data['movieId'].tolist()
    most_popular = PopularItems(train_data, item_mapping)
    for movie_id in items:
        if movie_id in item_mapping:
            title = item_mapping[movie_id]
            if title in most_popular:
                category_mapping[title] = "I1"
            else:
                category_mapping[title] = "I2"
    return category_mapping

# By valid_distr_extraction, we can extract valid items for calculating different metrics per each item category
# The output of this function will be used for calculate_KLD_items

def valid_distr_extraction(category_mapping,recommendations,interacted_items):
    """
        Extracts and computes valid distributions for interacted items and recommendations.

        Parameters:
        - category_mapping: A dictionary mapping items to their categories.
        - recommendations: A dictionary mapping user IDs to recommended items.
        - interacted_items: A dictionary mapping user IDs to interacted items.
        - calibration_func: A class or object with a method `compute_genre_distr` to compute distributions.

        Returns:
        - valid_interacted: A dictionary of genre distributions for interacted items.
        - valid_reco_distr: A dictionary of genre distributions for recommendations.
    """
    categories = {cat: [] for cat in set(category_mapping.values())}
    for item, cat in category_mapping.items():
        categories[cat].append(item)
    valid_interacted = defaultdict(dict)
    #print(f"categories: {categories}")
    #print(f"interacted items: {interacted_items}")

    for user_id in interacted_items:
        try:
            for category, items_in_category in categories.items():
                interacted_items_train = [item for item in interacted_items[user_id] if item in items_in_category]
                if interacted_items_train:
                    valid_interacted[category][user_id] = Calibration.compute_genre_distr(interacted_items_train)
        except (KeyError, IndexError, TypeError) as e:
            print(f"Skipping user {user_id} in interacted_items due to error: {e}")
            continue
    # valid reco_dist
    valid_reco_distr = defaultdict(dict)
    for user_id in recommendations:
        for category, items_in_category in categories.items():
            try:
                valid_reco = [item for item in recommendations[user_id] if item in items_in_category]
                if valid_reco:  # Ensure there are items to process
                 valid_reco_distr[category][user_id] = Calibration.compute_genre_distr(valid_reco)
            except (KeyError, IndexError, TypeError) as e:
                print(f"Skipping user {user_id} in recommendations due to error: {e}")

    return valid_interacted,valid_reco_distr

def calculate_KLD_items(category_mapping,recommendations,reco_distr,interacted_distr):

    kld_results_category = {}
    for category in reco_distr:
        kld_category=[]
        for user_id in recommendations:
            category_distr_interacted = interacted_distr.get(category,{}).get(user_id,None)
            category_distr_reco = reco_distr.get(category, {}).get(user_id, None)
            # print(f"User {user_id} for {category} : category_interacted: {category_distr_interacted}")
            # print(f"User {user_id} for {category} : category_distr: {category_distr_reco}")
            if not category_distr_interacted or not category_distr_reco:
                #print(f"No category_interacted or category_distr for category {category} and user {user_id}. Skipping.")
                continue
            kl_divergence= Calibration.compute_kl_divergence(category_distr_interacted, category_distr_reco)
            #print(f"KL Divergence for User {user_id}, Category {category}: {kl_divergence}")
            if kl_divergence is not None:
                kld_category.append(kl_divergence)
        if kld_category:
            kld_results_category[category] = np.mean(kld_category)
        else:
            kld_results_category[category] = None  # Handle empty results for the category

    return kld_results_category

def calculate_ndcg_items(category_mapping,recommendations,interacted_items,top_k):
    categories = {cat: [] for cat in set(category_mapping.values())}
    for item, cat in category_mapping.items():
        categories[cat].append(item)
    #print(f"Items in I1: {categories['I1']}")
    #print(f"Items in I2: {categories['I2']}")
    #print(f"Recommendations:{recommendations}")
    # Debugging: Print category_mapping and filtered recommendations
    """for user, recs in recommendations.items():
        print(f"User {user}: Recommendations: {recs}")
        filtered_recs = [rec for rec in recs if str(rec) in category_mapping]
        print(f"Filtered recommendations for User {user}: {filtered_recs}")"""

    ndcg_results_category = {}
    for category, items_in_category in categories.items():
        #print(f"category: {category}")
        #print(f"items_in_category: {items_in_category}")
        ndcg_users = []
        for u in recommendations:
            category_recommendations = [rec for rec in recommendations[u] if rec in items_in_category]

            category_interacted = [
                item for item in interacted_items.get(u, []) if item in items_in_category
            ]
            #print(f"User {u}: category_recommendations for {category}: {category_recommendations}")
            #print(f"User {u}: category_interacted for {category}: {category_interacted}")

            if not category_recommendations or not category_interacted:
                #print(f"No recommendations or interactions for category {category} and user {u}. Skipping.")
                continue
            if category_recommendations and category_interacted:
                common_items = set(category_recommendations).intersection(category_interacted)
                relevance_scores = [1 if i in common_items else 0 for i in category_recommendations]
                ndcg_normal = ndcg_at_k(relevance_scores[:top_k], top_k)
                ndcg_users.append(ndcg_normal)


        ndcg_results_category[category] = np.mean(ndcg_users)
    return ndcg_results_category

def catalog_coverage(predicted: list, catalog: list) -> float:
  """
  Computes the catalog coverage for k lists of recommendations
  Parameters
  ----------
  predicted : a list of lists
      Ordered predictions
      example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
  catalog: list
      A list of all unique items in the training data
      example: ['A', 'B', 'C', 'X', 'Y', Z]
  k: integer
      The number of observed recommendation lists
      which randomly choosed in our offline setup
  Returns
  ----------
  catalog_coverage:
      The catalog coverage of the recommendations as a percent rounded to 2 decimal places
      or as a fraction rounded to 4 decimal places
  ----------
  Metric Defintion:
  Ge, M., Delgado-Battenfeld, C., & Jannach, D. (2010, September).
  Beyond accuracy: evaluating recommender systems by coverage and serendipity.
  In Proceedings of the fourth ACM conference on Recommender systems (pp. 257-260). ACM.
  """
  predicted_flattened = [p for sublist in predicted for p in sublist]
  L_predictions = len(set(predicted_flattened)) # remove duplicates
  # output: precent (%)
  #catalog_coverage = round(L_predictions / (len(catalog) * 1.0) * 100, 2)
  # output: fraction (%)
  catalog_coverage = round(L_predictions / (len(catalog) * 1.0) , 4)

  return catalog_coverage




def novelty(list_recommendation, traning_data, item_mapping, k):
    """
    Computes the novelty for a list of recommended items for all users
    Parameters
    ----------
    list_recommendation : a dict of recommedned items for all users
        Ordered predictions
        example: {1:['X', 'Y', 'Z'],2:['X', 'Y', 'Z']}

    u: integer
        The number of users in the training data
    k: integer
        The length of recommended lists per user
    Returns
    ----------
    novelty:
        The novelty of the recommendations in system level
    mean_self_information:
        The novelty of the recommendations in recommended top-N list level
    ----------
      Metric Defintion:
    Zhou, T., Kuscsik, Z., Liu, J. G., Medo, M., Wakeling, J. R., & Zhang, Y. C. (2010).
    Solving the apparent diversity-accuracy dilemma of recommender systems.
    Proceedings of the National Academy of Sciences, 107(10), 4511-4515.
    """
    """popular: dictionary
        A dictionary of all items alongside of its occurrences counter in the training data
        example: {1198: 893, 1270: 876, 593: 876, 2762: 867}"""
    popular = popularity_id(traning_data, item_mapping)
    #print(popular)
    u = len(traning_data['userId'].unique())
    mean_self_information = []
    num_users_in_rec = len(list_recommendation)
    for user_id in list_recommendation.keys():
        self_information = 0
        for item in list_recommendation[user_id]:
            if item in popular.keys():
                item_popularity = popular[item] / u
                if item_popularity != 0:
                    item_novelty_value= np.sum(-np.log2(item_popularity))
            else:
                item_novelty_value = 0
            self_information += item_novelty_value
        novelty_score = self_information / k
        mean_self_information.append(novelty_score)
    novelty = sum(mean_self_information) / num_users_in_rec
    #return novelty, mean_self_information
    return novelty


def DPF(recommendations, category_mapping):

    categories = {cat: [] for cat in set(category_mapping.values())}
    for item, cat in category_mapping.items():
        categories[cat].append(item)
    len_I1= len(categories['I1'])
    #print("len I1: ",len_I1)
    len_I2 = len(categories['I2'])
    recommended_I1=set()
    recommended_I2=set()
    for u in recommendations:
        for item in recommendations[u]:
            if item in category_mapping:
                x=category_mapping[item]
                if x=="I1":
                    recommended_I1.add(item)
                else:
                    recommended_I2.add(item)
    #
    exposure_I1 = len(recommended_I1) / len_I1 if len_I1 > 0 else 0
    exposure_I2 = len(recommended_I2) / len_I2 if len_I2 > 0 else 0
    dpf = exposure_I1 - exposure_I2
    total_items=len(recommended_I1) +len(recommended_I2)
    normalized_exposure_1 = len(recommended_I1) / total_items if total_items > 0 else 0 # based on the paper results
    normalized_exposure_2 = len(recommended_I2) / total_items if total_items > 0 else 0
    normalized_dpf = normalized_exposure_1 - normalized_exposure_2
    return dpf,normalized_dpf, exposure_I1, exposure_I2 ,normalized_exposure_1, normalized_exposure_2  #  exposure_I1, exposure_I2 could be considered as I1 coverage and I2 coverage


