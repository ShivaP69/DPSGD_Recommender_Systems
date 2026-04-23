import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import StepLR
from MF_MLP import NeuMF
from DataProcess import MovieLensTrainDataset
from MF_MLP import calculate_loss
import torch.nn as nn
#import pytorch_lightning as pl
from train_test_split import train_test_split_version1
from torch.utils.data import DataLoader, RandomSampler
import evaluation
from plot import plotting,plotting_items
from opacus import PrivacyEngine
#from pandas.io.json import json_normalize
import json
import pickle
import requests
from bs4 import BeautifulSoup
from ItemMapping import create_item_mapping
import torch
import Preprocess
import calibration_Func
import os
import argparse
import ast
import sys

# Press the green button in the gutter to run the script.
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
#writer = SummaryWriter(log_dir='logs')

def read_df_item_and_df_business():
    file_name = 'df_item.csv'
    business_id_file = 'df_business_id.csv'
    #result_dict_json = 'result.json'

    if os.path.isfile('result.json'):
        with open('result.json', 'r') as json_file:
            result_dict = json.load(json_file)
    # flaggy=False
    # if flaggy==True:
    if os.path.isfile(file_name) and os.path.isfile(business_id_file):
        df_item = pd.read_csv(file_name)
        df_item['categories'] = df_item['categories'].apply(lambda x: ast.literal_eval(x))
        df_business_id = pd.read_csv(business_id_file)

        if os.path.isfile('result.json'):
            with open('result.json', 'r') as json_file:
                result_dict = json.load(json_file)
        else:
            print("result json is not here!")

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
        print("making df_business and other datasets")
        # url = 'https://raw.githubusercontent.com/melqkiades/yelp/master/notebooks/yelp_academic_dataset_business.json'
        url = 'yelp_academic_dataset_business.json'
        df = pd.read_json(url, lines=True)
        print(f"dataset business: {df.shape}")
        df = df[df['categories'].apply(lambda x: x is not None and len(x) > 0)]
        print(f"dataset business: {df.shape}")
        print("categorie before processing : ", df['categories'])

        def process_categories(categs):
            list_categ = []
            for item in categs:
                subcategories = item.split("/")
                list_categ.append(subcategories)
            return list_categ

        # df['categories'] = df['categories'].apply(process_categories) # for url

        def flatten_list_optimized(nested_list):
            return [item for sublist in nested_list for item in
                    (flatten_list_optimized(sublist) if isinstance(sublist, list) else [sublist])]

        # Apply the optimized flattening function to the specified column
        # df['categories'] = df['categories'].apply(flatten_list_optimized) # for url
        df['categories'] = df['categories'].apply(lambda x: [cat.strip() for cat in x.split(',')])  # for json not url

        def filter_categories(categories):
            return [category for category in categories if category in result_dict['restaurants']]

        # Apply the filtering function to the 'Categories' column
        args.filter = False
        if args.filter:
            df['categories'] = df['categories'].apply(filter_categories)
            print("Applying filter")

        print("Business df", df.columns)
        # print(f"is open: {df['is_open'].unique()}")
        print(f"state: {df['state'].unique()}")
        df = df[df['categories'].map(len) > 0]
        # print(f"Before removing non-open:{df.shape}")
        # df = df[df['is_open'] == 1]
        print(f"Before removing states:{df.shape}")
        df = df[df['state'].isin(['AZ'])]
        print(f"After removing states:{df.shape}")
        df = df[df['categories'].map(len) > 0]

        """full_data_bussiness = full_data_bussiness.drop('categories', axis=1)
        full_data_bussiness= full_data_bussiness.rename(columns={'new_categories':'categories'}, inplace=True)
        """
        print("categories: ", df.categories)
        business_mapping = {business_id: idx for idx, business_id in enumerate(df['business_id'])}
        df_business_id = pd.DataFrame(list(business_mapping.items()), columns=['business_id', 'business_idx'])
        full_data_bussiness = pd.merge(df, df_business_id, on='business_id', how='inner')
        df_item = full_data_bussiness[full_data_bussiness['categories'].apply(lambda x: len(x) > 0)]
        df_item.to_csv('df_item.csv', index=False)
        df_business_id.to_csv('df_business_id.csv', index=False)

    return df_item,df_business_id,result_dict


def read_data(df_business_id):
    data_file_name = 'data.csv'
    if os.path.isfile(data_file_name):
        data = pd.read_csv(data_file_name)
    else:
        # url = 'https://raw.githubusercontent.com/knowitall/yelp-dataset-challenge/master/data/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json'
        url = 'yelp_academic_dataset_review.json'
        user_df = pd.read_json(url, lines=True)
        number_users = len(user_df['user_id'].unique())
        num_items = len(user_df['business_id'].unique())
        print(f"number of items:{num_items}")
        print(f"number of users:{number_users}")
        print("percentage of interactions:", (num_interactions / (num_items * number_users)) * 100)
        # user_df= pd.read_json("yelp_academic_dataset_review.json", lines=True)
        sorted_user_ids = sorted(user_df['user_id'].unique())  # it should be unique
        # map each uset id to a numerical value
        users_mapping = {user_id: idx for idx, user_id in enumerate(sorted_user_ids)}
        df_users_id = pd.DataFrame(list(users_mapping.items()), columns=['user_id', 'user_idx'])
        # print(df_users_id)
        user_df = pd.merge(user_df, df_users_id, on='user_id', how='inner')
        #valid_id = df_business_id['business_id'].tolist()
        #user_df = user_df[user_df['business_id'].isin(valid_id)]
        data = pd.merge(user_df, df_business_id, on='business_id', how='inner')
        assert not data['business_idx'].isna().any(), "Found NaNs in business_idx after merging!"
        print("size of the original dataset after first filtering:", len(data))
        data.to_csv(data_file_name, index=False)

    return data



def plot(data,item_mapping):
    popular = evaluation.PopularItems(data, item_mapping)
    niche_user, blockbuster, diverse, new = evaluation.type_of_user_total(item_mapping, popular, data)
    # user_groups= [niche_user, blockbuster,diverse, new]
    user_groups = {
        "niche": niche_user,
        "blockbuster": blockbuster,
        "diverse": diverse,
        # "new": new
    }
    plotting(user_groups)
    total_users = sum(len(users) for users in user_groups.values())
    items_popularity = evaluation.popularity_id(data, item_mapping)
    # plot_popularity(items_popularity)
    popularity_df = pd.DataFrame(list(items_popularity.items()), columns=['Movie Title', 'Popularity'])
    popularity_df.to_csv('popularity.csv', index=False)
    sys.exit()  # This code is just used when we need to reproduce the distr figure in the paper, so after having the plot, we can exit

def read_train_test_and_user_indices(data,all_movieIds):
    # reading user indices
    user_col = 'user_idx'
    item_col = 'business_idx'
    user_indices = {}
    ratio = 50
    indices_file_path = f'indices_file_path_{ratio}.pkl'
    if os.path.isfile('train_data.csv') and os.path.isfile('test_data.csv'):
        train_data = pd.read_csv('train_data.csv')
        test_data = pd.read_csv('test_data.csv')
        test_user_item_dict = test_data.groupby(user_col)[item_col].apply(list).to_dict()
        user_interacted_items = train_data.groupby(user_col)[item_col].apply(list).to_dict()  ########## training_data
        if os.path.isfile(indices_file_path):
            with open(indices_file_path, 'rb') as f:
                user_indices = pickle.load(f)
                print("User Indices has been loaded")

        else:
            np.random.seed(42)  # Ensure reproducibility
            for u in test_user_item_dict:
                not_interacted_items = set(all_movieIds) - set(
                    user_interacted_items[u])  # user_interacted_items comes from train data
                not_interacted_items = set(not_interacted_items) - set(test_user_item_dict[u])
                interacted_test = test_user_item_dict[u]
                num_to_select = min(len(not_interacted_items), ratio * len(interacted_test))
                selected_not_interacted = list(
                    np.random.choice(list(not_interacted_items), size=num_to_select, replace=False))
                test_items = selected_not_interacted + list(interacted_test)
                test_items = list(set(test_items))
                user_indices[u] = test_items

            with open(indices_file_path, 'wb') as file:
                pickle.dump(user_indices, file)

    else:

        train_data, test_data = train_test_split_version1(data)
        train_data.to_csv('train_data.csv', index=False)
        test_data.to_csv('test_data.csv', index=False)
        test_user_item_dict = test_data.groupby(user_col)[item_col].apply(list).to_dict()
        user_interacted_items = train_data.groupby(user_col)[item_col].apply(list).to_dict()  ########## training_data
        np.random.seed(42)  # Ensure reproducibility
        for u in test_user_item_dict:
            if u not in user_interacted_items:
                print(u)
            if u in user_interacted_items:
                not_interacted_items = set(all_movieIds) - set(
                    user_interacted_items[u])  # user_interacted_items comes from train data
                not_interacted_items = set(not_interacted_items) - set(test_user_item_dict[u])
                interacted_test = test_user_item_dict[u]
                num_to_select = min(len(not_interacted_items), ratio * len(interacted_test))
                selected_not_interacted = list(
                    np.random.choice(list(not_interacted_items), size=num_to_select, replace=False))
                test_items = selected_not_interacted + list(interacted_test)
                test_items = list(set(test_items))
                user_indices[u] = test_items

        with open(indices_file_path, 'wb') as file:
            pickle.dump(user_indices, file)
    return user_indices,train_data,test_data,user_interacted_items,test_user_item_dict

def eval_model(model,val_dataloader,recall_test,KLD_test,test_ndcg,test_user_item_dict,user_indices,item_mapping,interacted_distr,interacted_items_test,top_k):
    # Validation
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss_test = []
    with (torch.no_grad()):
        for batch in val_dataloader:
            user_input, item_input, labels = [x.to(device) for x in batch]
            user_input = user_input.long()  # Convert to torch.long
            item_input = item_input.long()  # Convert to torch.long
            predicted_labels = model(user_input, item_input)
            all_predictions.extend(predicted_labels.squeeze().tolist())
            all_labels.extend(labels.tolist())
            # loss = nn.BCELoss()(predicted_labels, labels.float())
            loss = torch.nn.functional.binary_cross_entropy(predicted_labels, labels.view(-1,
                                           1).float())  # For validation, we can use this style because we are not using the loss of this step for training the model, so no need to be compatible with Opacus
            total_loss_test.append(loss.item())
    return total_loss_test



def generate_recommendations(model,test_user_item_dict,user_indices,item_mapping,top_k,interacted_distr,interacted_items_test,final_results=False):
    recommendations = {}
    average_ndcg_calibrated = []
    reco_distr = {}
    KLD_normal = []
    KLD_score_zero = []
    item_candidates = {}
    calibrated_recommendations = {}
    average_ndcg_normal=[]
    reco_distr_calibrated = {}
    KLD_calibrated=[]
    # num_users = len(test_user_item_dict)
    # Accuracy = []
    # AUC = []
    # Accuracy_train = []
    # AUC_train = []
    model.eval()
    with (torch.no_grad()):
        for u in test_user_item_dict: # generate recommendations for all users in the test dataset
            try:
                test_items = user_indices[u] # read user_indices : all recommendations will be done by evaluating these items
                test_items = [item_id for item_id in test_items if item_id in item_mapping]
                user_k = len(test_items)
                # Calculate predicted scores for the test items
                user_tensor = torch.tensor([u] * user_k, device=device).long()
                item_tensor = torch.tensor(test_items, device=device).long()
                predicted_scores = model(user_tensor, item_tensor).detach().cpu().numpy() # what is the model prediction score for user u and the items in the test_items
                predicted_scores_flattened = predicted_scores.flatten()
                top_indices = np.argsort(-predicted_scores_flattened)[:top_k] # sort the items based on their predicted score (highest to lowest, and then pick top_k)
                top20_items = [test_items[i] for i in top_indices]
                recommendations[u] = [item_mapping[item_id] for item_id in top20_items] # convert id to name of the items
                item_candidates[u] = (calibration_Func.generate_item_candidates(item_mapping, test_items, predicted_scores)) # generate item candidates for calibrated list
                calibrated_recommendations[u] = calibration_Func.calib_recommend(item_candidates[u], interacted_distr[u],topn=top_k, lmbda=0.8)
                common_items_normal = set(recommendations[u]).intersection(interacted_items_test[u])
                relevance_scores_normal = [1 if i in common_items_normal else 0 for i in recommendations[u]]
                common_items = set(calibrated_recommendations[u]).intersection(interacted_items_test[u])
                relevance_scores_calibrated = [1 if i in common_items else 0 for i in calibrated_recommendations[u]]

                ndcg_calibrated = evaluation.ndcg_at_k(relevance_scores_calibrated[:top_k], top_k)
                average_ndcg_calibrated.append(ndcg_calibrated)
                ndcg_normal = evaluation.ndcg_at_k(relevance_scores_normal[:top_k], top_k)
                average_ndcg_normal.append(ndcg_normal)
                # all_labels = [1 if item in test_user_item_dict[u] else 0 for item in test_items]
                # all_predictions = [1 if p >= 0.5 else 0 for p in predicted_scores]
                # accuracy = accuracy_score(all_labels, all_predictions)
                # auc = roc_auc_score(all_labels, all_predictions)
                # Accuracy.append(accuracy)
                reco_distr[u] = calibration_Func.compute_genre_distr(recommendations[u])
                # compute KLD
                KLD_normal.append(calibration_Func.compute_kl_divergence(interacted_distr[u], reco_distr[u]))
                KLD_score_zero.append(calibration_Func.compute_kl_divergence(interacted_distr[u], reco_distr[u]))

                # for calculating KLD for calibrated recommendations
                reco_distr_calibrated[u] = calibration_Func.compute_genre_distr(
                    calibrated_recommendations[u])
                KLD_calibrated.append(calibration_Func.compute_kl_divergence(interacted_distr[u],
                                                                             reco_distr_calibrated[u]))
            except Exception as e:
                print(f"[Warning] Skipped user {u} due to error: {e}")
                continue
    KLDN = np.mean(KLD_normal)
    KLDC = np.mean(KLD_calibrated)
    if final_results:
        return recommendations,calibrated_recommendations,average_ndcg_normal,average_ndcg_calibrated,KLDN,KLDC
    else:
        return recommendations, KLDN, average_ndcg_normal,KLD_score_zero


def train(args):
    max_grad_norm = args.max_grad_norm
    delta = args.delta
    top_k=args.k
    calibration=args.calibration
    batch_size=args.batch_size
    num_epochs = args.num_epochs
    initial_learning_rate = args.learning_rate  # initial learning rate
    learning_rate_factor = 0.2
    best_test_loss = float("inf")  # Initialize with a large value
    consecutive_epochs_without_improvement = 0
    max_consecutive_epochs_without_improvement = 6  # Adjust as needed
    mf_dim=args.mf_dim
    dropout= 0.5
    ratio = 50
    mlp_layer_sizes = args.mlp_layer_sizes # Example sizes, modify as needed
    mlp_layer_sizes = [int(x) for x in mlp_layer_sizes.split(',')]
    user_col = 'user_idx'
    item_col = 'business_idx'
    ###############################################################################
    # reading, and preparing data
    ###############################################################################
    df_item,df_business_id,result_dict= read_df_item_and_df_business()
    data=read_data(df_business_id)

    item_mapping = create_item_mapping(df_item, 'business_idx', 'name', 'categories')
    print(f"df item cats:{df_item['categories']}")

    genres_df = evaluation.genres_features(df_item, result_dict)
    print(f"generes df: {genres_df}")
    #print(data.head())
    num_users, num_items = data[user_col].max() + 1, data[item_col].max() + 1
    all_movieIds = data[item_col].unique()

    # train_loader,test_loader, train_data, test_data, all_movieIds= create_data_loader(data)
    print("before preprocessing:", data.shape)
    data = Preprocess.preprocess(data)
    print(data.head)
    print(f"data size is {data.shape}")
    num_users_pre, num_items_pre = data[user_col].max() + 1, data[item_col].max() + 1
    print(f"number of users: {len(data[user_col].unique())}")
    print(f"number of items: {len(data[item_col].unique())}")
    item_genre_counts = {item_id: len(item.genres) for item_id, item in
                         item_mapping.items()}  # how many genres/business we have for each item
    # print(item_genre_counts)
    counts = list(item_genre_counts.values())
    print(f"Total items: {len(counts)}")
    print(f"Min genres per item: {np.min(counts)}")
    print(f"Max genres per item: {np.max(counts)}")
    print(f"Mean genres per item: {np.mean(counts):.2f}")
    print(f"Median genres per item: {np.median(counts)}")
    #print(data.columns)
    # in case of Plotting
    if args.Plot:
        plot(data,item_mapping)
    # reading train, test, user_indices and user history in train data and users interactions from test data
    user_indices, train_data, test_data,user_interacted_items,test_user_item_dict= read_train_test_and_user_indices(data,all_movieIds)
    print("train_data shape",train_data.shape)
    interacted_distr = {}
    for user_id in user_interacted_items:
        interacted = user_interacted_items[user_id]
        interacted = [item_id for item_id in interacted if item_id in item_mapping]
        interacted_items_init= [item_mapping[item_id] for item_id in interacted]
        interacted_distr[user_id] = calibration_Func.compute_genre_distr(interacted_items_init)

    interacted_items = {}
    for u in user_interacted_items:
        interacted = user_interacted_items[u]
        interacted = [item_id for item_id in interacted if item_id in item_mapping]
        interacted_items[u] = [item_mapping[item_id] for item_id in interacted]

    interacted_items_test = {}
    for u in test_user_item_dict:
        interacted_items_t = [item_id for item_id in test_user_item_dict[u] if item_id in item_mapping]
        interacted_items_test[u] = [item_mapping[item_id] for item_id in interacted_items_t]


    # defining DataLoader for both train and test data
    train_dataloader = DataLoader(MovieLensTrainDataset(train_data,all_movieIds), batch_size=args.batch_size, num_workers=2)
    val_dataloader = DataLoader(MovieLensTrainDataset(test_data,all_movieIds), batch_size=args.batch_size, num_workers=2)

    ###############################################################################
    # defining the model
    ###############################################################################
    model = NeuMF(num_users, num_items, mf_dim=mf_dim, mlp_layer_sizes=mlp_layer_sizes, dropout=dropout)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=initial_learning_rate)
    ###############################################################################
    #  # setting the configuration for applying DPSGD
    ###############################################################################
    if args.DPSGD=='True':
        if args.filter:
         model_name= 'Yelp_DPSGD_Deep_restaurant'
        else:
            model_name= 'Yelp_DPSGD_Deep'
        delta = 1e-5
        noise_multiplier = args.noise_multiplier
        #max_grad_norm = 2
        max_grad_norm = args.max_grad_norm

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm, )
    else:
        delta='None'
        noise_multiplier='None'
        max_grad_norm='None'
        if args.filter:
             model_name='Yelp_Deep_restaurant'
        else:
            model_name='Yelp_Deep'

    # Initialize the StepLR scheduler
    step_size=4
    gamma= 0.9
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    test_losses=[]
    test_ndcg = []
    # accuracy_test=[]
    KLD_test = []
    #total_KLD_train=[]
    recall_test = []
    train_losses=[]
    privacy_epsilons=[]

    ###############################################################################
    # Training
    ###############################################################################
    for epoch in range(num_epochs):
        model.train()  # Set the model in training mode
        total_loss_train = []
        for batch in train_dataloader:
            user_input, item_input, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()  # Zero the gradients
            predicted_labels = model(user_input, item_input)
            loss = calculate_loss(predicted_labels, labels)  # because of opacus compatibility
            loss.backward()  # Backpropagation
            #nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()  # Update the model parameters
            total_loss_train.append(loss.item())

        if args.DPSGD=='True':
            epsilon = privacy_engine.get_epsilon(delta)
            privacy_epsilons.append(epsilon)
            print(
                f"\tTrain Epoch: [{epoch + 1}/{num_epochs}] \t"
                f"Train Loss: {np.mean(total_loss_train):.6f} "
                f"(ε = {epsilon:.2f}, δ = {delta})"
            )
        else:

            print(
                f"\tTrain Epoch: [{epoch + 1}/{num_epochs}] \t"
                f"Train Loss: {np.mean(total_loss_train):.6f} "
            )
        train_losses.append(np.mean(total_loss_train))

        # Validation
        print("start validation")
        total_loss_test = eval_model(model, val_dataloader, recall_test, KLD_test, test_ndcg, test_user_item_dict,
                                     user_indices,
                                     item_mapping, interacted_distr, interacted_items_test, top_k)
        # Generate recommendations for test users
        print("Generating recommendations")
        recommendations, KLD, average_ndcg_test, KLD_score_zero = generate_recommendations(model, test_user_item_dict,
                                                                                           user_indices,
                                                                                           item_mapping, top_k,
                                                                                           interacted_distr,
                                                                                           interacted_items_test)
        recall_test.append(evaluation.calculate_total_average_recall(interacted_items_test, recommendations))
        KLD_test.append(np.mean(KLD))
        test_ndcg.append(np.mean(average_ndcg_test))
        #writer.add_scalar("Loss/test", np.mean(total_loss_test), epoch)
        test_losses.append(np.mean(total_loss_test))
        print(f'Loss test: {np.mean(total_loss_test):.4f}')
        #accuracy = accuracy_score(all_labels, [1 if p >= 0.5 else 0 for p in all_predictions])
        #auc = roc_auc_score(all_labels, all_predictions)
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {np.mean(total_loss_test):.4f}')
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}')
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}, AUC: {auc:.4f}')

        scheduler.step()
        if np.mean(total_loss_test)< best_test_loss:
            best_test_loss=np.mean(total_loss_test)
            consecutive_epochs_without_improvement=0 # reset
        else:
            consecutive_epochs_without_improvement+=1

        if consecutive_epochs_without_improvement >= max_consecutive_epochs_without_improvement:
            print(
                f"Stopping training due to lack of improvement for {max_consecutive_epochs_without_improvement} epochs.")
            break  # Exit the training loop

        """if epoch > 0 and test_losses[-1] >= test_losses[-2]:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 4:  # Adjust the number as needed.
                # Increase the learning rate
                new_learning_rate = initial_learning_rate * learning_rate_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_learning_rate
                print(f"Increasing learning rate to {new_learning_rate}")
                epochs_without_improvement = 0  # Reset the counter"""

    print(f"test_losses= {test_losses}")
    print(f"train_losses= {train_losses}")
    print(f"KLD_test= {KLD_test}")
    #print(f"KLD_train={total_KLD_train}")
    print(f"NDCG_test ={test_ndcg}")
    print(f"recall_test= {recall_test}")
    print("KLD normal", np.mean(KLD))
    print(f"Average NDCG in normal mode @{top_k}: {np.mean(average_ndcg_test):.4f}")  # last epoch
    #print(f"Average NDCG in normal mode @{top_k}: {np.mean(average_ndcg_test):.4f}")

    ###############################################################################
    # Final experiments and saving the results
    ###############################################################################
    def evaluation_and_save():
        #user_col = 'user_idx'
        item_col = 'business_idx'
        #value_col = 'stars'
        top_k = 10
        unique_ids = list(train_data[item_col].unique())
        ununique_ids_valid = [item_id for item_id in unique_ids if item_id in item_mapping]
        ununique_items = [item_mapping[item_id] for item_id in ununique_ids_valid]
        # call generate_Recommendations
        recommendations, calibrated_recommendations, average_ndcg_normal, average_ndcg_calibrated,KLDN,KLDC = generate_recommendations(
            model, test_user_item_dict, user_indices, item_mapping, top_k, interacted_distr, interacted_items_test,
            final_results=True)
        print("End of the test recommendations creation")
        #print(f"ununique_items:{ununique_items}")
        #genres_df=NDCG.genres_features(df_item,result_dict)

        MRR= evaluation.MRR(recommendations,interacted_items_test)
        print(f"MRR in normal mode: {MRR}")
        Calibrated_MRR= evaluation.MRR(calibrated_recommendations,interacted_items_test)
        print(f"MRR in calibrated mode: {Calibrated_MRR}")

        NDCG_normal= np.mean(average_ndcg_normal)
        Calibrated_NDCG= np.mean(average_ndcg_calibrated)
        print(f"Average NDCG in normal mode @{top_k}: {NDCG_normal:.4f}")
        print(f"Average NDCG in Calibarated mode @{top_k}: {Calibrated_NDCG:.4f}")


        average_recall = evaluation.calculate_total_average_recall(interacted_items_test,recommendations )
        print(f"Total Average Recall in normal state: {average_recall:.4f}")

        average_recall_calibrated = evaluation.calculate_total_average_recall(interacted_items_test,calibrated_recommendations )
        print(f"Total Average Recall in calibrated state: {average_recall_calibrated:.4f}")


        normal_PL=evaluation.PL(train_data,item_mapping,recommendations,list(recommendations.keys()),model)
        print(f"PL normal: {normal_PL}")
        Calibrated_PL= evaluation.PL(train_data, item_mapping,calibrated_recommendations,list(calibrated_recommendations.keys()),model)
        print(f"PL calibrated: {Calibrated_PL}")
        novelty_normal= evaluation.novelty(recommendations,train_data,item_mapping,top_k)
        print("novelty normal :", novelty_normal)
        Calibrated_novelty=evaluation.novelty(calibrated_recommendations,train_data,item_mapping,top_k)
        print("novelty calibrated:",Calibrated_novelty)
        coverage = evaluation.catalog_coverage(list(recommendations.values()), ununique_items)
        print(f"coverage:{coverage}")

        #Diversity_normal= NDCG.Diversity(recommendations, genres_df)
        #print(f" Diversity normal : {Diversity_normal}")
        #Diversity_Calibrated= NDCG.Diversity(calibrated_recommendations, genres_df)
        #print(f" Diversity Calibrated : {Diversity_Calibrated}")

        # for all users ( we can substitute calibrate recommendations with recommnedations and vice versa)
        #Serendepity_normal=NDCG.serendepity_group(list(recommendations.keys()), test_user_item_dict,item_mapping,recommendations, genres_df )
        #print(f"Total Average Serendepity in normal model {Serendepity_normal}")
        #Serendepity_Calibrated= NDCG.serendepity_group(list(calibrated_recommendations.keys()), test_user_item_dict,item_mapping,calibrated_recommendations, genres_df )
        #print(f"Total Average Serendepity in calibrated model{Serendepity_Calibrated}")

        popular=evaluation.PopularItems(train_data, item_mapping)
        category_mapping = evaluation.categories(train_data, item_mapping)
        valid_interacted, valid_reco_distr = evaluation.valid_distr_extraction(category_mapping, recommendations,
                                                                               interacted_items)
        print(f"interacted_items_test:{interacted_items_test}")
        PL_per_category = evaluation.PL_items(train_data, item_mapping, recommendations, list(recommendations.keys()),
                                              model, category_mapping)
        kld_pre_category = evaluation.calculate_KLD_items(recommendations,valid_reco_distr,
                                                          valid_interacted)
        ndcg_per_category = evaluation.calculate_ndcg_items(category_mapping, recommendations, interacted_items_test,
                                                            top_k)

        average_total_dpf,normalized_dpf, exposure_I1, exposure_I2, normalized_exposure_1, normalized_exposure_2 = evaluation.DPF(recommendations,
                                                                     category_mapping)
        print(f"DPF:{average_total_dpf}")
        print(f"normalized_dpf: {normalized_dpf}")
        print(f"normalized_exposure_1: {normalized_exposure_1}")
        print(f"normalized_exposure_2: {normalized_exposure_2}")
        print(f"exposure_I1 : {exposure_I1}")
        print(f"exposure_I2 : {exposure_I2}")
        print(f"PL_per_category:{PL_per_category}")
        print(f"kld_pre_category:{kld_pre_category}")
        print(f"ndcg_per_category:{ndcg_per_category}")
        print(f"average_total_dpf:{average_total_dpf}")


        niche_user, blockbuster,diverse, new= evaluation.type_of_user_total(item_mapping, popular, train_data)
        #user_groups= [niche_user, blockbuster,diverse, new]
        user_groups = {
            "niche_user": niche_user,
            "blockbuster": blockbuster,
            "diverse": diverse,
            "new": new
        }

        def calculate_group_metrics(group_name, user_group, recommendations, calibrated_recommendations, train_data, item_mapping, test_user_item_dict, interacted_items_test, model):
            if len(user_group) == 0:
                return None

            user_group_recommendations = {u: recommendations[u] for u in recommendations if u in user_group}
            user_group_recommendations_calibrated = {u: calibrated_recommendations[u] for u in calibrated_recommendations if u in user_group}

            # Calculate metrics
            novelty_normal = evaluation.novelty(user_group_recommendations, train_data, item_mapping,top_k)
            novelty_calibrated = evaluation.novelty(user_group_recommendations_calibrated, train_data, item_mapping,top_k)
            #diversity_normal = NDCG.Diversity(user_group_recommendations, genres_df)
            #diversity_calibrated = NDCG.Diversity(user_group_recommendations_calibrated, genres_df)
            #serendipity_normal = NDCG.serendepity_group(user_group, test_user_item_dict, item_mapping, user_group_recommendations, genres_df)
            #serendipity_calibrated = NDCG.serendepity_group(user_group, test_user_item_dict, item_mapping, user_group_recommendations_calibrated, genres_df)
            mrr_normal = evaluation.MRR(user_group_recommendations, interacted_items_test)
            mrr_calibrated = evaluation.MRR(user_group_recommendations_calibrated, interacted_items_test)
            pl_normal = evaluation.PL(train_data, item_mapping, user_group_recommendations, user_group, model)
            pl_calibrated = evaluation.PL(train_data, item_mapping, user_group_recommendations_calibrated, user_group, model)
            # Calculate KLD
            kld_normal = np.mean([calibration_Func.compute_kl_divergence(interacted_distr[user_id], calibration_Func.compute_genre_distr(user_group_recommendations[user_id])) for user_id in user_group if user_id in user_group_recommendations])
            kld_calibrated = np.mean([calibration_Func.compute_kl_divergence(interacted_distr[user_id], calibration_Func.compute_genre_distr(user_group_recommendations_calibrated[user_id])) for user_id in user_group if user_id in user_group_recommendations_calibrated])
            ndcg_per_user_type = []
            for u in user_group_recommendations:
                common_items_normal = set(user_group_recommendations[u]).intersection(interacted_items_test[u])
                relevance_scores_normal = [1 if i in common_items_normal else 0 for i in user_group_recommendations[u]]
                ndcg_normal = evaluation.ndcg_at_k(relevance_scores_normal[:top_k], top_k)
                ndcg_per_user_type.append(ndcg_normal)
            ndcg_user_group = np.mean(ndcg_per_user_type)
            coverage_per_user_group = evaluation.catalog_coverage(list(user_group_recommendations.values()), ununique_items)
            average_total_dpf_user_group, normalized_dpf_user_group, coverage_category_I1_user_group, coverage_category_I2_user_group, normalized_coverage_category_I1_user_group, normalized_coverage_category_I2_user_group = evaluation.DPF(
                user_group_recommendations, category_mapping)
            return {
                f'{group_name} Novelty Normal': novelty_normal,
                f'{group_name} Novelty Calibrated': novelty_calibrated,
                #f'{group_name} Diversity Normal': diversity_normal,
                #f'{group_name} Diversity Calibrated': diversity_calibrated,
                #f'{group_name} Serendipity Normal': serendipity_normal,
                #f'{group_name} Serendipity Calibrated': serendipity_calibrated,
                f'{group_name} MRR Normal': mrr_normal,
                f'{group_name} MRR Calibrated': mrr_calibrated,
                f'{group_name} PL Normal': pl_normal,
                f'{group_name} PL Calibrated': pl_calibrated,
                f'{group_name} KLD Normal': kld_normal,
                f'{group_name} KLD Calibrated': kld_calibrated,
                f'{group_name} NCDG Normal': ndcg_user_group,
                f'{group_name} coverage': coverage_per_user_group,
                f'{group_name} DPF': average_total_dpf_user_group,
                f'{group_name} normalized_DPF': normalized_dpf_user_group,
                f'{group_name} coverage_category_I1': coverage_category_I1_user_group,
                f'{group_name} coverage_category_I2': coverage_category_I2_user_group,
                f'{group_name} normalized_coverage_category_I1': normalized_coverage_category_I1_user_group,
                f'{group_name} normalized_coverage_category_I2': normalized_coverage_category_I2_user_group,

            }
        all_metrics = {}
        for group_name, user_group in user_groups.items():
            group_metrics = calculate_group_metrics(group_name, user_group, recommendations, calibrated_recommendations, train_data, item_mapping, test_user_item_dict, interacted_items_test, model)
            print(f"{group_name}: {group_metrics}")
            if group_metrics:
                all_metrics.update(group_metrics)
        if args.DPSGD == 'True':
            epsilon_value= privacy_epsilons[-1]
        else:
            epsilon_value=0

        print("Saving to CSV file")

        all_metrics.update({
            'Learning Rate': initial_learning_rate,
            'learning_rate_factor':learning_rate_factor,
            'batch_size':batch_size,
            'dropout':dropout,
            'max_grad_norm': max_grad_norm,
            'noise_multiplier':noise_multiplier,
            'latent_dim': mf_dim,
            'learning_rate_factor':learning_rate_factor,
            'mlp_layer_sizes':mlp_layer_sizes,
            'num_epochs':num_epochs,
            'step_size':step_size,
            'gamma':gamma,
            'delta':delta,
            'Test Losses': test_losses[-1],
            'Train Losses': train_losses[-1],
            'KLD_test':KLD_test[-1],
            'NDCG_test':test_ndcg[-1],
            'recall_test':recall_test[-1],
            'Total MRR': MRR,
            'Calibrated_MRR':Calibrated_MRR,
            'NDCG_normal': NDCG_normal,
            'Calibrated_NDCG':Calibrated_NDCG,
            'average_recall':average_recall,
            'average_recall_calibrated':average_recall_calibrated,
            'KLD_Normal':KLDN,
            'KLD_Calibrated':KLDC,
            'normal_PL':normal_PL,
            'Calibrated_PL':Calibrated_PL,
            'novelty_normal': novelty_normal,
            'Calibrated_novelty':Calibrated_novelty,
            #'Diversity_normal': Diversity_normal,
            #'Diversity_Calibrated': Diversity_Calibrated,
            #'Serendepity_normal': Serendepity_normal,
            #'Serendepity_Calibrated': Serendepity_Calibrated,
            'Coverage':coverage,
            'exposure_I1':exposure_I1,
            'exposure_I2':exposure_I2,
            'DPF':average_total_dpf,
            'normalized_DPF':normalized_dpf,
            'normalized_exposure_1':normalized_exposure_1,
            'normalized_exposure_2':normalized_exposure_2,
            'PL_per_category': PL_per_category,
            'kld_pre_category':kld_pre_category,
            'ndcg_per_category':ndcg_per_category,
            'model_name': f"{model_name}_{ratio}",
            'epsilon': epsilon_value,
        })

        results_df = pd.DataFrame([all_metrics])
        # Define the file path for the results file
        results_file = f'Deep_Yelp_DPSGD.csv'
        # Check if the results file already exists
        if os.path.exists(results_file):
            # File exists, append the current results
            with open(results_file, 'a') as f:
                results_df.to_csv(f, header=False, index=False)
        else:
            # File does not exist, create a new file and write the results
            results_df.to_csv(results_file, index=False)
        print("saving has finished")

    evaluation_and_save()



"""print(f"niche_user={niche_user}")
print(f"blockbuster={blockbuster}")
print(f"diverse={diverse}")
print(f"new={new}")

niche_user_test=[]
blockbuster_test=[]
diverse_test=[]
new_test=[]
for user_id in test_user_item_dict:
    if user_id in niche_user:
        niche_user_test.append(user_id)
    elif user_id in blockbuster:
        blockbuster_test.append(user_id)
    elif user_id in diverse:
        diverse_test.append(user_id)
    else:
        new_test.append(user_id)

print("niche_user_test=",niche_user_test)
print("blockbuster_test=", blockbuster_test)
print("diverse_test=",diverse_test)
print("new_test=",new_test)
"""
"""
PL_per_category:{'I2': -0.3454058515582291, 'I1': -0.11741239852093369}
kld_pre_category:{'I2': nan, 'I1': nan}
ndcg_per_category:{'I2': 0.0, 'I1': 0.0}
"""

#Why KLD is negative? epsilon in compute_kl_divergence
if __name__ == '__main__':
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Run NCF model with different parameters.')

    # Add arguments to the parser
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate value')
    parser.add_argument('--max_grad_norm', default=2, type=float, help='max_grad_norm value')
    parser.add_argument('--mlp_layer_sizes', type=str, help='MLP layer sizes separated by commas')
    parser.add_argument('--DPSGD', type=str, default='False', help='True or False for DPSGD')
    parser.add_argument('--noise_multiplier', type=float, default=1, help='noise_multiplier value')
    parser.add_argument('--mf_dim', type=int, default=8, help='mf_dim value')
    parser.add_argument('--num_epochs', type=int, default=1, help='num_epochs')
    parser.add_argument('--Plot', type=bool, default=False, help='True or False for Plotting')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--delta', type=float, default=1e-4, help='delta')
    parser.add_argument('--k', type=int, default=10, help='k')
    parser.add_argument('--calibration', type=int, default=0, help='True or False for calibration (0 or 1)')
    parser.add_argument('--mlp_deep_fm_should_be_trained', type=str, default="False",
                        help='True or False for mlp_deep_fm_should_be_trained')
    parser.add_argument('--filter', type=bool, default=False, help='filter')


    # Parse the arguments
    args = parser.parse_args()
    train(args)
    print("Done")
