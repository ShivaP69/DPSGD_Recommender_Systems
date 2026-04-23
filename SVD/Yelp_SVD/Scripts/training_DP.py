import numpy as np  # linear algebra
# from pandas.io.json import json_normalize
import json
import requests
from torch.utils.data import DataLoader, TensorDataset
from bs4 import BeautifulSoup
from ItemMapping import create_item_mapping
import DP_DataProcess
import ast
import Evaluation
import torch
import models
import pandas as pd
import Preprocess
import torch.nn as nn
import Calibration_Func
import torch.optim as optim
import train_test_split
import os
import argparse
from DP_Code import apply_dp
import pickle

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def predict_score(user_ids, item_ids, model):
    user_embed = model.user_embeddings(user_ids)
    item_embed = model.item_embeddings(item_ids)
    scores = torch.sum(user_embed * item_embed, dim=1)
    return scores


model_path_template = "model_{noise_multiplier}_{epoch}.pth"
# Function to save model
def save_model(model, optimizer, epoch, loss, model_name):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, f"{model_name}_checkpoint.pth")
    print(f"Model saved as {model_name}_checkpoint.pth")


# Loading the model and optimizer
def load_or_initialize_model(model, optimizer, model_name,load_model=False):
    if load_model:
        checkpoint_path = f"{model_name}_checkpoint.pth"
        if os.path.isfile(checkpoint_path):
            # If checkpoint exists, load it
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Model loaded from {checkpoint_path}, starting from epoch {epoch + 1} with loss {loss}")
            return model, optimizer, epoch + 1, loss  # Start from the next epoch
        else:
            # If no checkpoint exists, start fresh
            print(f"No saved model found at {checkpoint_path}, starting from scratch.")
            return model, optimizer, 0, float("inf")  # Start from epoch 0 and set best loss to infinity
    else:
        return model, optimizer, 0, float("inf")  # Start from epoch 0 and set best loss to infinity


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

        df_business_id.to_csv('df_business_id.csv', index=False)

        full_data_bussiness = pd.merge(df, df_business_id, on='business_id', how='inner')

        df_item = full_data_bussiness[full_data_bussiness['categories'].apply(lambda x: len(x) > 0)]
        df_item.to_csv('df_item.csv', index=False)

    return df_item,df_business_id,result_dict


def read_data(df_business_id):
    data_file_name = 'data.csv'
    if os.path.isfile(data_file_name):
        data = pd.read_csv(data_file_name)
    else:
        # url = 'https://raw.githubusercontent.com/knowitall/yelp-dataset-challenge/master/data/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json'
        url = 'yelp_academic_dataset_review.json'
        user_df = pd.read_json(url, lines=True)
        # user_df= pd.read_json("yelp_academic_dataset_review.json", lines=True)
        sorted_user_ids = sorted(user_df['user_id'].unique())  # it should be unique
        # map each uset id to a numerical value
        users_mapping = {user_id: idx for idx, user_id in enumerate(sorted_user_ids)}
        df_users_id = pd.DataFrame(list(users_mapping.items()), columns=['user_id', 'user_idx'])
        # print(df_users_id)

        user_df = pd.merge(user_df, df_users_id, on='user_id', how='inner')
        valid_id = df_business_id['business_id'].tolist()
        user_df = user_df[user_df['business_id'].isin(valid_id)]
        data = pd.merge(user_df, df_business_id, on='business_id', how='inner')
        data.to_csv(data_file_name, index=False)

    return data

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

        train_data, test_data = train_test_split.train_test_split_version1(data)
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


def eval_model(model,test_dataloader):
    model.eval()  # Set the model to evaluation mode
    total_loss_test = []
    with torch.no_grad():
        for batch in test_dataloader:
            user_input, item_input, labels = batch
            user_input = user_input.to(device)
            item_input = item_input.to(device)
            labels=labels.to(device)
            predictions = predict_score(user_input, item_input, model)
            test_loss = nn.MSELoss()(predictions, labels.float())
            total_loss_test.append(test_loss.item())
        #print(f"Epoch [{epoch + 1}/{num_epochs}] Test Loss: {np.mean(total_loss_test):.4f}")
    return total_loss_test, np.mean(total_loss_test)


def generate_recommendations(model,test_user_item_dict,user_indices,item_mapping,top_k,interacted_distr,interacted_items_test,final_results=False):
    recommendations = {}
    average_ndcg_normal = []
    num_users = len(test_user_item_dict)
    Accuracy = []
    # AUC = []
    Accuracy_train = []
    # AUC_train = []
    reco_distr = {}
    KLD_normal = []
    reco_distr_calibrated = {}
    KLD_calibrated = []
    item_candidates = {}
    average_ndcg_calibrated=[]
    calibrated_recommendations = {}

    model.eval()
    for u in test_user_item_dict:
        try:
            test_items = user_indices[u]
            test_items = [item_id for item_id in test_items if item_id in item_mapping]
            k = len(test_items)

            # Move tensors to GPU for prediction
            user_ids_tensor = torch.tensor([u] * k).to(device)
            item_ids_tensor = torch.tensor(test_items).to(device)

            # Predict scores on GPU
            with torch.no_grad():
                predicted_scores = predict_score(user_ids_tensor, item_ids_tensor, model).cpu().numpy()

            # Processing results on CPU
            predicted_scores = np.squeeze(predicted_scores)
            top20_indices = np.argsort(-predicted_scores)[:top_k]
            top20_items = [test_items[i] for i in top20_indices]
            recommendations[u] = [item_mapping[item_id] for item_id in top20_items]
            item_candidates[u] = (
                Calibration_Func.generate_item_candidates(item_mapping, test_items, predicted_scores))
            calibrated_recommendations[u] = Calibration_Func.calib_recommend(item_candidates[u], interacted_distr[u],
                                                                        topn=top_k, lmbda=0.8)

            # NDCG calculations
            common_items_normal = set(recommendations[u]).intersection(interacted_items_test[u])
            relevance_scores_normal = [1 if i in common_items_normal else 0 for i in recommendations[u]]
            # print(relevance_scores)
            ndcg_normal = Evaluation.ndcg_at_k(relevance_scores_normal[:top_k], top_k)
            average_ndcg_normal.append(ndcg_normal)
            common_items_calibrated = set(calibrated_recommendations[u]).intersection(interacted_items_test[u])
            relevance_scores_calibrated = [1 if i in common_items_calibrated else 0 for i in calibrated_recommendations[u]]
            ndcg_calibrated = Evaluation.ndcg_at_k(relevance_scores_calibrated[:top_k], top_k)
            average_ndcg_calibrated.append(ndcg_calibrated)
            # all_labels = [1 if item in test_user_item_dict[u] else 0 for item in test_items]
            # all_predictions = [1 if p >= 0.5 else 0 for p in predicted_scores]
            # accuracy = accuracy_score(all_labels, all_predictions)
            # auc = roc_auc_score(all_labels, all_predictions)
            # Accuracy.append(accuracy)
            reco_distr[u] = Calibration_Func.compute_genre_distr(recommendations[u])
            # compute KLD
            KLD_normal.append(Calibration_Func.compute_kl_divergence(interacted_distr[u], reco_distr[u]))
            # for calculating KLD for calibrated recommendations
            reco_distr_calibrated[u] = Calibration_Func.compute_genre_distr(
                calibrated_recommendations[u])
            KLD_calibrated.append(Calibration_Func.compute_kl_divergence(interacted_distr[u],
                                                                         reco_distr_calibrated[u]))
        except Exception as e:
            print(f"[Warning] Skipped user {u} due to error: {e}")
            continue
    KLDN = np.mean(KLD_normal)
    KLDC = np.mean(KLD_calibrated)
    if final_results:
        return recommendations,calibrated_recommendations,average_ndcg_normal,average_ndcg_calibrated,KLDN,KLDC
    else:
        return recommendations, KLDN, average_ndcg_normal


def train(args):

    user_col = 'user_idx'
    item_col = 'business_idx'
    latent_dim = args.latent_dim
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    top_k = args.k
    initial_learning_rate = args.learning_rate
    learning_rate_factor = 0.2
    ratio = 50
    l2_reg_strength = 1e-5
    epochs_without_improvement = 0
    best_test_loss = float("inf")  # Initialize with a large value
    consecutive_epochs_without_improvement = 0
    max_consecutive_epochs_without_improvement = 6  # Adjust as needed
    new_learning_rate = initial_learning_rate

    df_item, df_business_id, result_dict = read_df_item_and_df_business()
    data = read_data(df_business_id)
    item_mapping = create_item_mapping(df_item, 'business_idx', 'name', 'categories')
    num_users, num_items = data[user_col].max() + 1, data[item_col].max() + 1
    all_movieIds = data[item_col].unique()

    # train_loader,test_loader, train_data, test_data, all_movieIds= create_data_loader(data)
    data = Preprocess.preprocess(data)
    print(data.head)
    # read train and test data
    user_indices, train_data, test_data, user_interacted_items, test_user_item_dict = read_train_test_and_user_indices(
        data, all_movieIds)
    print("test_data_size : ", test_data.shape)

    user_interacted_items = train_data.groupby(user_col)[item_col].apply(list).to_dict()  ########## user_interacted_items
    test_user_item_dict = test_data.groupby(user_col)[item_col].apply(list).to_dict()

    for u in user_interacted_items:
        positive_items = user_interacted_items[u]
        new_positive_samples = apply_dp(positive_items, all_movieIds, args.privacy) # Apply LDP
        user_interacted_items[u] = new_positive_samples

    interacted_distr = {}
    for user_id in user_interacted_items:
        interacted = user_interacted_items[user_id]
        interacted = [item_id for item_id in interacted if item_id in item_mapping]
        interacted_items_init = [item_mapping[item_id] for item_id in interacted]
        interacted_distr[user_id] = Calibration_Func.compute_genre_distr(interacted_items_init)

    interacted_items_test = {}
    for u in test_user_item_dict:
        interacted_items_init_test = [item_id for item_id in test_user_item_dict[u] if item_id in item_mapping]
        interacted_items_test[u] = [item_mapping[item_id] for item_id in interacted_items_init_test]

    interacted_items = {}
    for u in user_interacted_items:
        interacted = user_interacted_items[u]
        interacted = [item_id for item_id in interacted if item_id in item_mapping]
        interacted_items[u] = [item_mapping[item_id] for item_id in interacted]


    # defining DataLoader for both train and test data
    train_dataloader = DataLoader(DP_DataProcess.MovieLensTrainDataset(user_interacted_items, all_movieIds), batch_size=batch_size,
                                  num_workers=1)
    test_dataloader = DataLoader(DP_DataProcess.MovieLensTrainDataset(test_user_item_dict, all_movieIds), batch_size=batch_size,
                                 num_workers=1)
    # defining the model
    model = models.SVD(num_users, num_items, latent_dim)
    # Choose an optimizer (e.g., Adam)
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    model_name = model_name = f"SVD_{ratio}_{args.privacy}_{batch_size}_Yelp"
    model, optimizer, start_epoch, best_test_loss = load_or_initialize_model(model, optimizer, model_name,args.load_model)

    train_losses = []
    test_losses = []
    privacy_epsilons = []
    test_ndcg = []
    # accuracy_test=[]
    KLD_test = []
    recall_test = []
    # recall_train=[]
    print("start training ...")
    model.to(device)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        losses = []
        for batch in train_dataloader:
            user_input, item_input, labels = batch
            user_input=user_input.to(device)
            item_input=item_input.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predictions = predict_score(user_input, item_input, model)
            # loss = bpr_loss(pos_scores, neg_scores)+ model.weight_decay * regularization_loss(model)
            loss  = nn.MSELoss()(predictions, labels.float())
            # Add L2 regularization to the loss
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_reg_strength * l2_reg
            loss.backward()
            losses.append(loss)
            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            # total_loss += loss.item()
            optimizer.zero_grad()

        values = [tensor.item() for tensor in losses]
        print(
            f"\tTrain Epoch: [{epoch + 1}/{num_epochs}] \t"
            f"Train Loss: {np.mean(values):.6f} "
        )

        train_losses.append(np.mean(values))
        # Validation
        print("start validation")
        total_loss_test, mean_loss= eval_model(model, test_dataloader)
        test_losses.append(mean_loss)  # keep mean loss to adjust learning rate
        # Generate recommendations for test users
        print("Generating recommendations")
        recommendations, KLD, average_ndcg_normal = generate_recommendations(model, test_user_item_dict,
                                                                                           user_indices,
                                                                                           item_mapping, top_k,
                                                                                           interacted_distr,
                                                                                           interacted_items_test)
        recall_test.append(Evaluation.calculate_total_average_recall(interacted_items_test, recommendations))
        # AUC.append(auc)
        KLD_test.append(np.mean(KLD))
        test_ndcg.append(np.mean(average_ndcg_normal))

        # stop training when there is no improvement
        if np.mean(total_loss_test) < best_test_loss:
            best_test_loss = np.mean(total_loss_test)
            consecutive_epochs_without_improvement = 0  # reset
        else:
            consecutive_epochs_without_improvement += 1

        if consecutive_epochs_without_improvement >= max_consecutive_epochs_without_improvement:
            print(
                f"Stopping training due to lack of improvement for {max_consecutive_epochs_without_improvement} epochs.")
            break  # Exit the training loop

        # Adjust learning rate
        if epoch > 0 and test_losses[-1] >= test_losses[-2]:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 4:  # Adjust the number as needed.
                # Increase the learning rate
                new_learning_rate = new_learning_rate * learning_rate_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_learning_rate
                print(f"Changing learning rate to {new_learning_rate}")
                epochs_without_improvement = 0  # Reset the counter

        if args.save_model:
            if (epoch + 1) % 50 == 0:
                save_model(model,optimizer, epoch, loss, model_name)

    print(f"test_losses= {test_losses}")
    print(f"train_losses= {train_losses}")
    print(f"KLD_test= {KLD_test}")
    print(f"NDCG_test ={test_ndcg}")
    print(f"recall_test= {recall_test}")

    def evaluation_and_save():
        item_col = 'business_idx'
        # genres_df= Evaluation.genres_features(df_item,result_dict)
        unique_ids = list(train_data[item_col].unique())
        ununique_ids_valid = [item_id for item_id in unique_ids if item_id in item_mapping]
        ununique_items = [item_mapping[item_id] for item_id in ununique_ids_valid]
        # print(f"ununique_items:{ununique_items}")
        # call generate_Recommendations
        recommendations, calibrated_recommendations, average_ndcg_normal, average_ndcg_calibrated, KLDN, KLDC = generate_recommendations(
            model, test_user_item_dict, user_indices, item_mapping, top_k, interacted_distr, interacted_items_test,
            final_results=True)
        print("End of the test recommendations creation")


        # MRR
        MRR = Evaluation.MRR(recommendations, interacted_items_test)
        print(f"MRR in normal mode: {MRR}")
        Calibrated_MRR = Evaluation.MRR(calibrated_recommendations, interacted_items_test)
        print(f"MRR in calibrated mode: {Calibrated_MRR}")
        #NDCG
        NDCG = np.mean(average_ndcg_normal)
        Calibrated_NDCG = np.mean(average_ndcg_calibrated)
        print(f"Average NDCG in normal mode @{top_k}: {NDCG:.4f}")
        print(f"Average NDCG in Calibarated mode @{top_k}: {Calibrated_NDCG:.4f}")
        #recall
        average_recall = Evaluation.calculate_total_average_recall(interacted_items_test, recommendations)
        print(f"Total Average Recall in normal state: {average_recall:.4f}")
        average_recall_calibrated = Evaluation.calculate_total_average_recall(interacted_items_test, calibrated_recommendations)
        print(f"Total Average Recall in calibrated state: {average_recall_calibrated:.4f}")
        #PL
        normal_PL = Evaluation.PL(train_data, item_mapping, recommendations, list(recommendations.keys()), model)
        print(f"PL normal: {normal_PL}")
        Calibrated_PL = Evaluation.PL(train_data, item_mapping, calibrated_recommendations,
                                      list(calibrated_recommendations.keys()), model)
        print(f"PL calibrated: {Calibrated_PL}")
        # nivelty
        novelty_normal = Evaluation.novelty(recommendations, train_data, item_mapping,top_k)
        print("novelty normal :", novelty_normal)
        Calibrated_novelty = Evaluation.novelty(calibrated_recommendations, train_data, item_mapping,top_k)
        print("novelty calibrated:", Calibrated_novelty)
        # coverage
        coverage = Evaluation.catalog_coverage(list(recommendations.values()), ununique_items)
        print(f"coverage:{coverage}")
        """Diversity_normal = Evaluation.Diversity(recommendations, genres_df)
        print(f" Diversity normal : {Diversity_normal}")
        Diversity_Calibrated = Evaluation.Diversity(calibrated_recommendations, genres_df)
        print(f" Diversity Calibrated : {Diversity_Calibrated}")
        # for all users ( we can substitute calibrate recommendations with recommnedations and vice versa)
        Serendepity_normal = Evaluation.serendepity_group(list(recommendations.keys()), test_user_item_dict, item_mapping,
                                                          recommendations, genres_df)
        print(f"Total Average Serendepity in normal model {Serendepity_normal}")
        Serendepity_Calibrated = Evaluation.serendepity_group(list(calibrated_recommendations.keys()), test_user_item_dict,
                                                              item_mapping, calibrated_recommendations, genres_df)
        print(f"Total Average Serendepity in calibrated model{Serendepity_Calibrated}")"""

        popular = Evaluation.PopularItems(train_data, item_mapping)
        category_mapping = Evaluation.categories(train_data, item_mapping)

        valid_interacted, valid_reco_distr = Evaluation.valid_distr_extraction(category_mapping, recommendations,
                                                                               interacted_items)

        PL_per_category = Evaluation.PL_items(train_data, item_mapping, recommendations, list(recommendations.keys()),
                                              model, category_mapping)
        kld_pre_category = Evaluation.calculate_KLD_items(recommendations, valid_reco_distr,
                                                          valid_interacted)
        ndcg_per_category = Evaluation.calculate_ndcg_items(category_mapping, recommendations, interacted_items_test,
                                                            top_k)

        average_total_dpf, normalized_dpf, exposure_I1, exposure_I2, normalized_exposure_1, normalized_exposure_2 = Evaluation.DPF(
            recommendations,
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
        niche_user, blockbuster, diverse, new = Evaluation.type_of_user_total(item_mapping, popular, train_data)
        # user_groups= [niche_user, blockbuster,diverse, new]
        user_groups = {
            "niche_user": niche_user,
            "blockbuster": blockbuster,
            "diverse": diverse,
            "new": new
        }

        def calculate_group_metrics(group_name, user_group, recommendations, calibrated_recommendations, train_data,
                                    item_mapping, test_user_item_dict, interacted_items_test, model):
            if len(user_group) == 0:
                return None

            user_group_recommendations = {u: recommendations[u] for u in recommendations if u in user_group}
            user_group_recommendations_calibrated = {u: calibrated_recommendations[u] for u in calibrated_recommendations if
                                                     u in user_group}

            # Calculate metrics
            novelty_normal = Evaluation.novelty(user_group_recommendations, train_data, item_mapping,top_k)
            novelty_calibrated = Evaluation.novelty(user_group_recommendations_calibrated, train_data, item_mapping,top_k)
            """diversity_normal = Evaluation.Diversity(user_group_recommendations, genres_df)
            diversity_calibrated = Evaluation.Diversity(user_group_recommendations_calibrated, genres_df)
            serendipity_normal = Evaluation.serendepity_group(user_group, test_user_item_dict, item_mapping,
                                                              user_group_recommendations, genres_df)
            serendipity_calibrated = Evaluation.serendepity_group(user_group, test_user_item_dict, item_mapping,
                                                                  user_group_recommendations_calibrated, genres_df)"""
            mrr_normal = Evaluation.MRR(user_group_recommendations, interacted_items_test)
            mrr_calibrated = Evaluation.MRR(user_group_recommendations_calibrated, interacted_items_test)
            pl_normal = Evaluation.PL(train_data, item_mapping, user_group_recommendations, user_group, model)
            pl_calibrated = Evaluation.PL(train_data, item_mapping, user_group_recommendations_calibrated, user_group, model)

            # Calculate KLD
            kld_normal = np.mean([Calibration_Func.compute_kl_divergence(interacted_distr[user_id],
                                                                         Calibration_Func.compute_genre_distr(
                                                                             user_group_recommendations[user_id])) for user_id
                                  in user_group if user_id in user_group_recommendations])
            kld_calibrated = np.mean([Calibration_Func.compute_kl_divergence(interacted_distr[user_id],
                                                                             Calibration_Func.compute_genre_distr(
                                                                                 user_group_recommendations_calibrated[
                                                                                     user_id])) for user_id in user_group if
                                      user_id in user_group_recommendations_calibrated])
            ndcg_per_user_type = []
            for u in user_group_recommendations:
                common_items_normal = set(user_group_recommendations[u]).intersection(interacted_items_test[u])
                relevance_scores_normal = [1 if i in common_items_normal else 0 for i in user_group_recommendations[u]]
                ndcg_normal = Evaluation.ndcg_at_k(relevance_scores_normal[:top_k], top_k)
                ndcg_per_user_type.append(ndcg_normal)
            ndcg_user_group = np.mean(ndcg_per_user_type)
            coverage_per_user_group = Evaluation.catalog_coverage(list(user_group_recommendations.values()),
                                                                  ununique_items)
            average_total_dpf_user_group, normalized_dpf_user_group, coverage_category_I1_user_group, coverage_category_I2_user_group, normalized_coverage_category_I1_user_group, normalized_coverage_category_I2_user_group = Evaluation.DPF(
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
            group_metrics = calculate_group_metrics(group_name, user_group, recommendations, calibrated_recommendations,
                                                    train_data, item_mapping, test_user_item_dict, interacted_items_test,
                                                    model)
            if group_metrics:
                all_metrics.update(group_metrics)

        all_metrics.update({
            'Learning Rate': initial_learning_rate,
            'learning_rate_factor': learning_rate_factor,
            'batch_size': batch_size,
            'dropout': 'None',
            'latent_dim': latent_dim,
            'mlp_layer_sizes': 'None',
            'num_epochs': num_epochs,
            'step_size': 'None',
            'gamma': 'None',
            'Test Losses': test_losses[-1],
            'Train Losses': train_losses[-1],
            'KLD_test': KLD_test[-1],
            'NDCG_test': test_ndcg[-1],
            'recall_test': recall_test[-1],
            'Total MRR': MRR,
            'Calibrated_MRR': Calibrated_MRR,
            'NDCG_normal': NDCG,
            'Calibrated_NDCG': Calibrated_NDCG,
            'average_recall': average_recall,
            'average_recall_calibrated': average_recall_calibrated,
            'KLD_Normal': KLDN,
            'KLD_Calibrated': KLDC,
            'normal_PL': normal_PL,
            'Calibrated_PL': Calibrated_PL,
            'novelty_normal': novelty_normal,
            'Calibrated_novelty': Calibrated_novelty,
            #'Diversity_normal': Diversity_normal,
            #'Diversity_Calibrated': Diversity_Calibrated,
            #'Serendepity_normal': Serendepity_normal,
            #'Serendepity_Calibrated': Serendepity_Calibrated,
            'Coverage': coverage,
            'exposure_I1': exposure_I1,
            'exposure_I2': exposure_I2,
            'DPF': average_total_dpf,
            'normalized_DPF': normalized_dpf,
            'normalized_exposure_1': normalized_exposure_1,
            'normalized_exposure_2': normalized_exposure_2,
            'PL_per_category': PL_per_category,
            'kld_pre_category': kld_pre_category,
            'ndcg_per_category': ndcg_per_category,
            'epsilon': args.privacy,
            'model_name': f"{model_name}_{ratio}"
        })

        results_df = pd.DataFrame([all_metrics])
        # Define the file path for the results file
        results_file = 'SVD_Yelp_LDP.csv'
        # Check if the results file already exists
        if os.path.isfile(results_file):
            # File exists, append the current results
            with open(results_file, 'a') as f:
                results_df.to_csv(f, header=False, index=False)
        else:
            # File does not exist, create a new file and write the results
            results_df.to_csv(results_file, index=False)

    evaluation_and_save()


if __name__ == '__main__':
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Run BPR model with different parameters.')
    # Add arguments to the parser
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate value')
    parser.add_argument('--num_epochs', type=int, default=400, help='epochs')
    parser.add_argument('--privacy', type=float, default=0.1, help='privacy')
    parser.add_argument('--latent_dim', type=int, default=5, help='latent_dim')
    parser.add_argument('--load_model', type=bool, default=False, help='load_model')
    parser.add_argument('--save_model', type=bool, default=False, help='save_model')
    parser.add_argument('--k', type=int, default=10, help='k')
    parser.add_argument('--calibration', type=int, default=0, help='True or False for calibration (0 or 1)')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    # Parse the arguments
    args = parser.parse_args()
    train(args)