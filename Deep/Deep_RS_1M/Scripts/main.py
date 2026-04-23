import pickle
import pandas as pd
from sklearn.metrics import recall_score
import numpy as np
from MF_MLP import NeuMF
import torch.nn.init as init
from MF_MLP import DeepFM
from MF_MLP import MF
from DataProcess import MovieLensTrainDataset
from torch.optim.lr_scheduler import StepLR
import torch
from MF_MLP import calculate_loss
import torch.nn as nn
from opacus import PrivacyEngine
from Preprocess import preprocess
from train_test_split import train_test_split_version1
from torch.utils.data import DataLoader, RandomSampler
import seaborn as sns
from ItemMapping import create_item_mapping
import json
from csv import writer
import calibration_Func
import evaluation
import matplotlib.pyplot as plt
import argparse
torch.multiprocessing.set_sharing_strategy('file_system')
import sys
from Plot import plotting
import pickle
import os
import pytorch_lightning as pl

#writer = SummaryWriter(log_dir='logs')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_deep_fm_mf(model,mf_optimizer,train_dataloader,val_dataloader,mf_learning_rate,learning_rate_factor,number_epoch=5):
    """
    Train the MLP/Train GMF model
    This function is the implementation of the idea proposed in the NeuMF paper, to train MF/DeepFM : initialize NeuMF using the pretrained models of GMF and MLP
    """
    total_loss_train_fm = []
    num_epochs = number_epoch
    test_losses = []
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            user_input, item_input, labels = batch
            mf_optimizer.zero_grad()
            predictions = model(user_input, item_input)
            # loss = torch.nn.functional.binary_cross_entropy(predictions, labels.view(-1, 1).float())
            loss = model.calculate_loss(predictions, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            mf_optimizer.step()
            total_loss_train_fm.append(loss.item())
        print(f" train loss {model} : {np.mean(total_loss_train_fm)}")
        # print(f"last train loss:{loss.item()}")

        # evaluate the model
        model.eval()
        total_loss_test_fm = []
        with torch.no_grad():
            for batch in val_dataloader:
                user_input, item_input, labels = batch
                predictions = model(user_input, item_input)
                loss = torch.nn.functional.binary_cross_entropy(predictions, labels.view(-1, 1).float())
                total_loss_test_fm.append(loss.item())
        print((f' test Loss {model}: {np.mean(total_loss_test_fm):.5f}'))
        test_losses.append(np.mean(total_loss_test_fm))
        # regulate the learning rate
        if epoch > 0 and test_losses[-1] >= test_losses[-2]:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 3:  # Adjust the number as needed.
                # Increase the learning rate
                new_learning_rate = mf_learning_rate * learning_rate_factor
                for param_group in mf_optimizer.param_groups:
                    param_group['lr'] = new_learning_rate
                print(f"Increasing learning rate to {new_learning_rate}")
                epochs_without_improvement = 0  # Reset the counter
        if epoch > 0 and (test_losses[-1] < test_losses[-2]) and (
                epochs_without_improvement > 0):  # prevent the summation of non-sequential non-improvements
            epochs_without_improvement = 0
    return model


def deep_fm_mf(num_users,num_items,train_data,all_movieIds,train_dataloader,val_dataloader,learning_rate_factor):
    # DeepFM: MLP
    # Create instances of the DeepFM
    deepFM = DeepFM(num_users, num_items, mlp_layer_sizes=[16, 8, 4])
    # Define optimizers
    deepFM_learning_rate= 0.0001
    deepfm_optimizer = torch.optim.Adam(deepFM.parameters(), lr= deepFM_learning_rate)

    # Create instances of the MF:GMF
    mf_model = MF(num_users, num_items, mf_dim=8)
    mf_learning_rate=0.0005
    mf_optimizer =torch.optim.Adam(mf_model.parameters(), lr=mf_learning_rate)

    deepFM=train_deep_fm_mf(deepFM, deepfm_optimizer, train_dataloader, val_dataloader, deepFM_learning_rate, learning_rate_factor,
                     number_epoch=2)
    mf_model = train_deep_fm_mf(mf_model, mf_optimizer, train_dataloader, val_dataloader, mf_learning_rate,
                              learning_rate_factor,
                              number_epoch=5)
    return deepFM,mf_model



def save_categories_to_pickle(train_data, item_mapping, filename):
    category_mapping = evaluation.categories(train_data, item_mapping)
    with open(filename, 'wb') as file:
        pickle.dump(category_mapping, file) # category_mapping.pkl


def read_files():
    title_col = 'title'
    genre_col = 'genres'
    item_col = 'movieId'
    df_item = pd.read_csv('movie.csv') # all of the movies even those ones that are not essentially in ratings file
    df_item = df_item[
        df_item[genre_col] != '(no genres listed)']  # eliminate movies that had no genre information attached
    print('dimension: ', df_item.shape)
    #df,_,_= read_data_ml100k()
    """file_path = 'u.data'
    names = ['userId', 'movieId', 'rating', 'timestamp']
    df = pd.read_csv(file_path, sep='\t', names=names)"""

    # make a df based on the rating.dat files
    df_path='df.csv'
    if os.path.isfile(df_path):
        df= pd.read_csv(df_path)
        print(f"df shape:{df.shape}")
    else:
        names = ['userId', 'movieId', 'rating', 'timestamp'] # the columns that we need (timestamp can be removed)
        file_path = 'ratings.dat'
        df = pd.read_table(file_path, names=names, sep="::", engine='python')
        print(f"pre_preprocess:{df.shape}")
        df = preprocess(df) # applying filters
        print(f"post_preprocess:{df.shape}")
        print(len(np.unique(list(df['userId']))))
        print(len(np.unique(list(df['movieId']))))
        print(df.head)
        df.to_csv(df_path, index=False)
    #print(f"before {df_item.shape}")
    valid_ids = df['movieId'].unique()
    df_item = df_item[df_item['movieId'].isin(valid_ids)] # just keep those ones that are also in the rating
    #print(f"after:{df_item.shape}")
    item_mapping = create_item_mapping(df_item, item_col, title_col, genre_col)
    return df_item,item_mapping,df

def read_train_test(df):
    if os.path.isfile('train_data.csv') and os.path.isfile('test_data.csv'):
        train_data = pd.read_csv('train_data.csv')
        test_data = pd.read_csv('test_data.csv')
    else:

        train_data, test_data= train_test_split_version1(df)
        train_data.to_csv('train_data.csv', index=False)
        test_data.to_csv('test_data.csv', index=False)
    return train_data, test_data

def read_user_indices(test_user_item_dict,user_interacted_items, all_movieIds,ratio):# ratio percentage of data is considered to be recommended by recommender system which also includes test data
    user_indices = {}
    indices_file_path = f'indices_file_path_{ratio}.pkl'
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
            test_items = selected_not_interacted + list(
                interacted_test)  # a combinition of test items and random items
            test_items = list(set(test_items))
            user_indices[u] = test_items
        with open(indices_file_path, 'wb') as file:
            pickle.dump(user_indices, file)
    return user_indices


def plot(df,item_mapping):
    popular = evaluation.PopularItems(df, item_mapping)
    niche_user, blockbuster, diverse, new = evaluation.type_of_user_total(item_mapping, popular, df)
    # user_groups= [niche_user, blockbuster,diverse, new]
    user_groups = {
        "niche": niche_user,
        "blockbuster": blockbuster,
        "diverse": diverse,
        # "new": new
    }
    plotting(user_groups)
    total_users = sum(len(users) for users in user_groups.values())
    items_popularity = evaluation.popularity_id(df, item_mapping)
    # plot_popularity(items_popularity)
    popularity_df = pd.DataFrame(list(items_popularity.items()), columns=['Movie Title', 'Popularity'])
    popularity_df.to_csv('popularity.csv', index=False)
    sys.exit()  # This code is just used when we need to reproduce the distr figure in the paper, so after having the plot, we can exit


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
            calibrated_recommendations[u] = calibration_Func.calib_recommend(item_candidates[u], interacted_distr[u],
                                                                             topn=top_k, lmbda=0.8)
            common_items_normal = set(recommendations[u]).intersection(interacted_items_test[u])
            relevance_scores_normal = [1 if i in common_items_normal else 0 for i in recommendations[u]]
            common_items_calibrated = set(calibrated_recommendations[u]).intersection(interacted_items_test[u])
            relevance_scores_calibrated = [1 if i in common_items_calibrated else 0 for i in calibrated_recommendations[u]]

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
            KLD_normal.append(calibration_Func.compute_kl_divergence(interacted_distr[u], reco_distr[u], score=1))
            KLD_score_zero.append(calibration_Func.compute_kl_divergence(interacted_distr[u], reco_distr[u], score=0))

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



def train(args):

    num_epochs=args.num_epochs
    noise_multiplier = args.noise_multiplier
    delta = args.delta
    top_k=args.k
    Calibration=args.calibration
    initial_learning_rate = args.learning_rate
    learning_rate_factor = 0.2
    batch_size = args.batch_size
    mf_dim = args.mf_dim
    dropout = 0.4
    ratio=50
    # read all necessary files
    df_item,item_mapping,df= read_files()
    all_movieIds = df['movieId'].unique()
    num_users = df['userId'].max() + 1
    num_items = df['movieId'].max() + 1
    print("num_users",len(np.unique(df['userId'].tolist())))
    print("num_items", len(np.unique(df['movieId'].tolist())))
    num_users = int(num_users)
    num_items = int(num_items)
    print("Number of users for model structure: ", num_users)
    print("Number of items for model structure: ", num_items)

    if args.Plot: # to check the distribution of different user types
        plot(df,item_mapping)

    train_data, test_data=read_train_test(df) #read train test data
    print("test_data:",test_data.shape)
    print("train_data",train_data.shape)

    # User interacted data based on training data: extract the interacted items per user
    user_interacted_items = train_data.groupby('userId')['movieId'].apply(list).to_dict()

    # extracting the interacted items for users in the test data (the item ids for each user_id)
    test_user_item_dict = test_data.groupby('userId')['movieId'].apply(list).to_dict()

    # extract the name of the items in the user history for both train and test data
    interacted_items_test = {}
    for u in test_user_item_dict:
        interacted_items_t = [item_id for item_id in test_user_item_dict[u] if item_id in item_mapping] # keep just those items that we have its information in item mapping
        interacted_items_test[u] = [item_mapping[item_id] for item_id in interacted_items_t]

    interacted_distr = {}
    for user_id in user_interacted_items:
        interacted = user_interacted_items[user_id]
        interacted = [item_id for item_id in interacted if item_id in item_mapping]
        interacted_items_train= [item_mapping[item_id] for item_id in interacted]
        interacted_distr[user_id] = calibration_Func.compute_genre_distr(interacted_items_train) # calculate the genre distribution for items in the user history

    interacted_items = {}
    for u in user_interacted_items:
        interacted = user_interacted_items[u]
        interacted = [item_id for item_id in interacted if item_id in item_mapping]
        interacted_items[u] = [item_mapping[item_id] for item_id in interacted]

    # reading user indices: these are the items that our RS use to recommend the new items to the user
    user_indices= read_user_indices(test_user_item_dict,user_interacted_items, all_movieIds, ratio)

    # DataLoader for train and test data with determined batch size
    train_dataloader = DataLoader(MovieLensTrainDataset(train_data,all_movieIds), batch_size=batch_size, num_workers=2)
    val_dataloader = DataLoader(MovieLensTrainDataset(test_data,all_movieIds), batch_size=batch_size, num_workers=2)
    # defining the model

    # Define the MLP layer sizes
    mlp_layer_sizes = args.mlp_layer_sizes  # Example sizes, modify as needed
    mlp_layer_sizes = [int(x) for x in mlp_layer_sizes.split(',')]
    model = NeuMF(num_users, num_items, mf_dim=mf_dim, mlp_layer_sizes=mlp_layer_sizes, dropout=dropout)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    #criterion = nn.BCELoss()


    if args.mlp_deep_fm_should_be_trained=="True": # I would recommend False: this code pre-train MF and Deep FM of NCF
        deepFM,mf_model= deep_fm_mf(num_users, num_items, train_data, all_movieIds, train_dataloader, val_dataloader,learning_rate_factor)
        #user_embedding_weights_deepFM = deepFM.user_embedding.weight.data.clone()
        #item_embedding_weights_deepFM = deepFM.item_embedding.weight.data.clone()
        #model.mlp_user_embed.weight.data = user_embedding_weights_deepFM
        #model.mlp_item_embed.weight.data = item_embedding_weights_deepFM
        #user_embedding_weights_MF = mf_model.user_emb.weight.data.clone()
        #item_embedding_weights_MF = mf_model.item_emb.weight.data.clone()
        #model.mf_user_embed.weight.data = user_embedding_weights_MF
        #model.mf_item_embed.weight.data = item_embedding_weights_MF
        model.replace_mlp_weights(deepFM)
        model.replace_mf_weights(mf_model)


    # setting the configuration for applying DPSGD
    if args.DPSGD=='True':
        model_name= 'DPSGD_Deep_1M'
        delta = 1e-5
        max_grad_norm = args.max_grad_norm
        noise_multiplier=args.noise_multiplier
        # activate the following codes in case of using target epsilon
        """privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            target_epsilon=1,
            target_delta=delta,
            epochs=num_epochs,
            max_grad_norm=max_grad_norm, )"""
        # When the epsilon should be calculated by Opacus
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
        model_name='Deep_1M'

    #print(f"information: {evaluation.categories(train_data, item_mapping)}")
    # Initialize the StepLR scheduler
    step_size=4
    gamma=0.9
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
   # Initialize the training parameters
    consecutive_epochs_without_improvement = 0
    max_consecutive_epochs_without_improvement = 6  # Adjust as needed
    test_losses=[]
    test_ndcg = []
    # accuracy_test=[]
    KLD_test = []
    #total_KLD_train=[]
    recall_test = []
    train_losses=[]
    privacy_epsilons=[]
    best_test_loss = float("inf")  # Initialize with a large value

    #Start the training
    for epoch in range(num_epochs):
        model.train()  # Set the model in training mode
        total_loss_train = []
        for batch in train_dataloader:
            user_input, item_input, labels = [x.to(device) for x in batch]
            user_input = user_input.long()  # Convert to torch.long
            item_input = item_input.long()  # Convert to torch.long
            optimizer.zero_grad()  # Zero the gradients
            predicted_labels = model(user_input, item_input)
            #loss = torch.nn.functional.binary_cross_entropy(predicted_labels, labels.view(-1, 1).float())
            #loss= model.calculate_loss(predicted_labels, labels) # in normal cases (without DPSGD, this part can also be used)
            loss = calculate_loss(predicted_labels, labels)  # For opacus compatibility
            loss.backward()  # Backpropagation
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # Update the model parameters
            total_loss_train.append(loss.item())
        #writer.add_scalar("Loss/train", np.mean(total_loss_train), epoch)
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {np.mean(total_loss_train):}')
        if args.DPSGD=="True":
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
        # Evaluating the model
        print("starting validation")
        total_loss_test =eval_model(model, val_dataloader, recall_test, KLD_test, test_ndcg, test_user_item_dict, user_indices,
                   item_mapping, interacted_distr, interacted_items_test, top_k)
        # Generate recommendations for test users
        print("generate recommendations")
        recommendations, KLD, average_ndcg_test,KLD_score_zero = generate_recommendations(model, test_user_item_dict, user_indices,
                                                                           item_mapping, top_k, interacted_distr,interacted_items_test)
        recall_test.append(evaluation.calculate_total_average_recall(interacted_items_test, recommendations))
        KLD_test.append(np.mean(KLD))
        test_ndcg.append(np.mean(average_ndcg_test))
        # writer.add_scalar("Loss/test", np.mean(total_loss_test), epoch)
        print(f'Loss test: {np.mean(total_loss_test):.4f}')
        test_losses.append(np.mean(total_loss_test))
        #accuracy = accuracy_score(all_labels, [1 if p >= 0.5 else 0 for p in all_predictions])
        #auc = roc_auc_score(all_labels, all_predictions)
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {np.mean(total_loss_test):.4f}')
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}')
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}, AUC: {auc:.4f}')

        # Apply scheduler
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
    #print(f"epsilon= {privacy_epsilons}")
    print("KLD normal", np.mean(KLD)) # last epoch
    print("KLD normal KLD_score_zero", np.mean(KLD_score_zero))  # without normalization
    print(f"Average NDCG in normal mode @{top_k}: {np.mean(average_ndcg_test):.4f}") # last epoch


    def evaluation_and_save():
        #genres_df = evaluation.genres_features(df_item)
        # determine unique items based on training data
        unique_ids= list(train_data['movieId'].unique())
        ununique_ids_valid = [item_id for item_id in unique_ids if item_id in item_mapping]
        ununique_items = [item_mapping[item_id] for item_id in ununique_ids_valid]

        # call generate_Recommendations
        recommendations,calibrated_recommendations,average_ndcg_normal,average_ndcg_calibrated,KLDN,KLDC=generate_recommendations(model,test_user_item_dict,user_indices,item_mapping,top_k,interacted_distr,interacted_items_test,final_results=True)
        print("End of the test recommendations creation")

        MRR = evaluation.MRR(recommendations,interacted_items_test)
        print(f"MRR in normal mode: {MRR}")
        Calibrated_MRR = evaluation.MRR(calibrated_recommendations,interacted_items_test)
        print(f"MRR in calibrated mode: {Calibrated_MRR}")

        NDCG_normal = np.mean(average_ndcg_normal)
        Calibrated_NDCG = np.mean(average_ndcg_calibrated)
        print(f"Average NDCG in normal mode @{top_k}: {NDCG_normal:.4f}")
        print(f"Average NDCG in Calibarated mode @{top_k}: {Calibrated_NDCG:.4f}")

        average_recall = evaluation.calculate_total_average_recall(interacted_items_test,recommendations )
        print(f"Total Average Recall in normal state: {average_recall:.4f}")

        average_recall_calibrated = evaluation.calculate_total_average_recall(interacted_items_test,calibrated_recommendations )
        print(f"Total Average Recall in calibrated state: {average_recall_calibrated:.4f}")

        print("KLD calibrated", KLDC)

        normal_PL=evaluation.PL(train_data,item_mapping,recommendations,list(recommendations.keys()),model)
        print(f"PL normal: {normal_PL}")
        Calibrated_PL= evaluation.PL(train_data, item_mapping,calibrated_recommendations,list(calibrated_recommendations.keys()),model)
        print(f"PL calibrated: {Calibrated_PL}")
        novelty_normal= evaluation.novelty(recommendations, train_data, item_mapping, top_k)
        print("novelty normal :", novelty_normal)
        Calibrated_novelty=evaluation.novelty(calibrated_recommendations, train_data, item_mapping, top_k)
        print("novelty calibrated:",Calibrated_novelty)
        coverage =evaluation.catalog_coverage(list(recommendations.values()),ununique_items)
        print(f"coverage:{coverage}")

        #Diversity_normal= evaluation.Diversity(recommendations, genres_df)
        #print(f" Diversity normal : {Diversity_normal}")
        #Diversity_Calibrated= evaluation.Diversity(calibrated_recommendations, genres_df)
        #print(f" Diversity Calibrated : {Diversity_Calibrated}")

        # for all users ( we can substitute calibrate recommendations with recommnedations and vice versa)
        #Serendepity_normal=evaluation.serendepity_group(list(recommendations.keys()), test_user_item_dict,item_mapping,recommendations, genres_df )
        #print(f"Total Average Serendepity in normal model {Serendepity_normal}")
        #Serendepity_Calibrated= evaluation.serendepity_group(list(calibrated_recommendations.keys()), test_user_item_dict,item_mapping,calibrated_recommendations, genres_df )
        #print(f"Total Average Serendepity in calibrated model{Serendepity_Calibrated}")

        # calculate metrics per each item category
        popular=evaluation.PopularItems(train_data, item_mapping)
        category_mapping = evaluation.categories(train_data,item_mapping)
        valid_interacted,valid_reco_distr=evaluation.valid_distr_extraction(category_mapping, recommendations, interacted_items)

        #print(f"interacted_items_test:{interacted_items_test}")
        PL_per_category =evaluation.PL_items(train_data,item_mapping,recommendations,list(recommendations.keys()),model,category_mapping)
        kld_pre_category=evaluation.calculate_KLD_items(recommendations,valid_reco_distr,valid_interacted)
        ndcg_per_category=evaluation.calculate_ndcg_items(category_mapping, recommendations, interacted_items_test, top_k)
        average_total_dpf, normalized_dpf, exposure_I1, exposure_I2, normalized_exposure_1, normalized_exposure_2 = evaluation.DPF(recommendations, category_mapping)
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
        # calculate metrics per each user category
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
            #diversity_normal = evaluation.Diversity(user_group_recommendations, genres_df)
            #diversity_calibrated = evaluation.Diversity(user_group_recommendations_calibrated, genres_df)
            #serendipity_normal = evaluation.serendepity_group(user_group, test_user_item_dict, item_mapping, user_group_recommendations, genres_df)
            #serendipity_calibrated = evaluation.serendepity_group(user_group, test_user_item_dict, item_mapping, user_group_recommendations_calibrated, genres_df)
            mrr_normal = evaluation.MRR(user_group_recommendations, interacted_items_test)
            mrr_calibrated = evaluation.MRR(user_group_recommendations_calibrated, interacted_items_test)
            pl_normal = evaluation.PL(train_data, item_mapping, user_group_recommendations, user_group, model)
            pl_calibrated = evaluation.PL(train_data, item_mapping, user_group_recommendations_calibrated, user_group, model)
            kld_normal = np.mean([calibration_Func.compute_kl_divergence(interacted_distr[user_id], calibration_Func.compute_genre_distr(user_group_recommendations[user_id])) for user_id in user_group if user_id in user_group_recommendations])
            kld_calibrated = np.mean([calibration_Func.compute_kl_divergence(interacted_distr[user_id], calibration_Func.compute_genre_distr(user_group_recommendations_calibrated[user_id])) for user_id in user_group if user_id in user_group_recommendations_calibrated])
            ndcg_per_user_type=[]
            for u in user_group_recommendations:
                 common_items_normal = set(user_group_recommendations[u]).intersection(interacted_items_test[u])
                 relevance_scores_normal = [1 if i in common_items_normal else 0 for i in user_group_recommendations[u]]
                 ndcg_normal = evaluation.ndcg_at_k(relevance_scores_normal[:top_k], top_k)
                 ndcg_per_user_type.append(ndcg_normal)
            ndcg_user_group=np.mean(ndcg_per_user_type)
            coverage_per_user_group=evaluation.catalog_coverage(list(user_group_recommendations.values()),ununique_items)
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
        print("Saving to CSV file")

        if args.DPSGD == 'True':
            epsilon_value= privacy_epsilons[-1]
        else:
            epsilon_value=0

        all_metrics.update({

            'Learning Rate': initial_learning_rate,
            'learning_rate_factor':learning_rate_factor,
            'batch_size':batch_size,
            'dropout':dropout,
            'max_grad_norm': max_grad_norm,
            'noise_multiplier':noise_multiplier,
            'latent_dim': mf_dim,
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
            'model_name': f"{model_name}_{ratio}",
            'epsilon': epsilon_value,


        })

        results_df = pd.DataFrame([all_metrics])
        # Define the file path for the results file
        #results_file = 'revised_version_50.csv'
        results_file = 'Deep_1M_DPSGD.csv'
        # Check if the results file already exists
        if os.path.isfile(results_file):
            # File exists, append the current results
            with open(results_file, 'a') as f:
                results_df.to_csv(f, header=False, index=False)
        else:
            # File does not exist, create a new file and write the results
            results_df.to_csv(results_file, index=False)
        print("saving has finished")

    evaluation_and_save()




def print_results(users, niche_user,blockbuster,diverse):
    print(f"niche_user={niche_user}")
    print(f"blockbuster={blockbuster}")
    print(f"diverse={diverse}")
    #print(f"new={new}")

    niche_user_test = []
    blockbuster_test = []
    diverse_test = []
    new_test = []
    for user_id in users:
        if user_id in niche_user:
            niche_user_test.append(user_id)
        elif user_id in blockbuster:
            blockbuster_test.append(user_id)
        elif user_id in diverse:
            diverse_test.append(user_id)
        else:
            new_test.append(user_id)

    print("niche_user_test=", niche_user_test)
    print("blockbuster_test=", blockbuster_test)
    print("diverse_test=", diverse_test)
    print("new_test=", new_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Deep model with different parameters.')

    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate value')
    parser.add_argument('--max_grad_norm', default=2, type=float, help='max_grad_norm value')
    parser.add_argument('--mlp_layer_sizes', type=str, required=True, help='MLP layer sizes separated by commas')
    parser.add_argument('--DPSGD', type=str, default='False', help='True or False for DPSGD')
    parser.add_argument('--noise_multiplier', type=float, default=1, help='noise_multiplier value')
    parser.add_argument('--mf_dim', type=int, default=8, help='mf_dim value')
    parser.add_argument('--num_epochs', type=int, default=1, help='num_epochs')
    parser.add_argument('--Plot', type=bool, default=False, help='True or False for Plotting')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--delta', type=float, default=1e-4, help='delta')
    parser.add_argument('--k', type=int, default=10, help='k')
    parser.add_argument('--calibration', type=int, default=0, help='True or False for calibration (0 or 1)')
    parser.add_argument('--mlp_deep_fm_should_be_trained',type=str, default="False", help='True or False for mlp_deep_fm_should_be_trained')

    # Parse the arguments
    args = parser.parse_args()
    train(args)
    print("Done!")


#Why KLD is negative? epsilon in compute_kl_divergence
# it is possible to have a user in train but not in test ( However all test are in train)
# in case of having loop the following code is useful to clear the parameters for each run (for example when we are doing ablation study and iterating over the learning rates)
"""for name, module in model.named_children(): #inside for loop
    if hasattr(module, 'reset_parameters'):
        print('resetting ', name)
        module.reset_parameters()
# the following code is useful for defining initial weights
for param in model.parameters():
    if len(param.size())>1:
        init.normal_(param, mean=0.0, std=0.01)"""


