from torch.utils.data import DataLoader, TensorDataset
import Evaluation
import numpy as np
import pandas as pd
import Calibration_Func
import torch
from torch.utils.data import Dataset
from torch import nn
from ItemMapping import create_item_mapping
import DataProcess
from  train_test_split import train_test_split_version1
from opacus import PrivacyEngine
import os
import torch.optim as optim
import argparse
from model import SVD
from model import predict_score
import pickle
from preprocess import preprocess
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def predict_score(user_ids, item_ids, model):
    user_ids = user_ids.to(device)
    item_ids = item_ids.to(device)
    user_embed = model.user_embeddings(user_ids)
    item_embed = model.item_embeddings(item_ids)
    scores = torch.sum(user_embed * item_embed, dim=1)
    return scores

model_path_template = "{model_name}_{noise_multiplier}_{epoch}.pth"
# Function to save model
def save_model(model, model_name, optimizer, noise_multiplier, epoch):
    model_path = model_path_template.format(model_name=model_name,noise_multiplier=noise_multiplier, epoch=epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)
    print(f"Saved model to {model_path}")

# Function to load model if it exists
def load_model_if_exists(model,model_name, optimizer, noise_multiplier,trained_model=False):

    if trained_model==True:
        model_files = [f for f in os.listdir() if f.startswith(f"{model_name}_{noise_multiplier}_")]
        latest_model = max(model_files, key=os.path.getctime, default=None)
        if latest_model:
            checkpoint = torch.load(latest_model)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded model from {latest_model} starting at epoch {start_epoch+1}")
            return model, optimizer, start_epoch + 1
        else:
            print("No existing model found. Starting training from scratch.")
            return model, optimizer, 0
    else:
        return model, optimizer, 0

def read_files():
    title_col = 'title'
    genre_col = 'genres'
    item_col = 'movieId'
    df_item = pd.read_csv('movie.csv')
    df_item = df_item[
        df_item[genre_col] != '(no genres listed)']  # eliminate movies that had no genre information attached
    print('dimension: ', df_item.shape)
    item_mapping = create_item_mapping(df_item, item_col, title_col, genre_col)
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

    num_epochs = args.num_epochs
    batch_size=256
    noise_multiplier=args.noise_multiplier
    max_grad_norm=args.max_grad_norm
    top_k = args.k
    latent_dim=args.latent_dim
    delta = 1e-5
    learning_rate_factor = 0.2
    l2_reg_strength = 1e-5
    epochs_without_improvement = 0
    consecutive_epochs_without_improvement = 0
    max_consecutive_epochs_without_improvement = 6  # Adjust as needed
    initial_learning_rate = args.learning_rate # initial learning rate
    best_test_loss = float("inf")  # Initialize with a large value
    ratio = 50
    new_learning_rate = initial_learning_rate

    # read all necessary files
    df_item, item_mapping, df = read_files()
    num_users = df['userId'].max() + 1
    num_items = df['movieId'].max() + 1

    # read train and test data
    train_data, test_data= read_train_test(df)
    all_movieIds = df['movieId'].unique()
    print("test_data:", test_data.shape)
    print("train_data", train_data.shape)

    user_interacted_items = train_data.groupby('userId')['movieId'].apply(list).to_dict()  ########## training_data
    test_user_item_dict= test_data.groupby('userId')['movieId'].apply(list).to_dict()

    interacted_distr = {}
    for user_id in user_interacted_items:
        interacted = user_interacted_items[user_id]
        interacted = [item_id for item_id in interacted if item_id in item_mapping]
        interacted_items_init = [item_mapping[item_id] for item_id in interacted]
        interacted_distr[user_id] = Calibration_Func.compute_genre_distr(interacted_items_init)

    interacted_items = {}
    for u in user_interacted_items:
        interacted = user_interacted_items[u]
        interacted = [item_id for item_id in interacted if item_id in item_mapping]
        interacted_items[u] = [item_mapping[item_id] for item_id in interacted]

    interacted_items_test = {}
    for u in test_user_item_dict:
        interacted_items_init_test = [item_id for item_id in test_user_item_dict[u] if item_id in item_mapping]
        interacted_items_test[u] = [item_mapping[item_id] for item_id in interacted_items_init_test]

    user_indices = read_user_indices(test_user_item_dict, user_interacted_items, all_movieIds, ratio)

    # DataLoader for train and test data with determined batch size
    train_dataloader = DataLoader(DataProcess.MovieLensTrainDataset(train_data, all_movieIds), batch_size=batch_size,
                                  num_workers=2)
    test_dataloader = DataLoader(DataProcess.MovieLensTrainDataset(test_data, all_movieIds), batch_size=batch_size,
                                 num_workers=2)

    # Instantiate the model
    model = SVD(num_users, num_items, latent_dim)
    # Choose an optimizer (e.g., Adam)
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

    # Load model
    if args.DPSGD == 'True':
        model_name='1M_DPSGD_SVD'
    else:
        model_name = '1M_SVD'
    model, optimizer, start_epoch = load_model_if_exists(model, model_name, optimizer, args.noise_multiplier,
                                                         args.trained_model)
    # setting the configuration for applying DPSGD
    if args.DPSGD == 'True':
        model_name='1M_DPSGD_SVD'
        delta = 1e-5
        noise_multiplier = noise_multiplier
        max_grad_norm = max_grad_norm

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,)
    else:

        delta = 'None'
        noise_multiplier = 'None'
        max_grad_norm = 'None'
        model_name = '1M_SVD'
        # load model if exists

    # accuracy_test=[]
    #total_epsilon = []
    #KLD_train = []
    train_losses = []
    test_losses = []
    privacy_epsilons = []
    test_ndcg = []
    # accuracy_test=[]
    KLD_test = []
    recall_test = []
    # recall_train=[]
    print("Start training .....")
    model.to(device)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        losses = []
        for batch in train_dataloader:
            user_input, item_input, labels = batch
            user_input = user_input.to(device)
            item_input = item_input.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predictions = predict_score(user_input, item_input, model)
            # loss = bpr_loss(pos_scores, neg_scores)+ model.weight_decay * regularization_loss(model)
            loss = nn.MSELoss()(predictions, labels.float())
            # Add L2 regularization to the loss
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_reg_strength * l2_reg
            loss.backward()
            losses.append(loss)
            # Clip gradients
            #nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()
        values = [tensor.item() for tensor in losses]

        if args.DPSGD == "True":
            epsilon = privacy_engine.get_epsilon(delta)
            privacy_epsilons.append(epsilon)

            print(
                f"\tTrain Epoch: [{epoch + 1}/{num_epochs}] \t"
                f"Loss: {np.mean(values):.6f} "
                f"(ε = {epsilon:.2f}, δ = {delta})"
            )
        else:

            print(
            f"\tTrain Epoch: [{epoch+1}/{num_epochs}] \t"
            f"Loss: {np.mean(values):.6f} "

        )
        train_losses.append(np.mean(values))

        print("starting validation")
        total_loss_test, mean_loss = eval_model(model, test_dataloader)
        test_losses.append(mean_loss)  # keep mean loss to adjust learning rate
        print("generate recommendations")
        recommendations, KLD, average_ndcg_normal = generate_recommendations(model, test_user_item_dict, user_indices,
                                                                             item_mapping, top_k, interacted_distr,
                                                                             interacted_items_test, final_results=False)

        recall_test.append(Evaluation.calculate_total_average_recall(interacted_items_test, recommendations))
        # AUC.append(auc)
        KLD_test.append(np.mean(KLD))
        test_ndcg.append(np.mean(average_ndcg_normal))
        # accuracy_test.append(np.mean(Accuracy))
        # print(f"accuracy test:{np.mean(Accuracy)}")

        # stop the training after n consequent epochs without improvement
        if np.mean(total_loss_test) < best_test_loss:
            best_test_loss = np.mean(total_loss_test)
            consecutive_epochs_without_improvement = 0  # reset
        else:
            consecutive_epochs_without_improvement += 1

        if consecutive_epochs_without_improvement >= max_consecutive_epochs_without_improvement:
            print(
                f"Stopping training due to lack of improvement for {max_consecutive_epochs_without_improvement} epochs.")
            break  # Exit the training loop

        if epoch > 0 and test_losses[-1] >= test_losses[-2]:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 4:  # Adjust the number as needed.
                # Increase the learning rate

                new_learning_rate = new_learning_rate * learning_rate_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_learning_rate
                print(f"Changing learning rate to {new_learning_rate}")
                epochs_without_improvement = 0  # Reset the counter

        # save the model
        if (epoch + 1) % 50 == 0 & args.trained_model==True:
            if args.DPSGD == 'True':
                save_model(model,model_name, optimizer, noise_multiplier, epoch)
            else:
                save_model(model, model_name, optimizer,'normal', epoch)

    print(f"test_losses= {test_losses}")
    print(f"train_losses= {train_losses}")
    print(f"KLD_test= {KLD_test}")
    print(f"NDCG_test ={test_ndcg}")
    print(f"recall_test= {recall_test}")

    def evaluation_and_save():

        #genres_df= Evaluation.genres_features(df_item)
        unique_ids = list(train_data['movieId'].unique())
        ununique_ids_valid = [item_id for item_id in unique_ids if item_id in item_mapping]
        ununique_items = [item_mapping[item_id] for item_id in ununique_ids_valid]
        # print(f"ununique_items:{ununique_items}")
        # call generate_Recommendations
        recommendations, calibrated_recommendations, average_ndcg_normal, average_ndcg_calibrated, KLDN, KLDC = generate_recommendations(
            model, test_user_item_dict, user_indices, item_mapping, top_k, interacted_distr, interacted_items_test,
            final_results=True)
        print("End of the test recommendations creation")
        # MRR
        MRR= Evaluation.MRR(recommendations,interacted_items_test)
        print(f"MRR in normal mode: {MRR}")
        Calibrated_MRR= Evaluation.MRR(calibrated_recommendations,interacted_items_test)
        print(f"MRR in calibrated mode: {Calibrated_MRR}")
        # NDCG
        NDCG= np.mean(average_ndcg_normal)
        Calibrated_NDCG= np.mean(average_ndcg_calibrated)
        print(f"Average NDCG in normal mode @{top_k}: {NDCG:.4f}")
        print(f"Average NDCG in Calibarated mode @{top_k}: {Calibrated_NDCG:.4f}")
        # Recall
        average_recall = Evaluation.calculate_total_average_recall(interacted_items_test,recommendations )
        print(f"Total Average Recall in normal state: {average_recall:.4f}")
        average_recall_calibrated = Evaluation.calculate_total_average_recall(interacted_items_test,calibrated_recommendations )
        print(f"Total Average Recall in calibrated state: {average_recall_calibrated:.4f}")
        # PL
        normal_PL=Evaluation.PL(train_data,item_mapping,recommendations,list(recommendations.keys()),model)
        print(f"PL normal: {normal_PL}")
        Calibrated_PL= Evaluation.PL(train_data, item_mapping,calibrated_recommendations,list(calibrated_recommendations.keys()),model)
        print(f"PL calibrated: {Calibrated_PL}")
        # novelty
        novelty_normal= Evaluation.novelty(recommendations,train_data,item_mapping,top_k)
        print("novelty normal :", novelty_normal)
        Calibrated_novelty=Evaluation.novelty(calibrated_recommendations,train_data,item_mapping,top_k)
        print("novelty calibrated:",Calibrated_novelty)
        # coverage
        coverage = Evaluation.catalog_coverage(list(recommendations.values()), ununique_items)
        print(f"coverage:{coverage}")
        #Diversity_normal= Evaluation.Diversity(recommendations, genres_df)
        #print(f" Diversity normal : {Diversity_normal}")
        #Diversity_Calibrated= Evaluation.Diversity(calibrated_recommendations, genres_df)
        #print(f" Diversity Calibrated : {Diversity_Calibrated}")
        # for all users ( we can substitute calibrate recommendations with recommnedations and vice versa)
        #Serendepity_normal=Evaluation.serendepity_group(list(recommendations.keys()), test_user_item_dict,item_mapping,recommendations, genres_df )
        #print(f"Total Average Serendepity in normal model {Serendepity_normal}")
        #Serendepity_Calibrated= Evaluation.serendepity_group(list(calibrated_recommendations.keys()), test_user_item_dict,item_mapping,calibrated_recommendations, genres_df )
        #print(f"Total Average Serendepity in calibrated model{Serendepity_Calibrated}")
        popular = Evaluation.PopularItems(train_data, item_mapping)
        category_mapping = Evaluation.categories(train_data, item_mapping)

        # By valid_distr_extraction, we can extract valid items for calculating different metrics per each item category
        # The output of this function will be used for calculate_KLD_items
        valid_interacted, valid_reco_distr = Evaluation.valid_distr_extraction(category_mapping, recommendations,
                                                                               interacted_items)
        # PL per category (item)
        PL_per_category = Evaluation.PL_items(train_data, item_mapping, recommendations, list(recommendations.keys()),
                                              model, category_mapping)
        # KLD per category (item)
        kld_pre_category = Evaluation.calculate_KLD_items(recommendations, valid_reco_distr, valid_interacted)

        # NDCG per category (item)
        ndcg_per_category = Evaluation.calculate_ndcg_items(category_mapping, recommendations, interacted_items_test,
                                                            top_k)
        # DPF, Exposure
        average_total_dpf, normalized_dpf, exposure_I1, exposure_I2, normalized_exposure_1, normalized_exposure_2 = Evaluation.DPF(
            recommendations, category_mapping)

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


        def calculate_group_metrics(group_name, user_group, recommendations, calibrated_recommendations, train_data, item_mapping, test_user_item_dict, interacted_items_test, model):
            if len(user_group) == 0:
                return None

            user_group_recommendations = {u: recommendations[u] for u in recommendations if u in user_group}
            user_group_recommendations_calibrated = {u: calibrated_recommendations[u] for u in calibrated_recommendations if u in user_group}

            # Calculate metrics
            novelty_normal = Evaluation.novelty(user_group_recommendations, train_data, item_mapping,top_k)
            novelty_calibrated = Evaluation.novelty(user_group_recommendations_calibrated, train_data, item_mapping,top_k)
            #diversity_normal = Evaluation.Diversity(user_group_recommendations, genres_df)
            #diversity_calibrated = Evaluation.Diversity(user_group_recommendations_calibrated, genres_df)
            #serendipity_normal = Evaluation.serendepity_group(user_group, test_user_item_dict, item_mapping, user_group_recommendations, genres_df)
            #serendipity_calibrated = Evaluation.serendepity_group(user_group, test_user_item_dict, item_mapping, user_group_recommendations_calibrated, genres_df)
            mrr_normal = Evaluation.MRR(user_group_recommendations, interacted_items_test)
            mrr_calibrated = Evaluation.MRR(user_group_recommendations_calibrated, interacted_items_test)
            pl_normal = Evaluation.PL(train_data, item_mapping, user_group_recommendations, user_group, model)
            pl_calibrated = Evaluation.PL(train_data, item_mapping, user_group_recommendations_calibrated, user_group, model)

            # Calculate KLD
            kld_normal = np.mean([Calibration_Func.compute_kl_divergence(interacted_distr[user_id], Calibration_Func.compute_genre_distr(user_group_recommendations[user_id])) for user_id in user_group if user_id in user_group_recommendations])
            kld_calibrated = np.mean([Calibration_Func.compute_kl_divergence(interacted_distr[user_id], Calibration_Func.compute_genre_distr(user_group_recommendations_calibrated[user_id])) for user_id in user_group if user_id in user_group_recommendations_calibrated])
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
            group_metrics = calculate_group_metrics(group_name, user_group, recommendations, calibrated_recommendations, train_data, item_mapping, test_user_item_dict, interacted_items_test, model)
            if group_metrics:
                all_metrics.update(group_metrics)
        if args.DPSGD == 'True':
            epsilon_value= privacy_epsilons[-1]
        else:
            epsilon_value=0

        all_metrics.update({
            'Learning Rate': initial_learning_rate,
            'learning_rate_factor':learning_rate_factor,
            'batch_size':batch_size,
            'dropout':'None',
            'max_grad_norm': max_grad_norm,
            'noise_multiplier':noise_multiplier,
            'latent_dim':latent_dim ,
            'mlp_layer_sizes':'None',
            'num_epochs':num_epochs,
            'step_size':'None',
            'gamma':'None',
            'delta':delta,
            'Test Losses': test_losses[-1],
            'Train Losses': train_losses[-1],
            'KLD_test':KLD_test[-1],
            'NDCG_test':test_ndcg[-1],
            'recall_test':recall_test[-1],
            'Total MRR': MRR,
            'Calibrated_MRR':Calibrated_MRR,
            'NDCG_normal': NDCG,
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
            'model_name':f'{model_name}_{ratio}',
            'epsilon':epsilon_value


        })

        results_df = pd.DataFrame([all_metrics])
        # Define the file path for the results file
        results_file = 'SVD_1M_DPSGD.csv'
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
    parser.add_argument('--max_grad_norm', default=2, type=float)
    parser.add_argument('--noise_multiplier', default=0.4, type=float)
    parser.add_argument('--DPSGD', type=str, default='False', help='True or False for DPSGD')
    parser.add_argument('--trained_model', default=True, type=bool)
    parser.add_argument('--num_epochs', type=int, default=400, help='num_epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--k', type=int, default=10, help='k')
    parser.add_argument('--calibration', type=int, default=0, help='True or False for calibration (0 or 1)')
    parser.add_argument('--latent_dim', type=int, default=5, help='latent_dim')

    # Parse the arguments
    args = parser.parse_args()
    train(args)


