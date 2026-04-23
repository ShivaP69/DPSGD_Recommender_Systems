from model import BPR
import pandas as pd
from Calibration import create_item_mapping
import torch
import argparse
import os
import numpy as np
# import torch
import torch.nn as nn
import torch.optim as optim
import create_data_loader_DP
# from sklearn.model_selection import GroupKFold
# from sklearn.model_selection import KFold
import evaluation
import Calibration
# from opacus import PrivacyEngine
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))


def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()


def regularization_loss(model):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param)
    return l2_reg


def predict_score(user_ids, item_ids, model):
    user_embed = model.user_embeddings(user_ids)
    item_embed = model.item_embeddings(item_ids)
    scores = torch.sum(user_embed * item_embed, dim=1)
    return scores


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



def read_files():
    genre_col = 'genres'
    names = ['userId', 'movieId', 'rating', 'timestamp']
    file_path = 'ratings.dat'
    data = pd.read_table(file_path, names=names, sep="::", engine='python')
    """file_path = 'u.data'
    names = ['userId', 'movieId', 'rating', 'timestamp']
    data = pd.read_csv(file_path, sep='\t', names=names)
    """
    print(data.shape)
    df_item = pd.read_csv('movie.csv')
    df_item = df_item[
        df_item[genre_col] != '(no genres listed)']  # eliminate movies that had no genre information attached
    print('dimension: ', df_item.shape)
    print(df_item.columns)

    return data,df_item

def genres_gf(df_item,):
    title_col = 'title'
    genre_col = 'genres'
    item_col = 'movieId'
    # genres gf is usefull for calculating diversity and scerendipity
    all_genres = set()
    for genres in df_item['genres']:
        all_genres.update(genres.split('|'))

    genres_df = pd.DataFrame(df_item['title'].unique(), columns=['title'])
    for genre in all_genres:
        genres_df[genre] = 0

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

def eval_model(model,test_loader):
    model.eval()  # Set the model to evaluation mode
    total_loss_test = []
    with torch.no_grad():
        for batch in test_loader:
            user, pos_item, neg_item = [x.to(device) for x in batch]
            user=user.long()
            pos_item=pos_item.long()
            neg_item=neg_item.long()
            pos_scores, neg_scores = model(user, pos_item, neg_item)
            test_loss = bpr_loss(pos_scores, neg_scores)
            total_loss_test.append(test_loss.item())
    return total_loss_test,np.mean(total_loss_test)


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
                Calibration.generate_item_candidates(item_mapping, test_items, predicted_scores))
            calibrated_recommendations[u] = Calibration.calib_recommend(item_candidates[u], interacted_distr[u],
                                                                        topn=top_k, lmbda=0.8)
            # NDCG calculations
            common_items_normal = set(recommendations[u]).intersection(interacted_items_test[u])
            relevance_scores_normal = [1 if i in common_items_normal else 0 for i in recommendations[u]]
            # print(relevance_scores)
            ndcg_normal = evaluation.ndcg_at_k(relevance_scores_normal[:top_k], top_k)
            average_ndcg_normal.append(ndcg_normal)
            common_items_calibrated = set(calibrated_recommendations[u]).intersection(interacted_items_test[u])
            relevance_scores_calibrated = [1 if i in common_items_calibrated else 0 for i in calibrated_recommendations[u]]
            ndcg_calibrated = evaluation.ndcg_at_k(relevance_scores_calibrated[:top_k], top_k)
            average_ndcg_calibrated.append(ndcg_calibrated)
            # all_labels = [1 if item in test_user_item_dict[u] else 0 for item in test_items]
            # all_predictions = [1 if p >= 0.5 else 0 for p in predicted_scores]
            # accuracy = accuracy_score(all_labels, all_predictions)
            # auc = roc_auc_score(all_labels, all_predictions)
            # Accuracy.append(accuracy)
            reco_distr[u] = Calibration.compute_genre_distr(recommendations[u])
            # compute KLD
            KLD_normal.append(Calibration.compute_kl_divergence(interacted_distr[u], reco_distr[u]))
            # for calculating KLD for calibrated recommendations
            reco_distr_calibrated[u] = Calibration.compute_genre_distr(
                calibrated_recommendations[u])
            KLD_calibrated.append(Calibration.compute_kl_divergence(interacted_distr[u],
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
    latent_dim = args.latent_dim
    num_epochs = args.num_epochs
    initial_learning_rate = args.learning_rate
    batch_size = args.batch_size
    top_k=args.k
    learning_rate_factor = 0.2
    best_test_loss = float("inf")  # Initialize with a large value
    consecutive_epochs_without_improvement = 0
    max_consecutive_epochs_without_improvement = 6  # Adjust as needed
    epochs_without_improvement = 0
    weight_decay = 1e-5
    ratio=50
    new_learning_rate = initial_learning_rate
    title_col = 'title'
    genre_col = 'genres'
    item_col = 'movieId'

    # reading the data
    data, df_item = read_files()
    item_mapping = create_item_mapping(df_item, item_col, title_col, genre_col)
    num_users, num_items = data['userId'].max() + 1, data['movieId'].max() + 1
    train_loader, test_loader, train_data, test_data, all_movieIds = create_data_loader_DP.create_data_loader(data,args.privacy)
    test_user_item_dict = test_data.groupby('userId')['movieId'].apply(list).to_dict()
    user_interacted_items = train_data.groupby('userId')['movieId'].apply(list).to_dict()  ########## training_data
    # After appling DP

    interacted_items_test = {}
    for u in test_user_item_dict:
        interacted_items_t= [item_id for item_id in test_user_item_dict[u] if item_id in item_mapping]
        interacted_items_test[u] = [item_mapping[item_id] for item_id in interacted_items_t]

    interacted_distr = {}
    for user_id in user_interacted_items:
        interacted = user_interacted_items[user_id]
        interacted = [item_id for item_id in interacted if item_id in item_mapping]
        interacted_items_init = [item_mapping[item_id] for item_id in interacted]
        interacted_distr[user_id] = Calibration.compute_genre_distr(interacted_items_init)

    interacted_items = {}
    for u in user_interacted_items:
        interacted = user_interacted_items[u]
        interacted = [item_id for item_id in interacted if item_id in item_mapping]
        interacted_items[u] = [item_mapping[item_id] for item_id in interacted]


    user_indices = read_user_indices(test_user_item_dict, user_interacted_items, all_movieIds,
                                     ratio)  # reading user indices
    # Initialize the model and optimizer
    model = BPR(num_users, num_items, latent_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    model_name = f"BPR_{ratio}_{args.privacy}_{batch_size}_1M"
    model, optimizer, start_epoch, best_test_loss = load_or_initialize_model(model, optimizer, model_name,args.load_model)

    total_KLD_train = []
    train_losses = []
    test_losses = []
    #privacy_epsilons = []
    test_ndcg = []
    # accuracy_test=[]
    KLD_test = []
    recall_test = []
    # recall_train=[]
    # scheduler = StepLR(optimizer, step_size=4, gamma=0.9)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        losses = []
        for batch in train_loader:
            user, pos_item, neg_item = [x.to(device) for x in batch]
            optimizer.zero_grad()
            pos_scores, neg_scores = model(user, pos_item, neg_item)
            # loss = bpr_loss(pos_scores, neg_scores)+ model.weight_decay * regularization_loss(model)
            # private
            loss = bpr_loss(pos_scores, neg_scores) + weight_decay * regularization_loss(model)
            loss.backward()
            # Clip gradients
            losses.append(loss)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            # total_loss += loss.item()
            optimizer.zero_grad()
        # epsilon = privacy_engine.get_epsilon(delta)
        # privacy_epsilons.append(epsilon)
        values = [tensor.item() for tensor in losses]

        print(
            "\tTrain Epoch: [{}/{}] \t"
            "Loss: {:.6f} ".format(epoch + 1, num_epochs, np.mean(values))
        )

        train_losses.append(np.mean(values))

        print("starting validation")
        total_loss_test, mean_loss = eval_model(model, test_loader)
        test_losses.append(mean_loss)  # keep mean loss to adjust learning rate
        print("generate recommendations")
        recommendations, KLD, average_ndcg_normal = generate_recommendations(model, test_user_item_dict, user_indices,
                                                                             item_mapping, top_k, interacted_distr,
                                                                             interacted_items_test, final_results=False)
        recall_test.append(evaluation.calculate_total_average_recall(interacted_items_test, recommendations))
        # AUC.append(auc)
        KLD_test.append(np.mean(KLD))
        test_ndcg.append(np.mean(average_ndcg_normal))

        if np.mean(total_loss_test) < best_test_loss:
            best_test_loss = np.mean(total_loss_test)
            consecutive_epochs_without_improvement = 0  # reset
        else:
            consecutive_epochs_without_improvement += 1

        if consecutive_epochs_without_improvement >= max_consecutive_epochs_without_improvement:
            print(
                "Stopping training due to lack of improvement for {} epochs.".format(
                    max_consecutive_epochs_without_improvement))
            break  # Exit the training loop

        if epoch > 0 and test_losses[-1] >= test_losses[-2]:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 4:  # Adjust the number as needed.
                # Increase the learning rate
                new_learning_rate = new_learning_rate * learning_rate_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_learning_rate
                print("Changing learning rate to {}".format(new_learning_rate))
                epochs_without_improvement = 0  # Reset the counter
        # save model
        if args.save_model:
            if (epoch + 1) % 20 == 0 :
                save_model(model, optimizer, epoch, best_test_loss, model_name)

    print(f"test_losses= {test_losses}")
    print(f"train_losses= {train_losses}")
    print(f"KLD_test= {KLD_test}")
    print(f"KLD_train={total_KLD_train}")
    print(f"NDCG_test ={test_ndcg}")
    print(f"recall_test= {recall_test}")

    def evaluation_and_save():
        top_k=args.k
        #genres_df = evaluation.genres_featues(df_item)
        unique_ids = list(train_data['movieId'].unique())
        ununique_ids_valid = [item_id for item_id in unique_ids if item_id in item_mapping]
        ununique_items = [item_mapping[item_id] for item_id in ununique_ids_valid]
        # call generate_Recommendations
        recommendations, calibrated_recommendations, average_ndcg_normal, average_ndcg_calibrated, KLDN, KLDC = generate_recommendations(
            model, test_user_item_dict, user_indices, item_mapping, top_k, interacted_distr,
            interacted_items_test, final_results=True)

        #MRR
        MRR = evaluation.MRR(recommendations, interacted_items_test)
        print("MRR in normal mode: {}".format(MRR))
        Calibrated_MRR = evaluation.MRR(calibrated_recommendations, interacted_items_test)
        print("MRR in calibrated mode: {}".format(Calibrated_MRR))
        # NDCG
        NDCG = np.mean(average_ndcg_normal)
        Calibrated_NDCG = np.mean(average_ndcg_normal)
        print("Average NDCG in normal mode @{}: {:.4f}".format(top_k, NDCG))
        print("Average NDCG in Calibarated mode @{}: {:.4f}".format(top_k, Calibrated_NDCG))
        # Recall
        average_recall = evaluation.calculate_total_average_recall(interacted_items_test, recommendations)
        print("Total Average Recall in normal state: {:.4f}.".format(average_recall))
        average_recall_calibrated = evaluation.calculate_total_average_recall(interacted_items_test,
                                                                              calibrated_recommendations)
        print("Total Average Recall in calibrated state: {:.4f}".format(average_recall_calibrated))
        # PL
        normal_PL = evaluation.PL(train_data, item_mapping, recommendations, list(recommendations.keys()), model)
        print("PL normal: {}".format(normal_PL))
        Calibrated_PL = evaluation.PL(train_data, item_mapping, calibrated_recommendations,
                                      list(calibrated_recommendations.keys()), model)
        print("PL calibrated: {}".format(Calibrated_PL))
        # Novelty
        novelty_normal = evaluation.novelty(recommendations, train_data, item_mapping,top_k)
        print("novelty normal :", novelty_normal)
        Calibrated_novelty = evaluation.novelty(calibrated_recommendations, train_data, item_mapping,top_k)
        print("novelty calibrated:", Calibrated_novelty)
        # Coverage
        coverage = evaluation.catalog_coverage(list(recommendations.values()), ununique_items)
        print(f"coverage:{coverage}")
        """Diversity_normal = evaluation.Diversity(recommendations, genres_df)
        print(" Diversity normal : {}".format(Diversity_normal))
        Diversity_Calibrated = evaluation.Diversity(calibrated_recommendations, genres_df)
        print(" Diversity Calibrated : {}".format(Diversity_Calibrated))
        # for all users ( we can substitute calibrate recommendations with recommnedations and vice versa)
        Serendepity_normal = evaluation.serendepity_group(list(recommendations.keys()), test_user_item_dict,
                                                          item_mapping, recommendations, genres_df)
        print(f"Total Average Serendepity in normal model {Serendepity_normal}")
        Serendepity_Calibrated = evaluation.serendepity_group(list(calibrated_recommendations.keys()),
                                                              test_user_item_dict, item_mapping,
                                                              calibrated_recommendations, genres_df)
        print("Total Average Serendepity in calibrated model{}".format(Serendepity_Calibrated))"""
        # Category
        category_mapping = evaluation.categories(train_data, item_mapping)

        valid_interacted, valid_reco_distr = evaluation.valid_distr_extraction(category_mapping, recommendations,
                                                                               interacted_items)
        # PL per category (item)
        PL_per_category = evaluation.PL_items(train_data, item_mapping, recommendations, list(recommendations.keys()),
                                              model, category_mapping)
        # KLD per category (item)
        kld_pre_category = evaluation.calculate_KLD_items(category_mapping, recommendations, valid_reco_distr,
                                                          valid_interacted)
        # NDCG per category (item)
        ndcg_per_category = evaluation.calculate_ndcg_items(category_mapping, recommendations, interacted_items_test,
                                                            top_k)

        average_total_dpf, normalized_dpf, exposure_I1, exposure_I2, normalized_exposure_1, normalized_exposure_2 = evaluation.DPF(
            recommendations,
            category_mapping)

        popular = evaluation.PopularItems(train_data, item_mapping)
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
        niche_user, blockbuster, diverse, new = evaluation.type_of_user_total(item_mapping, popular, train_data)
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
            user_group_recommendations_calibrated = {u: calibrated_recommendations[u] for u in
                                                     calibrated_recommendations if u in user_group}

            # Calculate metrics
            novelty_normal = evaluation.novelty_R_total(user_group_recommendations, train_data, item_mapping)
            novelty_calibrated = evaluation.novelty_R_total(user_group_recommendations_calibrated, train_data,
                                                            item_mapping)
            """genres_df=genres_gf(df_item)
            diversity_normal = evaluation.Diversity(user_group_recommendations, genres_df)
            diversity_calibrated = evaluation.Diversity(user_group_recommendations_calibrated, genres_df)
            serendipity_normal = evaluation.serendepity_group(user_group, test_user_item_dict, item_mapping,
                                                              user_group_recommendations, genres_df)
            serendipity_calibrated = evaluation.serendepity_group(user_group, test_user_item_dict, item_mapping,
                                                                  user_group_recommendations_calibrated, genres_df)"""
            mrr_normal = evaluation.MRR(user_group_recommendations, interacted_items_test)
            mrr_calibrated = evaluation.MRR(user_group_recommendations_calibrated, interacted_items_test)
            pl_normal = evaluation.PL(train_data, item_mapping, user_group_recommendations, user_group, model)
            pl_calibrated = evaluation.PL(train_data, item_mapping, user_group_recommendations_calibrated, user_group,
                                          model)

            # Calculate KLD
            kld_normal = np.mean([Calibration.compute_kl_divergence(interacted_distr[user_id],
                                                                    Calibration.compute_genre_distr(
                                                                        user_group_recommendations[user_id])) for user_id in user_group])
            kld_calibrated = np.mean([Calibration.compute_kl_divergence(interacted_distr[user_id],
                                                                        Calibration.compute_genre_distr(
                                                                            user_group_recommendations_calibrated[
                                                                                user_id])) for user_id in user_group])
            ndcg_per_user_type = []
            for u in user_group_recommendations:
                common_items_normal = set(user_group_recommendations[u]).intersection(interacted_items_test[u])
                relevance_scores_normal = [1 if i in common_items_normal else 0 for i in user_group_recommendations[u]]
                ndcg_normal = evaluation.ndcg_at_k(relevance_scores_normal[:top_k], top_k)
                ndcg_per_user_type.append(ndcg_normal)
            ndcg_user_group = np.mean(ndcg_per_user_type)
            coverage_per_user_group = evaluation.catalog_coverage(list(user_group_recommendations.values()),
                                                                  ununique_items)
            average_total_dpf_user_group, normalized_dpf_user_group, coverage_category_I1_user_group, coverage_category_I2_user_group, normalized_coverage_category_I1_user_group, normalized_coverage_category_I2_user_group = evaluation.DPF(
                user_group_recommendations, category_mapping)
            """return {
                f'{group_name} Novelty Normal': novelty_normal,
                f'{group_name} Novelty Calibrated': novelty_calibrated,
                f'{group_name} Diversity Normal': diversity_normal,
                f'{group_name} Diversity Calibrated': diversity_calibrated,
                f'{group_name} Serendipity Normal': serendipity_normal,
                f'{group_name} Serendipity Calibrated': serendipity_calibrated,
                f'{group_name} MRR Normal': mrr_normal,
                f'{group_name} MRR Calibrated': mrr_calibrated,
                f'{group_name} PL Normal': pl_normal,
                f'{group_name} PL Calibrated': pl_calibrated,
                f'{group_name} KLD Normal': kld_normal,
                f'{group_name} KLD Calibrated': kld_calibrated
            }"""

            return {
                '{} Novelty Normal'.format(group_name): novelty_normal,
                '{} Novelty Calibrated'.format(group_name): novelty_calibrated,
                #'{} Diversity Normal'.format(group_name): diversity_normal,
                #'{} Diversity Calibrated'.format(group_name): diversity_calibrated,
                #'{} Serendipity Normal'.format(group_name): serendipity_normal,
                #'{} Serendipity Calibrated'.format(group_name): serendipity_calibrated,
                '{} MRR Normal'.format(group_name): mrr_normal,
                '{} MRR Calibrated'.format(group_name): mrr_calibrated,
                '{} PL Normal'.format(group_name): pl_normal,
                '{} PL Calibrated'.format(group_name): pl_calibrated,
                '{} KLD Normal'.format(group_name): kld_normal,
                '{} KLD Calibrated'.format(group_name): kld_calibrated,
                '{} NCDG Normal'.format(group_name): ndcg_user_group,
                '{} coverage'.format(group_name): coverage_per_user_group,
                '{} DPF'.format(group_name): average_total_dpf_user_group,
                '{} normalized_DPF'.format(group_name): normalized_dpf_user_group,
                '{} coverage_category_I1'.format(group_name): coverage_category_I1_user_group,
                '{} coverage_category_I2'.format(group_name): coverage_category_I2_user_group,
                '{} normalized_coverage_category_I1'.format(group_name): normalized_coverage_category_I1_user_group,
                '{} normalized_coverage_category_I2'.format(group_name): normalized_coverage_category_I2_user_group,
            }

        all_metrics = {}
        for group_name, user_group in user_groups.items():
            group_metrics = calculate_group_metrics(group_name, user_group, recommendations, calibrated_recommendations,
                                                    train_data, item_mapping, test_user_item_dict,
                                                    interacted_items_test, model)
            if group_metrics:
                all_metrics.update(group_metrics)
        print("saving to csv file")

        all_metrics.update({

            'Learning Rate': initial_learning_rate,
            'learning_rate_factor': learning_rate_factor,
            'batch_size': batch_size,
            'dropout': 'None',
            'latent_dim': args.latent_dim,
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
        results_file = 'BPR_1M_LDP.csv'

        # Check if the results file already exists
        if os.path.exists(results_file):
            # File exists, append the current results
            with open(results_file, 'a') as f:
                results_df.to_csv(f, header=False, index=False)
        else:
            # File does not exist, create a new file and write the results
            results_df.to_csv(results_file, index=False)

    evaluation_and_save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BPR model with different parameters.')

    # Add arguments to the parser

    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate value')
    parser.add_argument('--latent_dim', type=int, default=5, help='Latent dimension')
    parser.add_argument('--privacy', type=float, default=0.1, help='privacy')
    parser.add_argument('--num_epochs', type=int, default=400, help='epochs')
    parser.add_argument('--save_model', type=bool, default=False, help='save_model')
    parser.add_argument('--load_model', type=bool, default=False, help='load_model')
    parser.add_argument('--batch_size', type=int, default=200, help='batch_size')
    parser.add_argument('--k', type=int, default=10, help='k')
    parser.add_argument('--calibration', type=int, default=0, help='True or False for calibration (0 or 1)')

    # Parse the arguments
    args = parser.parse_args()
    train(args)


