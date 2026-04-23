import pickle

import pandas as pd
import numpy as np
from MF_MLP import NeuMF
from torch.optim.lr_scheduler import StepLR
import torch
from MF_MLP import calculate_loss
from Preprocess import preprocess
from train_test_split import train_test_split_version1
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os
from ItemMapping import create_item_mapping
import json
import calibration_Func

# Press the green button in the gutter to run the script.
import argparse
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def read_dataset():
    title_col = 'title'
    genre_col = 'genres'
    item_col = 'movieId'

    df_item = pd.read_csv('movie.csv')
    df_item = df_item[ df_item[genre_col] != '(no genres listed)']  # eliminate movies that had no genre information attached
    print('dimension: ', df_item.shape)
    item_mapping = create_item_mapping(df_item, item_col, title_col, genre_col)

    #df,_,_= read_data_ml100k()
    """file_path = 'u.data'
    names = ['userId', 'movieId', 'rating', 'timestamp']
    df = pd.read_csv(file_path, sep='\t', names=names)"""

    df_path='df.csv'
    if os.path.isfile(df_path):
        df= pd.read_csv(df_path)
        print(f"df shape:{df.shape}")
    else:
        names = ['userId', 'movieId', 'rating', 'timestamp']
        file_path = 'ratings.dat'
        df = pd.read_table(file_path, names=names, sep="::", engine='python')
        print(f"pre_preprocess:{df.shape}")
        df = preprocess(df)
        print(f"post_preprocess:{df.shape}")
        print(len(np.unique(list(df['userId']))))
        print(len(np.unique(list(df['movieId']))))
        print(df.head)
        df.to_csv(df_path, index=False)

    all_movieIds = df['movieId'].unique()
    num_users = df['userId'].max() + 1
    num_items = df['movieId'].max() + 1

    if os.path.isfile('train_data.csv') and os.path.isfile('test_data.csv'):
        train_data = pd.read_csv('train_data.csv')
        test_data = pd.read_csv('test_data.csv')
    else:
        train_data, test_data= train_test_split_version1(df)
        train_data.to_csv('train_data.csv', index=False)
        test_data.to_csv('test_data.csv', index=False)



    print("test_data:",test_data.shape)
    print("train_data",train_data.shape)
    user_interacted_items = train_data.groupby('userId')['movieId'].apply(list).to_dict()  ########## training_data
    Liked = user_interacted_items
    test_user_item_dict = test_data.groupby('userId')['movieId'].apply(list).to_dict()
    interacted_items_test = {}
    for u in test_user_item_dict:
        interacted_items = [item_id for item_id in test_user_item_dict[u] if item_id in item_mapping]
        interacted_items_test[u] = [item_mapping[item_id] for item_id in interacted_items]
    interacted_distr = {}
    for user_id in Liked:
        interacted = Liked[user_id]
        interacted = [item_id for item_id in interacted if item_id in item_mapping]
        interacted_items = [item_mapping[item_id] for item_id in interacted]
        interacted_distr[user_id] = calibration_Func.compute_genre_distr(interacted_items)


    return train_data, test_data, interacted_distr, num_users, num_items, all_movieIds


def evaluations():
    for u in test_user_item_dict:
        test_items = user_indices[u]
        # test_items= list(not_interacted_items)

        test_items = [item_id for item_id in test_items if item_id in item_mapping]
        # print("test_items", test_items)
        user_k = len(test_items)

        """predicted_scores = np.squeeze(model(torch.tensor([u] * k), torch.tensor(
            test_items)).detach().numpy())  # could be interpreted as score"""
        user_tensor = torch.tensor([u] * user_k, device=device).long()
        item_tensor = torch.tensor(test_items, device=device).long()
        predicted_scores = model(user_tensor, item_tensor).detach().cpu().numpy()

        """top20_items = [test_items[i] for i in
                       np.argsort(-predicted_scores)[:top_k].tolist()]  # recommendations based on item_id"""
        predicted_scores_flattened = predicted_scores.flatten()
        top_indices = np.argsort(-predicted_scores_flattened)[:top_k]
        top20_items = [test_items[i] for i in top_indices]

        recommendations[u] = [item_mapping[item_id] for item_id in top20_items]

        # NDCG calculations
        common_items = set(top20_items).intersection(test_user_item_dict[u])
        relevance_scores = [1 if i in common_items else 0 for i in top20_items]
        # print("common:", common_items)
        # print("top20_items: ", top20_items)
        # print("relevance:", relevance_scores)
        # print(relevance_scores)

        ndcg = NDCG.ndcg_at_k(relevance_scores[:top_k], top_k)
        average_ndcg_test.append(ndcg)

        # all_labels = [1 if item in test_user_item_dict[u] else 0 for item in test_items]
        # all_predictions = [1 if p >= 0.5 else 0 for p in predicted_scores]
        # accuracy = accuracy_score(all_labels, all_predictions)
        # auc = roc_auc_score(all_labels, all_predictions)
        # Accuracy.append(accuracy)
        reco_distr[u] = calibration_Func.compute_genre_distr(recommendations[u])

        # compute KLD
        KLD.append(calibration_Func.compute_kl_divergence(interacted_distr[u], reco_distr[u], score=1))
        KLD_score_zero.append(calibration_Func.compute_kl_divergence(interacted_distr[u], reco_distr[u], score=0))



# Assuming 'calculate_loss', 'NDCG', 'calibration_Func', and other necessary imports are done

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, device, num_epochs, step_size, gamma,
                 max_consecutive_epochs_without_improvement, top_k, delta=None, privacy_engine=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        self.max_consecutive_epochs_without_improvement = max_consecutive_epochs_without_improvement
        self.delta = delta
        self.privacy_engine = privacy_engine
        self.top_k = top_k

        # Metrics
        self.train_losses = []
        self.test_losses = []
        self.KLD_test = []
        self.test_ndcg = []
        self.recall_test = []
        self.privacy_epsilons = []

    def train_epoch(self):
        self.model.train()
        total_loss_train = []
        for batch in self.train_dataloader:
            user_input, item_input, labels = [x.to(self.device) for x in batch]
            user_input, item_input = user_input.long(), item_input.long()
            self.optimizer.zero_grad()
            predicted_labels = self.model(user_input, item_input)
            loss = calculate_loss(predicted_labels, labels)  # Adjust this to your loss calculation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss_train.append(loss.item())

        train_loss = np.mean(total_loss_train)
        self.train_losses.append(train_loss)
        return train_loss

    def validate(self):
        self.model.eval()
        total_loss_test = []
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                user_input, item_input, labels = [x.to(self.device) for x in batch]
                user_input, item_input = user_input.long(), item_input.long()
                predicted_labels = self.model(user_input, item_input)
                loss = torch.nn.functional.binary_cross_entropy(predicted_labels, labels.view(-1, 1).float())
                total_loss_test.append(loss.item())

                # Store predictions and labels for metric calculation
                all_predictions.extend(predicted_labels.squeeze().tolist())
                all_labels.extend(labels.tolist())

        validation_loss = np.mean(total_loss_test)
        self.test_losses.append(validation_loss)

        # Compute metrics
        ndcg_score , recall_score, kld_score= evaluations()
        self.test_ndcg.append(ndcg_score)
        self.recall_test.append(recall_score)
        self.KLD_test.append(kld_score)

        print(f'Validation NDCG: {ndcg_score:.4f}, Recall: {recall_score:.4f}, KLD: {kld_score:.4f}')

        return validation_loss

    def train(self):
        best_test_loss = float('inf')
        consecutive_epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            validation_loss = self.validate()

            # Scheduler step
            self.scheduler.step()

            if self.delta and self.privacy_engine:
                epsilon = self.privacy_engine.get_epsilon(self.delta)
                self.privacy_epsilons.append(epsilon)
                print(
                    f"Epoch: {epoch + 1}, Train Loss: {train_loss:.6f}, Validation Loss: {validation_loss:.6f}, ε: {epsilon:.2f}")
            else:
                print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.6f}, Validation Loss: {validation_loss:.6f}")

            # Check for early stopping
            if validation_loss < best_test_loss:
                best_test_loss = validation_loss
                consecutive_epochs_without_improvement = 0
            else:
                consecutive_epochs_without_improvement += 1

            if consecutive_epochs_without_improvement >= self.max_consecutive_epochs_without_improvement:
                print(
                    f"Stopping training due to lack of improvement for {self.max_consecutive_epochs_without_improvement} epochs.")
                break



def load_train_objs(mlp_layer_sizes,lr,mf_dim, dropout):
    train_data, test_data, interacted_distr,num_users, num_items, all_movieIds = read_dataset()
    num_users = int(num_users)
    num_items = int(num_items)
    model = NeuMF(num_users, num_items, mf_dim=mf_dim, mlp_layer_sizes=mlp_layer_sizes, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(),lr=initial_learning_rate)
    return train_data, test_data, interacted_distr, model, optimizer


def prepare_dataloader(dataset, batch_size: int):

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=  DistributedSampler(dataset))


def main(mlp_layer_sizes, train_data, test_data, rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int,dropout:int, mf_dim:int,lr):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(mlp_layer_sizes,lr,mf_dim, dropout)
    train_data = prepare_dataloader(train_data, batch_size)
    val_data = prepare_dataloader(test_data, batch_size)

    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()



if __name__ == '__main__':
    # Load configuration from the JSON file
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    num_epochs = config.get('num_epochs')
    num_epochs=400
    noise_multiplier = config.get('noise_multiplier')
    #max_grad_norm = config.get('max_grad_norm')
    delta = config.get('delta')
    top_k=config.get('k')
    Calibration=config.get('calibration')

    #learning_rate = config.get('learning_rate')

    # Add arguments to the parser
    parser = argparse.ArgumentParser(description='Run Deep model with different parameters.')

    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate value')
    parser.add_argument('--max_grad_norm', type=float, required=True, help='max_grad_norm value')
    parser.add_argument('--mlp_layer_sizes', type=str, required=True, help='MLP layer sizes separated by commas')
    parser.add_argument('--DPSGD', type=str, required=True, help='True or False for DPSGD')
    parser.add_argument('--noise_multiplier', type=float, required=True, help='noise_multiplier value')
    # Parse the arguments
    args = parser.parse_args()
    initial_learning_rate = args.learning_rate
    learning_rate_factor = 0.9
    best_test_loss = float("inf")  # Initialize with a large value

    mf_dim = 8
    dropout = 0.4
    # Define the MLP layer sizes
    mlp_layer_sizes = args.mlp_layer_sizes  # Example sizes, modify as needed
    mlp_layer_sizes = [int(x) for x in mlp_layer_sizes.split(',')]



