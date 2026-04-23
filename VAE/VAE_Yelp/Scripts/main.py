import argparse
import os.path
import time
import torch
import torch.optim as optim
import numpy as np
from opacus import PrivacyEngine
from create_item_mapping import create_item_mapping
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import models
import data_yelp
import metric
import calibration_Func
import evaluation
from torch.utils.data import DataLoader
import os


parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
parser.add_argument('--data', type=str, default='',
                    help='Movielens-20m dataset location')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.01,
                    help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=500,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.35,
                    help='largest annealing parameter') # Based on https://iris.unito.it/retrieve/1dcb49f4-a408-4180-b0e8-592973d273d6/icann_2022.pdf#:~:text=Yelp%20C,504
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--DPSGD', type=str, default="False", help='True or False for DPSGD') # if This is True, DP should be False
parser.add_argument('--privacy',type=float, default=0.1)
parser.add_argument('--noise_multiplier', type=float, default=0.2, help='noise_multiplier value')
parser.add_argument('--max_grad_norm', default=2, type=float, help='max_grad_norm value')
parser.add_argument('--DP', type=str, default="False", help='True of False for DP') # if This is true, DPSGD should be False
parser.add_argument('--read_data', type=bool, default=True, help='True or False for running data.py')
parser.add_argument('--save_model', type=bool, default=False, help='True of False for save_model')
parser.add_argument('--filter_restaurants', type=bool, default=True, help='True of False for filter (just keep rest)')
parser.add_argument('--filter_20', type=bool, default=True, help='True of False for filter_20')
parser.add_argument('--state_filter', type=bool, default=False, help='True of False for state_filter')

args = parser.parse_args()

def compute_avg_interaction_percentage(sparse_matrix):
    num_users, num_items = sparse_matrix.shape
    num_interactions = sparse_matrix.nnz  # Number of non-zero entries
    total_possible = num_users * num_items
    percentage = (num_interactions / total_possible) * 100
    return round(percentage, 4) # the percentage of non-empty entries in the user-item matrix.

def dataset_stats(sparse_matrix, name="Dataset"):
    num_users, num_items = sparse_matrix.shape
    num_interactions = sparse_matrix.nnz  # non-zero entries
    density = (num_interactions / (num_users * num_items)) * 100 #the percentage of non-empty entries in the user-item matrix.
    print(f"=== {name} ===")
    print(f"# of users         : {num_users:,}")
    print(f"# of items         : {num_items:,}")
    print(f"# of interactions  : {num_interactions:,}")
    print(f"% of interactions  : {density:.2f}%")

def calculate_sparsity(matrix):
    num_users, num_items = matrix.shape
    num_interactions = matrix.nnz  # number of non-zero entries
    total_possible = num_users * num_items
    interaction_percentage= (num_interactions / total_possible) *100
    sparsity = 100 - interaction_percentage # the percentage of empty interactions
    return sparsity

# Set the random seed manually for reproductibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")
if args.read_data:
    data_yelp.run_data(args)
###############################################################################
# Load data
###############################################################################
user_col = 'user_idx'
item_col = 'business_idx'
value_col = 'stars'
timestamp = 'data'
df_item = pd.read_csv('df_item.csv')
print(f"df item cats:{df_item['categories']}")
item_mapping = create_item_mapping(df_item, 'business_idx', 'name', 'categories')
# data.py already has been run and data is ready


path = ''
pro_dir = os.path.join(path, 'pro_sg')
raw_data = pd.read_csv(os.path.join(pro_dir, "raw_data.csv"))
#print("raw data users", len(raw_data[user_col].unique()))
all_movieIds = raw_data[item_col].unique()
num_users = raw_data[user_col].max() + 1
num_items = raw_data[item_col].max() + 1

# read unique UID
"""with open(os.path.join(pro_dir, 'unique_uid.txt'), 'r') as f:
    unique_uid = [line.strip() for line in f]
profile2id = dict((int(pid), i) for (i, pid) in enumerate(unique_uid))
"""

path = os.path.join(pro_dir, 'train.csv')
train_data = pd.read_csv(path)
nan_rows = train_data[train_data[item_col].isna()]
print("Rows with NaN business_idx:", nan_rows)
train_data=train_data.dropna(subset=[item_col])
train_data[item_col]=train_data[item_col].astype(int)

path_validation_tr=os.path.join(pro_dir, 'validation_tr.csv')
validation_tr=pd.read_csv(path_validation_tr)
validation_tr[item_col]=validation_tr[item_col].astype(int)

path_validation_te=os.path.join(pro_dir, 'validation_te.csv')
validation_te=pd.read_csv(path_validation_te)
validation_te[item_col]=validation_te[item_col].astype(int)

path_test_tr=os.path.join(pro_dir, 'test_tr.csv')
test_tr= pd.read_csv(path_test_tr)
test_tr[item_col]=test_tr[item_col].astype(int)

path_test_te= os.path.join(pro_dir, 'test_te.csv')
test_te=pd.read_csv(path_test_te)
test_te=pd.read_csv(path_test_te)


unique_sid = []
with open(os.path.join(pro_dir,'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(int(line.strip()))

show2id = {int(uid): idx for idx, uid in enumerate(unique_sid)}
index_to_item_global = {idx: int(uid) for idx, uid in enumerate(unique_sid)}


unique_uid = []
with open(os.path.join(pro_dir, 'unique_uid.txt'), 'r') as f:
    for line in f:
        unique_uid.append(int(line.strip()))

raw_data_user_ids = set(raw_data[user_col].unique())
# Step 2: Convert unique_uid to a set if it's not already
unique_uid_set = set(unique_uid)
# Step 3: Check if both sets are identical
if raw_data_user_ids == unique_uid_set:
    print("All unique_ids and user_ids in raw_data are identical!")
else:
    # If they're not identical, find the differences
    missing_in_unique_uid = raw_data_user_ids - unique_uid_set
    missing_in_raw_data = unique_uid_set - raw_data_user_ids

    print(f"IDs in raw_data but not in unique_uid: {missing_in_unique_uid}")
    print(f"IDs in unique_uid but not in raw_data: {missing_in_raw_data}")

profile2id = {int(uid): idx for idx, uid in enumerate(unique_uid)}
id_to_profile = {v: k for k, v in profile2id.items()}

validation_tr=validation_tr[[user_col, item_col]]
test_tr= test_tr[[user_col, item_col]]
cat_val,popular_list_val,users_train_val = evaluation.categories(train_data,validation_tr, item_mapping) # for validation
cat_test, popular_list_test , users_train_test= evaluation.categories(train_data,test_tr, item_mapping) # for test data

print (f"train_data size: {train_data.shape}")
true_batch= int(0.01 * train_data.shape[0])
args.batch_size=true_batch
print(f"Updated batch size: {args.batch_size}")

# This approach does not work, beacues here there are some users in test that are not in train
"""user_indices = {}
ratio = 50
indices_file_path = f'indices_file_path_{ratio}.pkl'
if os.path.isfile(indices_file_path):
    with open(indices_file_path, 'rb') as f:
        user_indices = pickle.load(f)
        print("User Indices has been loaded")

else:

    np.random.seed(42)  # Ensure reproducibility
    for u in test_user_item_dict:
        not_interacted_items = set(all_movieIds) - set(user_interacted_items[u])  # user_interacted_items comes from train data
        not_interacted_items= set(not_interacted_items) - set(test_user_item_dict[u])
        interacted_test= test_user_item_dict[u]
        num_to_select = min(len(not_interacted_items), ratio * len(interacted_test))
        selected_not_interacted=list(np.random.choice(list(not_interacted_items), size=num_to_select, replace=False))
        test_items = selected_not_interacted + list(interacted_test)
        test_items = list(set(test_items))
        user_indices[u] = test_items

    with open(indices_file_path, 'wb') as file:
        pickle.dump(user_indices, file)

"""


loader = data_yelp.DataLoader_classic(args.data,show2id,index_to_item_global,profile2id,item_mapping)
n_items = loader.load_n_items()
train_data = loader.load_data('train')
print("how dense the interaction matrix (train_data) is : ",compute_avg_interaction_percentage(train_data))
print("sparcity of data ",calculate_sparsity(train_data))
dataset_stats(train_data)
item_genre_counts = {item_id: len(item.genres) for item_id, item in item_mapping.items()} # how many genres/business we have for each item
#print(item_genre_counts)
counts = list(item_genre_counts.values())
print(f"Total items: {len(counts)}")
print(f"Min genres per item: {np.min(counts)}")
print(f"Max genres per item: {np.max(counts)}")
print(f"Mean genres per item: {np.mean(counts):.2f}")
print(f"Median genres per item: {np.median(counts)}")

"""tp_tr_val,tp_te_val, vad_data_tr, vad_data_te, interacted_distr_val_tr  = loader.load_data(
    'validation')"""
tp_tr_test, tp_te_test, test_data_tr, test_data_te, interacted_distr_test_tr = loader.load_data(
    'test')

# For test set
print(f"# test users (hisotry) (unique): {tp_tr_test['uid'].nunique()}")
print(f"# test users (target)(unique): {tp_te_test['uid'].nunique()}")
print(f"# test items (unique in tr+te): {len(set(tp_tr_test['sid']).union(tp_te_test['sid']))}")
print(f"# interactions in test_tr: {len(tp_tr_test)}")
print(f"# interactions in test_te: {len(tp_te_test)}")


train_dataloader=DataLoader(data_yelp.InteractionDataset(train_data))

N = train_data.shape[0]
idxlist = list(range(N))

###############################################################################
# Build the model
###############################################################################
print("n items is:", n_items)
p_dims = [200, 600, n_items]
model = models.MultiVAE(p_dims).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
criterion = models.loss_function

if args.DPSGD == 'True':
    model_name="vae_DPSGD_Yelp"
    delta = 1e-5
    noise_multiplier = args.noise_multiplier
    max_grad_norm = args.max_grad_norm

    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm, )
else:
    if args.DP=="True":
        delta = 'None'
        noise_multiplier = 'None'
        max_grad_norm = 'None'
        model_name = 'DP_vae_Yelp'
    else:
        delta = 'None'
        noise_multiplier = 'None'
        max_grad_norm = 'None'
        model_name='vae_Yelp'

###############################################################################
# Training code
###############################################################################

# TensorboardX Writer

writer = SummaryWriter()


def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i: row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def train():
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    global update_count
    #np.random.shuffle(idxlist)

    for batch_idx, data in enumerate(train_dataloader):
        data = data.to(device)  # Move batch to device (GPU/CPU)
        if data.dim() == 1:  # If data is 1D, add a batch dimension
            data = data.unsqueeze(0)

        if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap,
                         1. * update_count / args.total_anneal_steps)
        else:
            anneal = args.anneal_cap

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        loss = criterion(recon_batch, data, mu, logvar, anneal)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        update_count += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                  'loss {:4.2f}'.format(
                epoch, batch_idx, len(train_dataloader),
                elapsed * 1000 / args.log_interval,
                train_loss / args.log_interval))
            # Log loss to tensorboard
            n_iter = (epoch - 1) * len(train_dataloader) + batch_idx
            writer.add_scalars('data/loss', {'train': train_loss / args.log_interval}, n_iter)

            start_time = time.time()
            train_loss = 0.0

    if args.DPSGD == "True":
        epsilon = privacy_engine.get_epsilon(delta)
        print(
            f"(ε = {epsilon:.2f}, δ = {delta})"
        )
    if args.DP=="True":
        epsilon=args.privacy
    else:
        epsilon = 0
    return train_loss / args.log_interval, epsilon


def get_top_k_recommendations(recon_batch, k=50):
    """
    Extracts top-K recommendations for each user.
    recon_batch: The output of the model (predicted scores for each item).
    k: The number of recommendations to retrieve for each user.
    """
    # Get the indices of the top-K items in descending order of scores
    top_k_items = np.argpartition(-recon_batch, k, axis=1)[:, :k]

    # Sort the top K items for each user
    top_k_items_sorted = np.argsort(-recon_batch[np.arange(recon_batch.shape[0])[:, None], top_k_items])[:, :k]

    return top_k_items[np.arange(top_k_items.shape[0])[:, None], top_k_items_sorted]


# for calculating miss-calibration, we have two different concepts:1. miss-calibration between reconstructed output and actual output
# miss-calibration between history of the user and reconstructed output (we need to have same users in training and validation) or we have another option: calculate
# miss-calibration between tr and te (Because in this code we don't have same users for train and test/val)

def evaluate(model,tp_tr,tp_te,data_tr, data_te,interacted_distr,popular_id,cat,debug=False):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))  # users
    e_N = data_tr.shape[0]
    n10_list_init = []
    r20_list = []
    r50_list = []
    n50_list=[]
    n100_list=[]
    KLD = []  # store KLD of all users (all users, all batches)
    recommendations = {}

    unique_tp_tr = tp_tr.drop_duplicates(subset='uid', keep='first')
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            batch_users = e_idxlist[start_idx:end_idx]  # get a batch of user indices
            numerized_user_ids = unique_tp_tr['uid'].values[batch_users]
            data = data_tr[batch_users]
            heldout_data = data_te[batch_users]  # what we want

            data_tensor = naive_sparse2tensor(data).to(device)

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap,
                             1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            recon_batch, mu, logvar = model(data_tensor)
            loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)
            total_loss += loss.item()
            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf  # to not recommend same items (same as history)
            top_k = get_top_k_recommendations(recon_batch, k=10)
            for i, user_id in enumerate(numerized_user_ids):
                recomme = top_k[i]
                recomme =[index_to_item_global[r] for r in recomme]
                recomme = [item_id for item_id in recomme if item_id in item_mapping]
                recommendations[user_id] = [item_mapping[item_id] for item_id in recomme]
                # print(calibration_Func.compute_genre_distr(recommendations[user_id]))
                reco_distr = calibration_Func.compute_genre_distr(recommendations[user_id])
                # actual_user = index_to_userid_tr[user_id]
                if user_id in interacted_distr:
                    # print("Calculating KLD")
                    KLD.append(
                        calibration_Func.compute_kl_divergence(interacted_distr[user_id], reco_distr, score=1))
                else:
                    print(f"User {user_id} not found in interacted_distr.")
            #print(f"recon_batch: {recon_batch}")
            #print(f"heldout_data:{heldout_data}")
            n10 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 10)
            n50= metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 50)
            n100=metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r20 = metric.Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = metric.Recall_at_k_batch(recon_batch, heldout_data, 50)
            #print(f"n10: {n10}, r20: {r20}, r50: {r50},n50:{n50},n100:{n100}")


            n10_list_init.append(n10)
            #print(f"n10_list_init: {n10_list_init}")
            r20_list.append(r20)
            r50_list.append(r50)
            n50_list.append(n50)
            n100_list.append(n100)
        # users= tp_tr['userId'].tolist()
        # print(all(elem in users for elem in set_users))

        normal_PL = evaluation.PL(tp_tr, item_mapping, recommendations, list(recommendations.keys()), model, popular_id,
                                  index_to_item_global)
        print(f"normal_PL validation:{normal_PL}")


    """result = all(elem in set(set_users)  for elem in set(index_to_userid_tr.values()) )
    print(f"result of this data: {result}")"""
    # real_users= tp_tr['userId'].tolist()
    # result = all(elem in set_users for elem in real_users)
    # print(f"result: {result}")
    total_loss /= len(range(0, e_N, args.batch_size))
    n10_list = np.concatenate(n10_list_init)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)
    n100_list = np.concatenate(n100_list)
    n50_list = np.concatenate(n50_list)
    print(f"KLD:{(np.mean(KLD))}")

    return total_loss, np.mean(n10_list), np.mean(r20_list), np.mean(r50_list), np.mean(KLD),normal_PL, np.mean(n100_list),np.mean(n50_list)



def evaluate_and_save(tp_tr,tp_te,data_tr, data_te,interacted_distr,popular_id,train_loss,privacy_epsilon,cat,users):
    user_interacted_items = tp_tr.groupby('uid')['sid'].apply(list).to_dict()  ########## training_data of test part
    interacted_items = {}
    for u in user_interacted_items:  # u is an index
        numerized_items = user_interacted_items[u]
        #print(f"index_to_item_global:{index_to_item_global}")
        real_interacted_item_ids = [index_to_item_global[item_id] for item_id in numerized_items if
                                    item_id in index_to_item_global]
        interacted_items[u] = [item_mapping[item_id] for item_id in real_interacted_item_ids if item_id in item_mapping]

    test_user_item_dict= tp_te.groupby('uid')['sid'].apply(list).to_dict()  ########## test data
    interacted_items_test = {}
    for u in test_user_item_dict:
        numerized_items = test_user_item_dict[u]
        real_test_item_ids=[index_to_item_global[item_id] for item_id in numerized_items if item_id in index_to_item_global]
        interacted_items_test[u] =  [item_mapping[item_id] for item_id in real_test_item_ids if item_id in item_mapping]

    unique_ids = unique_sid
    ununique_ids_valid = [item_id for item_id in unique_ids if item_id in item_mapping]
    ununique_items = [item_mapping[item_id] for item_id in ununique_ids_valid]
    eval_start_time = time.time()
    popular = evaluation.PopularItems(popular_id)
    niche_user, blockbuster, diverse, new = evaluation.type_of_user_total(item_mapping, popular, tp_tr,index_to_item_global)

    # user_groups= [niche_user, blockbuster,diverse, new]
    user_groups = {
        "niche_user": niche_user,
        "blockbuster": blockbuster,
        "diverse": diverse,
        "new": new
    }
    #print("user_groups",user_groups)
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0])) # users
    #print("Unique users:", len(set(e_idxlist)) == len(e_idxlist)) # True
    e_N = data_tr.shape[0]
    n10_list = []
    r10_list = []
    r50_list = []
    k=50
    KLD=[] # store KLD of all users (all users, all batches)
    recommendations={}
    my_users=[]
    unique_tp_tr = tp_tr.drop_duplicates(subset='uid', keep='first')

    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            batch_users = e_idxlist[start_idx:end_idx]  # get a batch of user indices

            #print(f"Processing batch from {start_idx} to {end_idx}, user indices: {batch_users}")
            #numerized_user_ids = tp_tr['uid'].values[batch_users]
            #print(f"numerized_user_ids:{numerized_user_ids}")

            #print(f"Processing batch from {start_idx} to {end_idx}, user indices: {batch_users}")
            #numerized_user_ids = tp_tr['uid'].values[batch_users]
            numerized_user_ids = unique_tp_tr['uid'].values[batch_users]

            #print(f"numerized_user_ids: {numerized_user_ids}")
            data = data_tr[batch_users]
            heldout_data = data_te[batch_users]  # actual data

            data_tensor = naive_sparse2tensor(data).to(device)

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            recon_batch, mu, logvar = model(data_tensor)
            loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)
            total_loss += loss.item()
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf  # to not recommend same items (same as history)
            top_k = get_top_k_recommendations(recon_batch, k)
            # Add more detailed debug information
            print(f"Processing batch from {start_idx} to {end_idx}")
            #print("interacted",interacted_distr.keys())

            for i, user_id in enumerate(numerized_user_ids):
                my_users.append(user_id)
                recomme = top_k[i]
                recomme = [index_to_item_global[r] for r in recomme]
                recomme = [item_id for item_id in recomme if item_id in item_mapping]
                recommendations[user_id] = [item_mapping[item_id] for item_id in recomme] # this can not make a re-writing issue because user_ids came from numerized_user_ids not batch index
                #print(f"recommendations {user_id}: {recommendations[user_id]}")
                reco_distr = calibration_Func.compute_genre_distr(recommendations[user_id])
               # user id and different users ids are not the same!
                if user_id in interacted_distr:
                    #print("Calculating KLD")
                    KLD.append(
                        calibration_Func.compute_kl_divergence(interacted_distr[user_id], reco_distr, score=1))
                else:
                    print(f"User {user_id} not found in interacted_distr.")

                #set_users.append(actual_user)
            n10 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 10)
            r10= metric.Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = metric.Recall_at_k_batch(recon_batch, heldout_data, 50)
            n10_list.append(n10)
            r10_list.append(r10)
            r50_list.append(r50)
        #users= tp_tr['userId'].tolist()
        #print(all(elem in users for elem in set_users))
        novelty_normal = evaluation.novelty(recommendations,popular_id,users,k)
        #print("novelty normal :", novelty_normal)
        coverage = evaluation.catalog_coverage(list(recommendations.values()),ununique_items )
        #print(f"coverage:{coverage}")
        normal_PL = evaluation.PL(tp_tr, item_mapping, recommendations, list(recommendations.keys()), model,popular_id,index_to_item_global)
        PL_per_category = evaluation.PL_items(tp_tr, item_mapping, recommendations, list(recommendations.keys()), model, cat,
                                       index_to_item_global,
                                       popular_id)

        valid_interacted, valid_reco_distr = evaluation.valid_distr_extraction(cat, recommendations,interacted_items)
        kld_per_category = evaluation.calculate_KLD_items(recommendations, valid_reco_distr,
                                                          valid_interacted)
        ndcg_per_category = evaluation.calculate_ndcg_items(cat, recommendations, interacted_items_test,k)
        average_total_dpf, normalized_dpf, exposure_I1, exposure_I2, normalized_exposure_1, normalized_exposure_2 = evaluation.DPF(
            recommendations, cat)

    print(f"DPF:{average_total_dpf}")
    print(f"normalized_dpf: {normalized_dpf}")
    print(f"normalized_exposure_1: {normalized_exposure_1}")
    print(f"normalized_exposure_2: {normalized_exposure_2}")
    print(f"exposure_I1 : {exposure_I1}")
    print(f"exposure_I2 : {exposure_I2}")
    print(f"PL_per_category:{PL_per_category}")
    print(f"kld_pre_category:{kld_per_category}")
    print(f"ndcg_per_category:{ndcg_per_category}")
    print(f"average_total_dpf:{average_total_dpf}")
    total_loss /= len(range(0, e_N, args.batch_size))
    #print(f"n10_list:{n10_list}")
    #print(f"r10_list:{r10_list}")
    #print(f"r50_list:{r50_list}")
    n10_list = np.concatenate(n10_list)
    r10_list = np.concatenate(r10_list)
    r50_list = np.concatenate(r50_list)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:6.4f} | '
          'n10 {:5.3f} | r20 {:5.3f} | r50 {:5.3f} | KLD {:5.3f} | PL {:5.3f}'.format(
        epoch, time.time() - eval_start_time, total_loss,
         np.mean(n10_list), np.mean(r10_list), np.mean(r50_list), np.mean(KLD),normal_PL))

    #return total_loss, np.mean(n10_list), np.mean(r10_list), np.mean(KLD),normal_PL

    def calculate_group_metrics(group_name, user_group, recommendations, item_mapping,
                                                model):
        if len(user_group) == 0:
            return None

        items=[recommendations[u] for u in recommendations if u in user_group]
        #print(f"{group_name}:{items}")
        #print(f"recommednations for {group_name}: {items}")
        user_group_recommendations = {u: recommendations[u] for u in recommendations if u in user_group}
        #user_group_recommendations_calibrated = {u: calibrated_recommendations[u] for u in calibrated_recommendations if u in user_group}

        # Calculate metrics
        #novelty_normal = evaluation.novelty_R_total(user_group_recommendations, train_data, item_mapping)
        #novelty_calibrated = evaluation.novelty_R_total(user_group_recommendations_calibrated, train_data, item_mapping)
        #diversity_normal = evaluation.Diversity(user_group_recommendations, genres_df)
        #diversity_calibrated = evaluation.Diversity(user_group_recommendations_calibrated, genres_df)
        #serendipity_normal = evaluation.serendepity_group(user_group, test_user_item_dict, item_mapping, user_group_recommendations, genres_df)
        #serendipity_calibrated = evaluation.serendepity_group(user_group, test_user_item_dict, item_mapping, user_group_recommendations_calibrated, genres_df)
        #mrr_normal = evaluation.MRR(user_group_recommendations, interacted_items_test)
        #mrr_calibrated = evaluation.MRR(user_group_recommendations_calibrated, interacted_items_test)
        pl_normal = evaluation.PL(tp_tr, item_mapping, recommendations, list(recommendations.keys()), model,popular_id,index_to_item_global)
        novelty_normal = evaluation.novelty(user_group_recommendations, popular_id, users, k)
        #pl_calibrated = evaluation.PL(train_data, item_mapping, user_group_recommendations_calibrated, user_group, model)
        # Calculate KLD
        user_interaction=[user_group_recommendations[user_id] for user_id in user_group if user_id in user_group_recommendations]
        #print(f"interacted_distr for {user_group}: {user_interaction}")
        kld_normal = np.mean([calibration_Func.compute_kl_divergence(interacted_distr[user_id], calibration_Func.compute_genre_distr(user_group_recommendations[user_id])) for user_id in user_group if user_id in user_group_recommendations])
        #kld_calibrated = np.mean([calibration_Func.compute_kl_divergence(interacted_distr[user_id], calibration_Func.compute_genre_distr(user_group_recommendations_calibrated[user_id])) for user_id in user_group if user_id in user_group_recommendations_calibrated])
        ndcg_per_user_type = []
        num_user_ids_dict1 = len(user_group_recommendations)
        num_user_ids_dict2 = len(interacted_items_test)

        # Find similar user IDs
        similar_user_ids = set(interacted_items_test.keys()) & set(user_group_recommendations.keys())
        num_similar_user_ids = len(similar_user_ids)

        # Print results
        print(f"Number of user IDs in dict1: {num_user_ids_dict1}")
        print(f"Number of user IDs in dict2: {num_user_ids_dict2}")
        print(f"Number of similar user IDs: {num_similar_user_ids}")
        for u in user_group_recommendations:
            if u in interacted_items_test:
                common_items_normal = set(user_group_recommendations[u]).intersection(interacted_items_test[u])
                relevance_scores_normal = [1 if i in common_items_normal else 0 for i in user_group_recommendations[u]]
                ndcg_normal = evaluation.ndcg_at_k(relevance_scores_normal[:k], k)
                ndcg_per_user_type.append(ndcg_normal)
        ndcg_user_group = np.mean(ndcg_per_user_type)
        coverage_per_user_group = evaluation.catalog_coverage(list(user_group_recommendations.values()), ununique_items)
        average_total_dpf_user_group, normalized_dpf_user_group, coverage_category_I1_user_group, coverage_category_I2_user_group, normalized_coverage_category_I1_user_group, normalized_coverage_category_I2_user_group = evaluation.DPF(
            user_group_recommendations, cat)

        return {
            f'{group_name} Novelty Normal': novelty_normal,
            #f'{group_name} Novelty Calibrated': novelty_calibrated,
            #f'{group_name} Diversity Normal': diversity_normal,
            #f'{group_name} Diversity Calibrated': diversity_calibrated,
            #f'{group_name} Serendipity Normal': serendipity_normal,
            #f'{group_name} Serendipity Calibrated': serendipity_calibrated,
            #f'{group_name} MRR Normal': mrr_normal,
            #f'{group_name} MRR Calibrated': mrr_calibrated,
            f'{group_name} PL Normal': pl_normal,
            #f'{group_name} PL Calibrated': pl_calibrated,
            f'{group_name} KLD Normal': kld_normal,
            #f'{group_name} KLD Calibrated': kld_calibrated,
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
    #print("Users in recommendations:", set(recommendations.keys()))
    #print("Users in niche_user:", set(niche_user))
    print("Missing niche users in recommendations:", set(niche_user) - set(recommendations.keys()))

    for group_name, user_group in user_groups.items():
        group_metrics = calculate_group_metrics(group_name, user_group, recommendations, item_mapping,
                                                model)
        print(f"{group_name}: {group_metrics}")
        if group_metrics:
            all_metrics.update(group_metrics)

    print("Saving to CSV file")


    epsilon_value = privacy_epsilon[-1]  # it could be 0 or privacy or calculated based on DPSGD (in all cased train has returned one element per each iteration, therefore we have a list as privacy_epsilon (fullfilled by 0 or privacy or calculated by opacus)

    all_metrics.update({

        'Learning Rate': args.lr,
        'learning_rate_factor': "None",
        'batch_size': args.batch_size,
        'dropout': "None",
        'max_grad_norm': max_grad_norm,
        'noise_multiplier': noise_multiplier,
        'latent_dim': "None",
        'mlp_layer_sizes': "None",
        'num_epochs': args.epochs,
        'step_size': "None",
        'gamma': "None",
        'delta': delta,
        'Test Losses': total_loss,
        'Train Losses': train_loss,
        'KLD_test': np.mean(KLD),
        'NDCG_test':np.mean(n10_list) ,
        'recall_test': np.mean(r10_list),
        'Total MRR': "None",
        'Calibrated_MRR': "None",
        'NDCG_normal':np.mean(n10_list) ,
        'Calibrated_NDCG': "None",
        'average_recall': np.mean(r10_list),
        'average_recall_calibrated': "None",
        'KLD_Normal': np.mean(KLD),
        'KLD_Calibrated': "None",
        'normal_PL': normal_PL,
        'Calibrated_PL': "None",
        'novelty_normal': novelty_normal,
        'Calibrated_novelty': "None",
        #'Diversity_normal': "None",
        #'Diversity_Calibrated': "None",
        #'Serendepity_normal': "None",
        #'Serendepity_Calibrated': "None",
        'Coverage': coverage,
        'exposure_I1': exposure_I1,
        'exposure_I2': exposure_I2,
        'DPF': average_total_dpf,
        'normalized_DPF': normalized_dpf,
        'normalized_exposure_1': normalized_exposure_1,
        'normalized_exposure_2': normalized_exposure_2,
        'PL_per_category': PL_per_category,
        'kld_pre_category': kld_per_category,
        'ndcg_per_category': ndcg_per_category,
        'epsilon':epsilon_value,
        'model_name': f"{model_name}"

    })

    results_df = pd.DataFrame([all_metrics])
    # Define the file path for the results file
    if args.DP=="True":
        results_file = 'edited_DP_VAE_Yelp.csv'
        print("DP csv file will be saved")
    else:
        results_file = 'edited_version_VAE_Yelp.csv'
        print("DPSGD csv file will be saved")

    # Check if the results file already exists
    if os.path.isfile(results_file):
        # File exists, append the current results
        with open(results_file, 'a') as f:
            results_df.to_csv(f, header=False, index=False)
    else:
        # File does not exist, create a new file and write the results
        results_df.to_csv(results_file, index=False)
    print("saving has finished")



best_n10 = -np.inf
update_count = 0
privacy_epsilon=[]
max_consecutive_epochs_without_improvement=50
best_test_loss= float("inf")  # Initialize with a large value
# At any point you can hit Ctrl + C to break out of training early.
validation_loss=[]
epochs_without_improvement = 0
new_learning_rate = args.lr
learning_rate_factor = 0.2
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss,epsilon = train()
        privacy_epsilon.append(epsilon)
        model.eval()
        """val_loss, n10, r20, r50, kld, pl,n100,n50 = evaluate(model,tp_tr_val,tp_te_test, vad_data_tr, vad_data_te,
                                                     interacted_distr_val_tr,popular_list_val,cat_val)"""
        val_loss, n10, r20, r50, kld, pl, n100, n50 = evaluate(model,tp_tr_test, tp_te_test, test_data_tr, test_data_te,
                                                               interacted_distr_test_tr, popular_list_val, cat_val)
        validation_loss.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
              'n10 {:5.3f} | r20 {:5.3f} | r50 {:5.3f} | n50 {:5.3f} | n100 {:5.3f}| KLD {:4.2f}'.format(
            epoch, time.time() - epoch_start_time, val_loss,
            n10, r20, r50, n50, n100, kld))
        print('-' * 89)

        n_iter = epoch * len(range(0, N, args.batch_size))
        writer.add_scalars('data/loss', {'valid': val_loss}, n_iter)
        writer.add_scalar('data/n10', n10, n_iter)
        #writer.add_scalar('data/n10', n10, n_iter)
        writer.add_scalar('data/r20', r20, n_iter)
        writer.add_scalar('data/r50', r50, n_iter)

        if val_loss< best_test_loss:
            best_test_loss= val_loss
            consecutive_epochs_without_improvement=0 # reset
        else:
            consecutive_epochs_without_improvement+=1

        if consecutive_epochs_without_improvement >= max_consecutive_epochs_without_improvement:
            print(
                f"Stopping training due to lack of improvement for {max_consecutive_epochs_without_improvement} epochs.")
            break  # Exit the training loop

        if epoch > 1 and validation_loss[-1] >= validation_loss[-2]:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 4:  # Adjust the number as needed.
                # Increase the learning rate

                new_learning_rate = new_learning_rate * learning_rate_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_learning_rate
                print(f"Changing learning rate to {new_learning_rate}")
                epochs_without_improvement = 0  # Reset the counter
        # Save the model if the n100 is the best we've seen so far.
        """if n10 > best_n100:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_n100 = n10"""


except KeyboardInterrupt:
    #print('-' * 89)
    print('Exiting from training early')

#total_loss,test_ndcg, recall_test20,recall_test50, KLD_test,normal_PL= evaluate(tp_tr_test,test_data_tr, test_data_te,interacted_distr_test_tr,popular_id)
evaluate_and_save(tp_tr_test,tp_te_test, test_data_tr, test_data_te,interacted_distr_test_tr,popular_list_test,train_loss,privacy_epsilon,cat_test,users_train_test)

print("Saving to CSV file")