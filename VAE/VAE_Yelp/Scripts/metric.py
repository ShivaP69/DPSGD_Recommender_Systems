import bottleneck as bn
import numpy as np

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=10):
    '''
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]

    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    non_zero_relevance = heldout_batch.getnnz(axis=1)

    """print(f"Number of non-zero entries per user: {non_zero_relevance}")
    for user_idx in range(batch_users):
        print(f"User {user_idx}: Predicted scores for top-{k} items: {X_pred[user_idx, idx_topk[user_idx]]}")
        print(
            f"User {user_idx}: Ground truth relevance for top-{k} items: {heldout_batch[user_idx, idx_topk[user_idx]].toarray()}")"""

    print(f"Number of non-zero entries per user: {non_zero_relevance}")
    for user_idx in range(batch_users):
        print(f"User {user_idx}: Predicted scores for top-{k} items: {X_pred[user_idx, idx_topk[user_idx]]}")
        print(
            f"User {user_idx}: Ground truth relevance for top-{k} items: {heldout_batch[user_idx, idx_topk[user_idx]].toarray()}")


    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    #print(f"DCG@{k}: {DCG}")
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    with np.errstate(divide='ignore', invalid='ignore'):
        ndcg = np.divide(DCG, IDCG, out=np.zeros_like(DCG), where=IDCG != 0)

    return ndcg


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)

    #recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    denom = np.minimum(k, X_true_binary.sum(axis=1))
    recall = np.where(denom == 0, 0, tmp / denom)


    return recall