
import pandas as pd
from scipy.sparse import csr_matrix

# Step 1: Create consistent mappings
all_users = [1, 2, 8, 5, 6]
all_items = [1, 2, 3, 4, 6, 9, 10]

profile2id = {uid: i for i, uid in enumerate(all_users)}
show2id = {sid: i for i, sid in enumerate(all_items)}

id2profile = {v: k for k, v in profile2id.items()}
id2show = {v: k for k, v in show2id.items()}

# Step 2: Numerize each subset
subset1_data = {
    'userId': [2, 2, 8, 8, 8, 8],
    'movieId': [1, 2, 3, 6, 9, 10],
    'rating': [5, 4, 5, 3, 2, 4]
}
subset1_df = pd.DataFrame(subset1_data)
subset1_df['uid'] = subset1_df['userId'].map(profile2id)
subset1_df['sid'] = subset1_df['movieId'].map(show2id)

subset2_data = {
    'userId': [5, 6],
    'movieId': [2, 4],
    'rating': [1, 1]
}
subset2_df = pd.DataFrame(subset2_data)
subset2_df['uid'] = subset2_df['userId'].map(profile2id)
subset2_df['sid'] = subset2_df['movieId'].map(show2id)
print(subset2_df)
# Step 3: Convert to CSR matrices
csr_subset1 = csr_matrix((subset1_df['rating'], (subset1_df['uid'], subset1_df['sid'])))
csr_subset2 = csr_matrix((subset2_df['rating'], (subset2_df['uid'], subset2_df['sid'])))

# Step 4: Extract original IDs from CSR matrix for Subset 1
row_index, col_index = csr_subset2.nonzero()[0][1], csr_subset2.nonzero()[0][0]  # Find first non-zero entry


original_user_id = id2profile[row_index]
original_movie_id = id2show[col_index]

print(f"Original userId from Subset  CSR (row {row_index}):", original_user_id)
print(f"Original movieId from Subset  CSR (col {col_index}):", original_movie_id)
print("\nCSR format components:")

print("data:", csr_subset2.data)
print("indices:", csr_subset2.indices)
print("indptr:", csr_subset2.indptr)

# Display the CSR matrix
print("CSR Matrix (in dense format):")
print(csr_subset2.toarray())