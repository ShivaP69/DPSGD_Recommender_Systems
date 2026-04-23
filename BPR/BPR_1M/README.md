# BPR Model

Our BPR implementation provides a Bayesian Personalized Ranking (BPR) model for implicit feedback recommendation tasks, designed using PyTorch. It is compatible with SGD-based optimizers, including Differentially Private SGD (DPSGD) from Opacus.
The BPR model learns latent user and item embeddings and optimizes them to rank positive items higher than negative ones. This is done using a pairwise ranking loss, not a pointwise rating prediction.

## Description(model.py):

- User Embedding: Each user is mapped to a latent vector of dimension latent_dim.

- Item Embedding: Each item is also mapped to a latent vector.

- Forward Pass: Given triplets (user, positive_item, negative_item), the model computes:

        A positive score via dot product between the user and the positive item.

        A negative score via dot product between the user and the negative item.
- These scores are passed to a custom BPR loss function that encourages the model to rank the positive item higher than the negative one.
    - Example Output:

     ```
     pos_scores, neg_scores = model(user_ids, pos_item_ids, neg_item_ids)
    loss = bpr_loss(pos_scores, neg_scores)
  
     ```  
     The model does not include the loss directly — it is computed externally using:
     ```
     def bpr_loss(pos_scores, neg_scores):
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
    ```
# Running the training script

```bash
python training.py --learning_rate 0.001 --batch_size 256 --num_epochs 400 --DPSGD False --k 10 --calibration False --latent_dim 5

```
To have a complete view of arguments, you can check the training.py script. 

Install the following Python packages before running the code:
```bash
pip install numpy torch pandas scikit-learn scipy opacus pytorch_lightning
```
# Running the LDP training script

```bash
python3 training_DP.py --learning_rate 0.001 --privacy 0.1 --num_epochs 400 --batch_size 256 --k 10 
```
To have a complete view of arguments, you can check the training_DP.py script. 

Applying DP is handled during preparation of data: for detailed description, you can check: create_data_loader_DP.py script. 
