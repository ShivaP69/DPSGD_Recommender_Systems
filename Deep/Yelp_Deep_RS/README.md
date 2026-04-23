# run the model

To run the NCF (Neural Collaborative Filtering) model, use the following command (for DPSGD cases)

```bash
    python3 main.py \
  --mlp_layer_sizes 64,32,16 \
  --DPSGD "True" \
  --learning_rate 0.001 \
  --max_grad_norm 1.0 \
  --noise_multiplier 1.1 \
  --mf_dim 8 \
  --num_epochs 100 \
  --Plot False \
  --batch_size 256 \
  --delta 1e-5 \
  --k 10 \
  --calibration 0 \
  --mlp_deep_fm_should_be_trained "False"
  ```


-mlp_layer_sizes: Comma-separated list of hidden layer sizes for the MLP (e.g., 64,32,16).

-DPSGD: Set to "True" to enable training with Differential Privacy using DP-SGD.

-learning_rate: Learning rate for the optimizer.

-max_grad_norm, --noise_multiplier, --delta: Parameters specific to DP-SGD. If DPSGD is "False", these can be ignored.

-mf_dim: Dimensionality of the matrix factorization latent factors.

-num_epochs: Number of training epochs.

-Plot: Set to True to enable plotting (requires matplotlib and seaborn).

-batch_size: Batch size for training.

-k: Number of top-K recommendations to consider.

-calibration: Use 1 to enable calibration, 0 to disable.

-mlp_deep_fm_should_be_trained: Set to "True" if MF and MLP should be pretrained.


Install the following Python packages before running the code:
```bash
pip install numpy torch pandas scikit-learn scipy opacus pytorch_lightning matplotlib seaborn
```
Note: matplotlib and seaborn are only required if plotting is enabled.

# LDP

To run the NCF (Neural Collaborative Filtering) model in LDP manner:

```bash
python3 training_DP.py --learning_rate 0.0005 --mlp_layer_sizes 64,32,16 --mf_dim 8 --privacy 0.1 --num_epochs 400 --load_model {True/False} --batch_size 256 --k 10
```
Privacy parameter represents epsilon value.

# Yelp Dataset Categories

For having Yelp categories, we crawled the following URL:
```
https://blog.yelp.com/businesses/yelp_category_list/
```
The details are provided in the "read_df_item_and_df_business" function (you need to have internet connection to crawl the categories). The categories are already collected and saved in "result.json". 


# Yelp dataset

You need "yelp_academic_dataset_business.json" and "yelp_academic_dataset_review.json" to run the codes from scratch. Otherwise, all the files we need are extracted from these JSON files, and no need to recreate them.