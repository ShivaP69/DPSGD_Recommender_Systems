# Variational Autoencoder for Collaborative Filtering with Differential Privacy

VAE implementation follows: [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814) (Liang et al. 2018) 
This repository contains a PyTorch implementation of a Variational Autoencoder (VAE) for collaborative filtering on the MovieLens dataset, with optional support for Differential Privacy (DP) using either standard DP or DP-SGD via Opacus.
The backbone of the VAE model follows: [vae-cf-pytorch](https://github.com/younggyoseo/vae-cf-pytorch).

# Requirements
Before running the code, install the following dependencies:
```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn scipy
pip install opacus tensorboard
```
# How to run
You can train the model using (for DPSGD):
```bash
python3 main.py --data /path/to/data --DPSGD "True" --DP "False" --read_data True

```
Or, for Local Differential Privacy:
```bash
python3 main.py --data /path/to/data --DP "True" --DPSGD "False" --read_data True

```
To train without differential privacy :
```bash
python3 main.py --data /path/to/data --DPSGD "False" --DP "False" --read_data True

```
# Arguments

| Argument               | Type    | Default      | Description                                                   |
| ---------------------- | ------- | ------------ | ------------------------------------------------------------- |
| `--data`               | `str`   | `''`         | Path to MovieLens-20M data                                    |
| `--lr`                 | `float` | `0.01`       | Learning rate                                                 |
| `--wd`                 | `float` | `0.0`        | Weight decay                                                  |
| `--batch_size`         | `int`   | `224`        | Batch size (will be dynamically adjusted internally)          |
| `--epochs`             | `int`   | `10`         | Number of training epochs                                     |
| `--total_anneal_steps` | `int`   | `200000`     | Steps for KL annealing                                        |
| `--anneal_cap`         | `float` | `0.2`        | Max annealing value                                           |
| `--seed`               | `int`   | `1111`       | Random seed                                                   |
| `--cuda`               | `flag`  | `False`      | Use GPU (CUDA) if available                                   |
| `--log-interval`       | `int`   | `100`        | Log interval for training                                     |
| `--save`               | `str`   | `'model.pt'` | Path to save the model                                        |
| `--DPSGD`              | `str`   | `"True"`     | Use DP-SGD (Opacus). Must set `--DP "False"`                  |
| `--DP`                 | `str`   | `"False"`    | Use standard Differential Privacy. Must set `--DPSGD "False"` |
| `--privacy`            | `float` | `0.1`        | Privacy budget (used if `--DP "True"` only)                   |
| `--noise_multiplier`   | `float` | `0.1`        | Noise multiplier for DP-SGD                                   |
| `--max_grad_norm`      | `float` | `2.0`        | Gradient clipping norm for DP-SGD                             |
| `--read_data`          | `bool`  | `True`       | Run data preparation step                                     |
| `--save_model`         | `bool`  | `False`      | Save the best model during training                           |

# Important Notes
- --DPSGD and --DP cannot be "True" at the same time.
- If --DPSGD "True" then --DP must be "False", and vice versa.
- Make sure --read_data True is set if you are switching between DP and non-DP runs (it re-processes the data appropriately).
