# MyNextMovie — VAE Movie Recommendation System

A movie recommendation system built with a **Variational Autoencoder (VAE)** trained on the
[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset. Includes a
**Streamlit** web interface with personalized recommendations for authenticated users and
genre-filtered top-rated movies for guests.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red.svg)](https://streamlit.io/)
[![Release](https://img.shields.io/github/v/release/diego-terzi/Naml-Project)](https://github.com/diego-terzi/Naml-Project/releases/latest)

---

## How It Works

The VAE learns a compressed latent representation of each user's rating history. At inference
time, it reconstructs predicted ratings for all movies and recommends the top-ranked unseen
titles. The reconstruction loss is computed only on observed (non-zero) ratings, and a KL
divergence term regularizes the latent space.

**Evaluation (MovieLens 100K, 80/20 split):**

| Metric | @k=5 |
|--------|------|
| Precision | computed via `main.py` |
| Recall | computed via `main.py` |
| F1-score | computed via `main.py` |

---

## Project Structure

```
Naml-Project/
├── main.py                          # Train the VAE and evaluate metrics
├── data_cleaning/
│   └── data_cleaning.py             # Clean raw MovieLens data
├── model/
│   ├── vae_architecture.py          # Encoder and decoder definitions
│   └── vae_model.py                 # CustomVAE Keras model class
├── recommendation/
│   └── recommendations.py           # Streamlit app (MyNextMovie)
├── data/
│   ├── ml-100k/                     # Raw MovieLens data (not committed — download below)
│   └── cleaned/                     # Output of data_cleaning.py
└── saved_models/                    # Saved VAE weights (not committed)
```

---

## Installation

**Requirements:** Python 3.9+

```bash
git clone https://github.com/diego-terzi/Naml-Project.git
cd Naml-Project
python -m venv naml_venv
source naml_venv/bin/activate   # Windows: naml_venv\Scripts\activate
pip install tensorflow streamlit pandas scikit-learn numpy matplotlib
```

---

## Usage

### 1. Get the data

Download MovieLens 100K from https://grouplens.org/datasets/movielens/100k/ and place the
contents at `data/ml-100k/`. Then run the cleaning step:

```bash
python data_cleaning/data_cleaning.py
```

This produces `data/cleaned/ratings_clean.csv` and `data/cleaned/items_clean.csv`.

### 2. Train the VAE

```bash
python main.py
```

Training runs for up to 200 epochs with early stopping (patience=8). The trained model is
saved to `saved_models/vae_model.keras`. Precision@5, Recall@5, and F1@5 are printed at the end.

### 3. Launch the app

```bash
streamlit run recommendation/recommendations.py
```

Set the `APP_PASSWORD` environment variable to control the login password:

```bash
APP_PASSWORD=mysecret streamlit run recommendation/recommendations.py
```

The app opens at `http://localhost:8501`.

---

## Architecture

```
Input (n_items,)
      │
   Encoder
  Dense → Dense → [z_mean, z_log_var] → sampling (reparameterization)
      │
   Latent z (dim=64)
      │
   Decoder
  Dense → Dense → Output (n_items,)
```

**Loss:** `reconstruction_loss(observed ratings) + KL_divergence(z_mean, z_log_var)`

The reconstruction loss is a masked MSE that ignores zero-rated (unobserved) entries, so
the model only learns from movies the user has actually rated.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and contribution guidelines.

## Citation

If you use this project, please cite it using [CITATION.cff](CITATION.cff).
This project uses the MovieLens dataset — please also cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
> ACM Transactions on Interactive Intelligent Systems, 5(4). https://doi.org/10.1145/2827872

## License

MIT — see [LICENSE](LICENSE) for details.

> **Dataset note:** The MovieLens 100K dataset is not included in this repository.
> Download it directly from [GroupLens](https://grouplens.org/datasets/movielens/100k/)
> and place it at `data/ml-100k/` before running the pipeline.
