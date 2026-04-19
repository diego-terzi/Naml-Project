# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-19

### Added
- Variational Autoencoder (VAE) trained on MovieLens 100K dataset
- Encoder-decoder architecture with latent dimension of 64
- Custom reconstruction loss computed only on non-zero (observed) ratings
- KL divergence regularization (beta-VAE formulation)
- Early stopping callback to prevent overfitting
- Evaluation metrics: Precision@5, Recall@5, F1@5
- Training loss visualization saved as `training_loss_plot.png`
- Streamlit web interface ("MyNextMovie") with two modes:
  - Authenticated user mode: VAE-powered personalized recommendations
  - Guest mode: genre-filtered top-rated movie recommendations
- Persistent seen-movies tracking per user via JSON file
- Data cleaning pipeline producing `ratings_clean.csv` and `items_clean.csv`
- Model serialization to `saved_models/vae_model.keras`
