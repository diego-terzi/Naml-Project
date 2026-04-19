# Contributing

Thank you for your interest in contributing to MyNextMovie!

## Environment Setup

**Requirements:** Python 3.9+

```bash
git clone https://github.com/diego-terzi/Naml-Project.git
cd Naml-Project
python -m venv naml_venv
source naml_venv/bin/activate  # Windows: naml_venv\Scripts\activate
pip install tensorflow streamlit pandas scikit-learn numpy matplotlib
```

## Getting the Data

This project uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/).
Download it and place the contents under `data/ml-100k/`, then run the data cleaning step:

```bash
python data_cleaning/data_cleaning.py
```

This produces `data/cleaned/ratings_clean.csv` and `data/cleaned/items_clean.csv`.

## Running the Project

**Train the VAE model:**
```bash
python main.py
```

**Launch the Streamlit app:**
```bash
streamlit run recommendation/recommendations.py
```

## Submitting Changes

1. Fork the repository and create a branch: `git checkout -b feature/your-feature`
2. Make your changes and test them
3. Open a pull request with a clear description of what you changed and why

## Reporting Issues

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml) to report
reproducibility issues, dataset loading errors, or model training problems.
