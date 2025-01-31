# Adult Income Dataset Analysis

## Description

This project focuses on data preprocessing and analysis of the **Adult Income Dataset** from the UCI Machine Learning Repository. It involves handling missing values, exploratory data analysis, feature engineering, and applying preprocessing pipelines to prepare the dataset for machine learning models.

## Features

- Data extraction from UCI ML repository
- Handling missing values
- Exploratory data analysis (EDA) with visualizations
- Feature engineering using preprocessing pipelines
- Data normalization and transformation

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn ucimlrepo
```

## Dataset

The **Adult Income Dataset** is retrieved from the UCI Machine Learning Repository using `ucimlrepo`. It contains demographic and economic attributes for individuals, used to predict income level.

## Usage

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo

# Fetch dataset
adult = fetch_ucirepo(id=2)
X = adult.data.features
y = adult.data.targets

# Handling missing values
X = X.replace('?', np.nan)

# Exploratory Data Analysis
X.hist(figsize=(24, 16))
plt.show()

# Preprocessing Pipelines
num_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
```

## Project Tasks

1. **Data Import:** Fetch dataset using `ucimlrepo`.
2. **Data Exploration:** Analyze dataset structure, missing values, and distributions.
3. **Data Cleaning:** Replace missing values and handle categorical variables.
4. **Feature Engineering:** Apply preprocessing transformations using Scikit-learn pipelines.
5. **Data Visualization:** Generate histograms and summary statistics.

## License

This project is for educational purposes and follows open-source principles.

## Author

**Mustafa Yildirim**

## Acknowledgments

- UCI Machine Learning Repository for the dataset.
- Scikit-learn and Pandas for data processing.
- Matplotlib and Seaborn for visualizations.

