# Package Template


# Environment setup
- Install pyenv
- Install poetry

```bash
pyenv install 3.9.4
pyenv local 3.9.4
```
```bash
poetry config virtualenvs.in-project true
poetry add --dev pytest
poetry add numpy pandas
```

Using isort
```bash
isort abimca
```

# Save credentials for an hour:
git config --local credential.helper 'cache --timeout=3600'

# Install package from git#https with poetry:
poetry add git+https://git.tu-berlin.de/jkoehne/mts2p.git#main

mts2p = {git = "https://git.tu-berlin.de/jkoehne/mts2p.git", rev = "develop"}


# Features
- Analyse an unknown multivariate time series data
    - The basic an main use case of this application can identify unique subsequences in unlabeled data. A dataframe with the data can be passed and a percentage or number of instances for training will be used based on the parameter passed to the datamodule.

- Generate an autoencoder per class
    - When having labeled multivariate time series data, the optional argument 'train_based_on_class_labels = True' generates Autoencodeder models per class, which are then used as the usual subsequence models for inferencing.

- Analysis of unlabeled univariate time series data
    - With the preprocessing toolbox, new features can be generated. These features are moving average, and 

