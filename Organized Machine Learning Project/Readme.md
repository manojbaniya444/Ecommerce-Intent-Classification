## Text Classification
A simple text classification project.

- Easily Change the hyperparameters in [Config File](./config/config.yaml)
- Modify data transforms in [Dataset File](./src/data/dataset.py)

### Install the required libraries

```bash
pip install -r requirements.txt
```

### Train the model with the CONFIG.YAML file

```bash
python train.py --config config/config.yaml
```

### Run Test

```bash
python -m pytest
```

## Directory

```
Organized Machine Learning Project/
├── config/                             # config file store
│   └── config.yaml
├── data/                               # train.csv and val.csv file
│   └── train.csv
|   └── val.csv
├── checkpoints/                        # saved model
|   └── model.csv
├── notebooks/                          # testing notebooks
├── src/
│   ├── data/                           # dataset preparation
│   │   └── dataset.py
│   ├── models/                         # neural network architecture model
│   └── utils/                          # utils files
|   └── training/                       # code to train the model
|── logs/                               # logging info
|── Docker/                             # docker files
|── api/                                # api for inference to clients
├── tests/                              # Test file
├── train.py                            # main training file
└── requirements.txt                    # requirements file
```