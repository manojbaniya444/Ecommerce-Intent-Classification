## Ecommerce Intent Classification
Intent classification model for Ecommerce customer queries. Intent is classified among 4 different classes:
- PRODUCT INQUIRY
- PAYMENT METHOD
- ORDER STATUS
- ORDER PRODUCT

This Project is organized so that,

- Easily Change the hyperparameters in [Config File](./config/config.yaml)
- Modify data transforms in [Dataset File](./src/data/dataset.py)

### Install the required libraries

```bash
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```bash
python -c "from src.models.model import TextClassifier; print('Success')"
```

### Train the model with the CONFIG.YAML file

```bash
python train.py --config config/config.yaml
```

### Run Test

```bash
python -m pytest
```

### Run the API Server

```bash
fastapi dev api/app.py
```

**Make Prediction**:

URL:  `http://localhost:8080/single-predict`

BODY PARAMETER: `{"text": "malai Redmi mobile ko barema bujhna man xa"}` 

RESPONSE: `{predicted_intent: "product_inquiry"}`

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