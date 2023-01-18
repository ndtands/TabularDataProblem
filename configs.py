import json
PATH_TRAIN = 'dataset/train.csv'
PATH_VAL = 'dataset/val.csv'
PATH_TEST = 'dataset/test.csv'

PATH_WEIGHTS = 'artifact/weights'
PATH_HISTORY = 'artifact/history'
CATEGORICAL_FEATURES=['workclass', 'education', 'marital_status', 'occupation',
       'relationship', 'race', 'gender', 'native_country']
NUMERIC_FEATURES=['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
       'hours_per_week']
LABEL = 'income_bracket'
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

VOCAB=json.loads(open('vocab.json').read())