from tqdm import tqdm 
import tensorflow as tf
import pandas as pd
import pickle


def build_categorical_prep(vocab_json: dict, categorical_features=list):
    category_prep_layers = {}
    for c in tqdm(categorical_features):
        lookup = tf.keras.layers.StringLookup(vocabulary=vocab_json[c])
        category_prep_layers[c] = lookup

    return category_prep_layers

def df_to_dataset(
    dataframe: pd.DataFrame,
    target: str = None,
    shuffle: bool = True,
    batch_size: int = 512,
):
    df = dataframe.copy()
    if target:
        labels = df.pop(target)
        dataset = {}
        for key, value in df.items():
            value = value.to_numpy()
            dataset[key] = value[:, tf.newaxis]

        dataset = tf.data.Dataset.from_tensor_slices((dict(dataset), labels))
    else:
        dataset = {}
        for key, value in df.items():
            dataset[key] = value[:, tf.newaxis]

        dataset = tf.data.Dataset.from_tensor_slices(dict(dataset))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset

def read_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)

