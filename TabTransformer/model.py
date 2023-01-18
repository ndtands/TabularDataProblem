from tabtransformertf.models.tabtransformer import TabTransformer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import build_categorical_prep, df_to_dataset
import tensorflow as tf
import numpy as np
from configs import *
import tensorflow_addons as tfa
import time
from pathlib import Path
import pandas as pd

from sklearn.metrics import (
        roc_auc_score, 
        average_precision_score, 
        accuracy_score, 
        f1_score,
        classification_report
        )

class Tab_Transformer():
    def __init__(self, c_feature: list, vocab_c_feature: dict, n_feature: list, hyperparameters: dict, **kwargs):
        category_prep_layers = build_categorical_prep(
                vocab_json=vocab_c_feature,
                categorical_features=c_feature
                )
        self.hyp = hyperparameters
        self.model = TabTransformer(
            numerical_features = n_feature,
            categorical_features = c_feature,
            categorical_lookup=category_prep_layers,
            embedding_dim=self.hyp['embedding_dim'],
            out_dim=self.hyp['out_dim'],
            out_activation=self.hyp['out_activation'],
            depth=self.hyp['depth'],
            heads=self.hyp['heads'],
            attn_dropout=self.hyp['attn_dropout'],
            ff_dropout=self.hyp['ff_dropout'],
            mlp_hidden_factors=self.hyp['mlp_hidden_factors'],
            use_column_embedding=self.hyp['use_column_embedding'],
        )
        

    def train(self, 
        train_data: pd.DataFrame, 
        val_data: pd.DataFrame, 
        test_data: pd.DataFrame=None,
        ):
        name = f"tabtransform_{time.time()}"
        # DF to dataset
        train_dataset = df_to_dataset(train_data[FEATURES + [LABEL]], LABEL)
        val_dataset = df_to_dataset(val_data[FEATURES + [LABEL]], LABEL, shuffle=False)  # No shuffle
        
        # Get optimizer
        self._get_optimizer()
        self.model.compile(
            optimizer = self.optimizer,
            loss = self.loss,
            metrics= self.metrics
        )

        path_checkpoint = f'artifact/weights/{name}/'
        history_path = f'{PATH_HISTORY}/{name}.pkl' 
        Path(path_checkpoint).mkdir(parents=True, exist_ok=True)
        early_callback = EarlyStopping(monitor="val_loss", mode="min", patience=self.hyp['patience'], restore_best_weights=True)
        checkpoint_callback = ModelCheckpoint(filepath=path_checkpoint,
                                    save_weights_only=True,
                                    verbose=1)
        callback_list = [early_callback, checkpoint_callback]
        print(">> Training ................")

        self.history = self.model.fit(
            train_dataset, 
            epochs=self.hyp['epochs'], 
            validation_data=val_dataset,
            callbacks=callback_list,
            verbose=1
        )
        self.save_history(history_path)

        # Testing model
        if test_data is not None:
            self.test_model(test_data)
            
        return self.history

    def test_model(self, 
            test_data: pd.DataFrame
            ) -> None:
        test_dataset = df_to_dataset(test_data[FEATURES + [LABEL]], shuffle=False)
        test_preds = self.model.predict(test_dataset)
        y_predict = (test_preds.ravel() > 0.5).astype(int)
        print("Test ROC AUC:", np.round(roc_auc_score(test_data[LABEL], test_preds.ravel()), 4))
        print("Test PR AUC:", np.round(average_precision_score(test_data[LABEL], test_preds.ravel()), 4))
        print("Test Accuracy:", np.round(accuracy_score(test_data[LABEL], y_predict), 4))
        print("F1 score:",np.round(f1_score(test_data[LABEL], y_predict, average='macro'), 4))
        print(classification_report(test_data[LABEL], y_predict))

    def inference(self, feature: dict):
        df_input = pd.DataFrame(feature, index=[0])
        predict_dataset = df_to_dataset(df_input[FEATURES + [LABEL]], LABEL)
        test_preds = self.model.predict(predict_dataset)
        return test_preds.tolist()[0][0]

    def _get_optimizer(self) -> None:
        self.optimizer = tfa.optimizers.AdamW(
                            learning_rate=self.hyp['lr'], weight_decay=self.hyp['weight_decay']
                        )
        self.metrics = [tf.keras.metrics.AUC(name="PR AUC", curve='PR')]
        self.loss =  tf.keras.losses.BinaryCrossentropy()

    def load_model(self, path: str) -> None:
        self.model.load_weights(path)
    
    def save_model(self, path: str) -> None:
        self.model.save_weights(path)
    
    def save_history(self, path: str) -> None:
        import pickle
        with open(path, 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)

    