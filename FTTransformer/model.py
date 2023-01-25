import pandas as pd
import time
from configs import *
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import tensorflow_addons as tfa
from sklearn.metrics import (
        roc_auc_score, 
        average_precision_score, 
        accuracy_score, 
        f1_score,
        classification_report
        )
import numpy as np
from tabtransformertf.models.fttransformer import FTTransformerEncoder, FTTransformer
from tabtransformertf.utils.preprocessing import df_to_dataset
class FT_Transformer():
    def __init__(self,
                c_feature: list, 
                X_train: pd.DataFrame, 
                n_feature: list, 
                hyperparameters: dict, 
                **kwargs) -> None:
        self.hyp = hyperparameters
        ft_linear_encoder = FTTransformerEncoder(
            numerical_features = n_feature,
            categorical_features = c_feature,
            numerical_data = X_train[NUMERIC_FEATURES].values,
            categorical_data = X_train[CATEGORICAL_FEATURES].values,
            y = None,
            numerical_embedding_type=self.hyp['numerical_embedding_type'],
            embedding_dim=self.hyp['embedding_dim'],
            depth=self.hyp['depth'],
            heads=self.hyp['heads'],
            attn_dropout=self.hyp['attn_dropout'],
            ff_dropout=self.hyp['ff_dropout'],
            explainable=self.hyp['explainable']
        )
        self.model = FTTransformer(
                encoder=ft_linear_encoder,
                out_dim=self.hyp['out_dim'],
                out_activation=self.hyp['out_activation'],
            )
        
    def train(self,
            train_data: pd.DataFrame, 
            val_data: pd.DataFrame, 
            test_data: pd.DataFrame=None, 
        ):

        name = f"{self.hyp['numerical_embedding_type']}_fft_transform_{time.time()}"
        # DF to dataset
        train_dataset = df_to_dataset(train_data[FEATURES + [LABEL]], LABEL)
        val_dataset = df_to_dataset(val_data[FEATURES + [LABEL]], LABEL, shuffle=False)  # No shuffle
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Get optimizer
        self._get_optimizer()
        self.model.compile(
            optimizer = self.optimizer,
            loss = {"output":self.loss, "importances": None},
            metrics= {"output": self.metrics, "importances": None}
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
        y_predict = (test_preds['output'].ravel() > 0.5).astype(int)
        print("Test ROC AUC:", np.round(roc_auc_score(test_data[LABEL], test_preds['output'].ravel()), 4))
        print("Test PR AUC:", np.round(average_precision_score(test_data[LABEL], test_preds['output'].ravel()), 4))
        print("Test Accuracy:", np.round(accuracy_score(test_data[LABEL], y_predict), 4))
        print("F1 score:",np.round(f1_score(test_data[LABEL], y_predict, average='macro'), 4))
        print(classification_report(test_data[LABEL], y_predict))


    def inference(self, feature):
        if type(feature) == dict:
            df_input = pd.DataFrame(feature, index=[0])
        else:
            df_input = feature
        predict_dataset = df_to_dataset(df_input[FEATURES + [LABEL]], LABEL)
        preds = self.model.predict(predict_dataset)
        return preds
    

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