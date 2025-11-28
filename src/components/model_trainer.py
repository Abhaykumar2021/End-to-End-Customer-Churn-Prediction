import os
import sys
from dataclasses import dataclass

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model = Sequential()
            model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

            # Calculate Class Weights
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(enumerate(class_weights))
            logging.info(f"Class Weights: {class_weight_dict}")

            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            logging.info("Training the model")
            model.fit(
                X_train, y_train, 
                epochs=100, 
                batch_size=32, 
                validation_split=0.2, 
                class_weight=class_weight_dict,
                callbacks=[early_stop],
                verbose=1
            )

            logging.info("Model training completed")

            # Save the model using Keras save method
            model.save(self.model_trainer_config.trained_model_file_path)
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")

            y_pred_prob = model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype("int32")

            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logging.info(f"Model Metrics - Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1-Score: {f1}")
            
            return accuracy

        except Exception as e:
            raise CustomException(e,sys)
