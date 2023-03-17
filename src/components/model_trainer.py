from sklearn.ensemble import (AdaBoostRegressor,
                            GradientBoostingRegressor,
                            RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from src.utils import save_object
from src.utils import evaluate_models
from dataclasses import dataclass
from xgboost import XGBRegressor
from src.logger import logging
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Dividiendo los datos input en entrenamiento y prueba")

            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boostig": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBoosting": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor() 
            }

            params ={
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Decision Tree":{
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Gradient Boostig":{
                    'learning_rate': [.1, .01,.05, .001],
                    'subsample': [.6, .7, .75, .8, .85, .9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression":{
                
                },
                "K-Neighbors Regressor":{
                    'n_neighbors': [5, 7, 9, 11]
                },
                "XGBoosting":{
                    'learning_rate': [.1, .01,.05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [.1, .01,.05, .001],
                    'iterations': [30, 50, 100],
                },
                "AdaBoost Regressor":{
                    'learning_rate': [.1, .01,.05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }
            
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
        
            # Para elegir el mejor modelo que puntuo del diccionario
            best_model_score = max(sorted(model_report.values()))

            # Para elegir el nombre del mejor modelo del diccionario
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index((best_model_score))
            ]

            # Para elegir el mejor model
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No se encontro un model suficientemente bueno")
        
            logging.info(f"Se encontro el mejor modelo en los conjuntos de entrenamiento y prueba")

            #preprocessing_obj = 
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)