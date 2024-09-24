import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib


class ModelObject:
    def __init__(self, model_name, model, params, best_params, evaluation_metrics, version):
        self.model_name = model_name
        self.model = model
        self.params = params
        self.best_params = best_params
        self.evaluation_metrics = evaluation_metrics
        self.version = version

    def log_details(self):
        log_message = f"Model: {self.model_name} (Version: {self.version})\n"
        log_message += f"Initial Parameters: {self.params}\n"
        log_message += f"Best Parameters after tuning: {self.best_params}\n"
        log_message += f"Evaluation Metrics: {self.evaluation_metrics}\n"
        return log_message

    def save(self, save_path):
        joblib.dump(self, save_path)
        print(f"Model saved at: {save_path}")

class Dataset:
    def __init__(self):
        self.data = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def generate_random_data(self, num_samples=1000):
       
        np.random.seed(42)
        ages = np.random.randint(18, 65, size=num_samples)
        activity_levels = np.random.rand(num_samples) 
        
        calories_intake = (ages * 5) + (activity_levels * 200) + np.random.normal(0, 50, size=num_samples)

      
        self.data = pd.DataFrame({
            'User_ID': np.arange(1, num_samples + 1),
            'Age': ages,
            'Activity_Level': activity_levels,
            'Calories_Intake': calories_intake
        })
        
        self.target = self.data['Calories_Intake']
        self.data.drop(['User_ID', 'Calories_Intake'], axis=1, inplace=True)

    def preprocess(self):
       
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=0.2, random_state=42
        )


class ModelSelector:
    def __init__(self):
        self.models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(),
            'GradientBoosting': GradientBoostingRegressor()
        }
        self.best_model_object = None
        self.version = 1 

    def hyperparameter_tuning(self, model, param_grid, X_train, y_train):
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_

    def select_model(self, X_train, y_train, X_test, y_test):
        
        param_grids = {
            'RandomForest': {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 7]},
            'GradientBoosting': {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1, 0.2]},
            'LinearRegression': {}  
        }

        best_score = float('inf')
        for model_name, model in self.models.items():
            print(f"Tuning {model_name}...")
            tuned_model, best_params = self.hyperparameter_tuning(model, param_grids[model_name], X_train, y_train)
            
           
            y_pred = tuned_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            evaluation_metrics = {"mse": mse, "r2": r2}

            print(f"{model_name} Test MSE: {mse}, R²: {r2}")

           
            if mse < best_score:
                best_score = mse
                self.best_model_object = ModelObject(
                    model_name=model_name,
                    model=tuned_model,
                    params=param_grids[model_name],
                    best_params=best_params,
                    evaluation_metrics=evaluation_metrics,
                    version=self.version
                )

        print(f"Best Model: {self.best_model_object.model_name}")
        return self.best_model_object

    def save_best_model(self):
        if self.best_model_object:
            
            save_path = f"models/{self.best_model_object.model_name}_v{self.version}.pkl"
            self.best_model_object.save(save_path)
            self.version += 1  

class AutoMLPipeline:
    def __init__(self):
        self.dataset = Dataset()
        self.model_selector = ModelSelector()

    def run(self):
       
        print("Generating Random Data...")
        self.dataset.generate_random_data(num_samples=1000) 
        self.dataset.preprocess()


        print("Selecting the best model...")
        best_model = self.model_selector.select_model(
            self.dataset.X_train, self.dataset.y_train, 
            self.dataset.X_test, self.dataset.y_test
        )

     
        self.model_selector.save_best_model()



if __name__ == "__main__":
    pipeline = AutoMLPipeline()
    pipeline.run()
