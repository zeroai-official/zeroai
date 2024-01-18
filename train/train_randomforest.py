from sklearn.ensemble import RandomForestRegressor
import os
from preprocess_data import preprocess_data
import json
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from joblib import dump

current_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_path))
config_path = os.path.join(root_path, 'config.json')
data_path = os.path.join(root_path, 'dataset/data')

with open(config_path) as f:
    config = json.load(f)

symbol = config['symbol']
data_frequency = config['data_frequency']
modeltype = config['modeltype']
test_size = config['test_size']

features, target = preprocess_data(
    symbol, f'{data_path}/{symbol}/{symbol}_{data_frequency}.csv', 100, data_frequency, 'randomforest')

features_train, features_test = features[:-test_size], features[-test_size:]
target_train, target_test = target[:-test_size], target[-test_size:]

# # Define the parameter distribution to search
# param_dist = {
#     'n_estimators': randint(100, 1000),
#     'max_depth': [None] + list(randint(1, 20).rvs(size=10)),
#     'min_samples_split': randint(2, 10),
#     'min_samples_leaf': randint(1, 10)
# }

# model = RandomForestRegressor(random_state=48)

# # Create a RandomizedSearchCV object
# random_search = RandomizedSearchCV(
#     estimator=model, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=48, n_jobs=-1)

# # Perform the random search
# random_search.fit(features_train, target_train)

best_params = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 982} 
print(best_params)

model = RandomForestRegressor(n_estimators=best_params['n_estimators'], 
                              max_depth=best_params['max_depth'], 
                              min_samples_split=best_params['min_samples_split'], 
                              min_samples_leaf=best_params['min_samples_leaf'], 
                              verbose=1,
                              random_state=48)

model.fit(features_train, target_train)

dump(model, f'../models/{symbol}_{data_frequency}_randomforest.joblib', compress=5) 

predictions = model.predict(features_test)

results = pd.DataFrame({
    'Actual': target_test,
    'Predicted': predictions
})

print(results)

# best params {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 982} 