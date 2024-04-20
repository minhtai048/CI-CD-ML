import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi
from scipy import stats
from datetime import datetime
from utils.preprocess import replace_categories
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

scaler = pickle.load(open('models/mm_encoder.pkl', 'rb'))
lmbda_charges = pickle.load(open('models/lmbda_price.pkl', 'rb'))

def load_and_process():
    data = pd.read_csv("data/metadata.csv")
    data = replace_categories(data)
    data['charges_fixed'], lmbda_price = stats.yeojohnson(data['charges'])
    X = data.drop(columns=['charges', 'charges_fixed'])
    y = data['charges_fixed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_and_evaluate():
    # prepare data for training
    X_train, X_test, y_train, y_test = load_and_process()

    # define model and stuffs for evaluation
    models= [LinearRegression(), DecisionTreeRegressor(), 
             RandomForestRegressor(), KNeighborsRegressor(n_neighbors=11), 
             AdaBoostRegressor(), XGBRegressor(), SVR()]
    scores= []
    test_score = []
    train_times = []
    names= []

    # training
    for model in models:
        start= time.time()
        scores.append(cross_val_score(model, X_train, y_train, 
                                      scoring= 'r2', cv= 5).mean())
        end = time.time()
        train_times.append(end-start)
        names.append(model.__class__.__name__)
    df= pd.DataFrame(scores, columns=['Score'], index= range(len(models)))
    df.insert(1, 'Time', pd.Series(train_times))
    df.insert(0, 'Model', pd.Series(names))

    best_score = 0
    best_model = models[0]
    # evaluate on test set
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        if best_score < score:
            best_score = score
            best_model = model
        test_score.append(score) 
    df['Test_Score'] = test_score
    return best_model, df

def saving_plotting(best_model, df):
    pickle.dump(best_model, open('models/best_model.pkl', 'wb'))
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = './history/' + current_time + '.png'
    dfi.export(df, save_path)
    dfi.export(df, './history/current_result.png')

if __name__ == "__main__":
    best_model, df = train_and_evaluate()
    saving_plotting(best_model, df)