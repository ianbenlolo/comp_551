#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def cross_validation(X, y,model,splits=5,shuffle=False,**kwargs):
    kf = KFold(n_splits=splits,shuffle=shuffle)
    kf.get_n_splits(X)
    total_acc = 0
    split=0

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train,**kwargs)
        y_pred = model.predict(X_test)
        curr_acc = accuracy_score(y_pred, y_test)
        print('Split Accuracy: ', curr_acc)
        total_acc += curr_acc
        split+=1

    print('Avg Accuracy: ', total_acc/splits)
    return total_acc


def predict_and_save_test(data_TEST,vectorizer, model, le,fp='../submissions/submission.csv'):
    """
    Saves the submission file submission.csv
    Parameters
    ----------
    data_TEST:np.array
        the CLEANED data to be predicted
    vectorizer:sklearn.feature_extraction.text.VECTORIZER
        the same vectorizer used for training data
    model:
        model used in training
    le:sklearn.preprocessing.LabelEncoder()
        the label encoder used to encode the training targer data
    
    
    
    """
    import csv
    try:
        X_TEST_tf = vectorizer.transform(data_TEST[:,1])
        predictions_TEST = model.predict(X_TEST_tf)
        
        with open(fp, mode='w') as sub:
            writer = csv.writer(sub, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(['Id','Category'])
        
            for id_,category in zip(data_TEST[:,0],le.inverse_transform(predictions_TEST)):
                writer.writerow([id_,category])
    except Exception as e:
        print(e)
        return 0
    return 1
