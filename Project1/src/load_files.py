import numpy as np
import csv
#wine_data_path = '../data/winequality-red.csv'
#breast_cancer_data_path = '../data/breast-cancer-wisconsin.data'
#breast_cancer_names_path = '../data/breast-cancer-wisconsin.names'


def load_wine(wine_data_path ='../data/winequality-red.csv', preprocess = 1,norm = True):
    with open(wine_data_path, 'r') as f:
        wines = list(csv.reader(f, delimiter=';'))

    wine_headers = wines[0]
    wines = np.array(wines[1:], dtype=np.float)
    if preprocess:
        for i in wines:
            if i[-1] >= 6:
                i[-1] = 1
            else:
                i[-1] = 0
    if norm:
        wines[:,:-1] = normalize(np.copy(wines[:,:-1]))
    return wines, wine_headers


def load_cancer(breast_cancer_data_path = '../data/breast-cancer-wisconsin.data' , preprocess = 1, norm = True):
    with open(breast_cancer_data_path, 'r') as f:
        cancer = list(csv.reader(f ))
    cancer_headers = ['id number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
    
    if preprocess:
        cancer_good = []
        j = 0
        for i,line in enumerate(cancer):
            try:
                cancer_good.append( [int(x) for x in line])
            except ValueError as e:
                #to get rid of the '?'s
                j += 1
                continue
            if cancer_good[i-j][-1] == 2:
                cancer_good[i-j][-1] = 0

            elif cancer_good[i-j][-1] == 4:
                cancer_good[i-j][-1] = 1
            else:
                print('Something weird. Check cancer data.',i)
    # return list into numpy array
    cancer = np.asarray(cancer_good)
    if norm:
        cancer = normalize(cancer)
 #   cancer = np.concatenate((cancer, y[np.newaxis:]), axis=1)

    return cancer, cancer_headers

def normalize(arr,axis = 0):
    """
    normalize (put in range [0,1]) ane array along an axis

    """
    min = arr.min(axis=axis) 
    minus_arr = arr.T - min[:,np.newaxis]

    norm_arr = minus_arr / minus_arr.T.max(axis =axis)[:,np.newaxis]
    return norm_arr.T



