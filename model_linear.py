# -*- coding: utf-8 -*-


import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pickle

data_file = 'app/Warrior_Wild Pirate WarriorVsWarrior_Wild Pirate Warrior_Play RandomVsPlay Random_feature_fh_0_29_50.log'

def get_fea_label(info):
    X_game = []
    y_game = []

    fea0 = list(map(float, info['playerActive'].split('|')))
    fea1 = list(map(float, info['playerOpposite'].split('|')))

    #print(fea0, len(fea0))
    #print(fea1, len(fea1))

    X_game.append(fea0 + fea1)
    y_game.append(1 if int(info['Active']) == info['winner'] else -1)

    return X_game, y_game
            

def load_data(file_name, is_discounted):
    X, y = [], []
    data_dict = defaultdict(dict)
    with open(file_name, 'r') as f:
        for line in f:
            if line[0] != '{':
                continue            
            info = eval(line)           
            #print(info)
            X_game, y_game = get_fea_label(info)
            X += X_game
            y += y_game
            #break

    X = np.stack(X, axis=0)
    y = np.array(y)   
    print(X.shape, y.shape)   
    return X, y
                
if __name__ == '__main__':
    
    is_discounted = False
    X, y = load_data(data_file, is_discounted)
    
    lr = LogisticRegression(max_iter=500, verbose=0)  
    #lr = LinearRegression()
    lr.fit(X, y)
    res = mean_squared_error(lr.predict(X), y)

    coef = lr.coef_.flatten()
    print(','.join(['{:.3f}'.format(c) for c in coef]))
    print('res: ', res)
    print(np.sum(lr.predict(X) == y) / X.shape[0])

    
    
