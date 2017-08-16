# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:08:13 2017

@author: sjxn2423

解析MetaStone里提取的每回合结束时的双方的状态信息，提取特征向量，并尝试对应上不同的标签：
1. 根据最后的胜负，胜利方每个特征向量都对应标签 1，失败方每个特征向量都对应标签 -1
2. 根据距离结束的回合数，对标签乘上一定的discount，类似于RL中的discounted reward

使用GameStateValue 对局产生的数据为什么效果这么差？

使用训练得到的Linear状态评估函数到GreedyBestMove中，名字GreedyBestMoveLinear，然后和原来的GreedyBestMove进行1000局对局评估
正负1 binary 标签， LogisticRegression模型
1. randomPlay_1000games数据，对称提取特征，胜率 49.5%
2. GameStateValue_300games数据，对称提取特征，胜率 41.1%
3. randomPlay_1000games数据，不对称提取特征，胜率 51.6%      后面只考虑不对称提取特征这种方式
4. GameStateValue_300games数据，不对称提取特征，胜率 43.5%   

discounted real valued标签， 线性回归模型
1. randomPlay_1000games数据，对称提取特征，胜率 45.9%    gamma=0.9
2. GameStateValue_300games数据，对称提取特征，胜率 42.7%  gamma=0.95
3. randomPlay_1000games数据，不对称提取特征，胜率 43.8%     gamma=0.9， 胜率 52.1%    gamma=0.99  
（将上面52.1%这个LinearHuristic用于GameStateValue， VS GreedyBestMove 胜率73%， VS 用同一个LinearHuristic的GreedyBestMoveLinear 胜率 61%，说明加深搜索能提升性能）
（但自带的基于ThreatBasedHuristic的GameStateValue，VS GreedyBestMove 胜率76%， VS LinearHuristic的GreedyBestMoveLinear 胜率 68%，说明还是要更好一点）
4. GameStateValue_300games数据，不对称提取特征，胜率 50.9%  gamma=0.95

discounted real valued标签， 神经网络MLP模型 (目前实验来看，MLP的Hidden层不能太大，整体效果也没有超过linear model)
1. randomPlay_1000games数据，不对称提取特征，胜率 32.1%    gamma=0.99 (效果远不及简单的Linear Model)  hidden_layer_sizes=(100,)
调整模型参数：
hidden_layer_sizes=(25,)   VS GreedyBestMove 胜率47.2% 
hidden_layer_sizes=(10,)   VS GreedyBestMove 胜率54%   VS GreedyBestMoveLinear(上面52.1%的LinearHuristic)  胜率 49.2%
hidden_layer_sizes=(5,)    VS GreedyBestMove 胜率53.7%  VS GreedyBestMoveLinear  胜率  49.6%  (尝试对输入log变换 X = np.log(X + 1)，VS GreedyBestMove 胜率47% 效果变差)
hidden_layer_sizes=(1,)    VS GreedyBestMove 胜率48.1%
hidden_layer_sizes=(1,)  activation='identity'  VS GreedyBestMove 胜率49%
                   

待尝试用GreedyBestMove产生数据
1. GreedyBestMove_1000games数据，不对称提取特征，胜率 54.2%

# 尝试不同的gamma （之前偶然设置成0.8后效果极差，胜率0.5%，不确定是不是偶然，对于gamma这么敏感？）
# GreedyBestMove_1000games数据，discounted real valued标签，不对称提取特征
1. gamma=0.8，胜率 0.4%
2. gamma=0.85，胜率 4.8%
3. gamma=0.9，胜率 26.6%
4. gamma=0.95，胜率 51.5%
5. gamma=0.99，胜率 55.1%   没想到gamma的影响如此之大，整体来看基本上discount=0.99或者直接用正负1标签最好

# 尝试增加训练数据 （效果反而下降了一些，所以直接增加数据可能帮助不大）
1. randomPlay_4000games数据，不对称提取特征，正负1标签，胜率 48.9%  
2. GreedyBestMove_8000games数据，不对称提取特征，正负1标签，胜率 58% (这个效果有所提升)
3. GreedyBestMove_8000games数据，不对称提取特征，discounted标签，gamma=0.99，胜率 51%


# 尝试使用英雄A的数据训练，使用英雄B对战评估 （GreedyBestMoveLinear vs GreedyBestMove）
# 训练 GreedyBestMove_8000games数据，Hunter， randomDeck, standard， vs Hunter目前最佳 58%
# 不同测试英雄胜率情况：（整体来看都比较差，说明目前得到的局面评估函数基本只适用于训练时的英雄）
1. warrior vs warrior 5.7%
2. Druid vs Druid  19.2%
3. rogue vs rogue  33%

# 尝试提取warrior vs warrior1000局（GreedyBestMove vs GreedyBestMove）数据进行训练
1. GreedyBestMove_1000games数据，不对称提取特征，正负1标签，胜率 51.7% （相比上面的5.7%大幅提升，说明不同英雄需要用自己的数据训练）
发现用Warrior数据训练的模型使用到hunter时也有49%胜率。。。但反过来运用胜率就奇低。。。

# 尝试混合不同英雄的对战数据进行训练 （似乎可以让不同英雄都达到不算太差的效果）
# GreedyBestMove, hunter1000 + warrior1000，不对称提取特征，正负1标签，胜率情况：
1. hunter vs hunter  63.3%  （意想不到，加了warrior的数据，hunter的效果更好了？）
2. warrior vs warrior  47.6%

# GreedyBestMove， hunter1000 + warrior1000 + Druid1000 + rogue1000，不对称提取特征，正负1标签，胜率情况：
1. hunter vs hunter     61.9%
2. warrior vs warrior   41.5%
3. Druid vs Druid       39.3%
4. rogue vs rogue       40.4%

单独使用混合数据中的Druid数据训练，然后测试 Druid vs Druid，胜率 24.4% （出乎意料，竟然更低）

单独使用混合数据中的rogue数据训练，然后测试 rogue vs rogue，胜率 27.4% （出乎意料，竟然更低）

#尝试使用训练出来的模型产生训练数据，循环迭代训练，看看能不能进一步改进模型效果
#使用randomPlay_1000games数据，不对称提取特征，胜率 51.6% 模型生成1000局对局数据GreedyBestMoveLinear_1000games，再训练局面评估函数
 胜率 10.7%, 无法理解的差。。。。 重新提取数据试了一次，胜率更低了。。。

# 尝试提取中间每次action之后的局面数据，而不只是turn结束时的情况

# 尝试根据最后的Hp差值来设定标签
# 看训练得到的coef，基本权重都在Hp维度上，直观感觉不会太好
# 胜率 2.5%，果然效果很差

将supervised训练的模型作为RL的初始模型


"""

import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#import cPickle as pickle
import pickle
from sklearn.neural_network import MLPRegressor
#from sklearn2pmml import PMMLPipeline
#from sklearn2pmml import sklearn2pmml


#data_file = 'HunterVsHunter_randomDeck_randomPlay_1000games.log'
#data_file = 'HunterVsHunter_randomDeck_GameStateValue_300games.log'
#data_file = 'HunterVsHunter_randomDeck_GreedyBestMove_1000games.log'
#data_file = 'HunterVsHunter_randomDeck_randomPlay_4000games.log'
#data_file = 'HunterVsHunter_randomDeck_GreedyBestMove_8000games.log'
#data_file = 'WarriorVsWarrior_randomDeck_GreedyBestMove_1000games.log'
#data_file = 'HunterandWarrior_randomDeck_GreedyBestMove_both1000game.log'
#data_file = 'Hunter_Warrior_Druid_Rogue_randomDeck_GreedyBestMove_all1000game.log'
#data_file = 'DruidVsDruid_randomDeck_GreedyBestMove_1000games.log'
#data_file = 'RogueVsRogue_randomDeck_GreedyBestMove_1000games.log'
#data_file = 'HunterVsHunter_randomDeck_GreedyBestMoveLinear_1000games.log'  #尝试使用训练出来的模型产生训练数据，循环迭代训练
#data_file = 'WarriorVsWarrior_basicDeck_randomPlay_1000games.log'  #在自己构造的简单卡牌组上实验
# ------------------------------------------------------------------------------------------
#data_file = 'HunterVsHunter_randomDeck_randomPlay_1000games.log'
data_file = 'app/HunterVsHunter_randomDeck_Play Random_50000.log'
#data_file = 'app/HunterVsHunter_randomDeck_Play Random_fea0_50000.log'

def feature_filter(data_X):
    return np.delete(data_X, [3, 12, 18, 27], 1)

def get_fea_label(data_dict, game_hash, total_turn, winner):
    """提取完整一局的特征数据和对应标签"""
    X_game = []
    y_game = []
    data = data_dict[game_hash]
    for turn, fea in data.items():
        if winner == 0:
            X_game.append(fea[0] + fea[1])
            y_game.append(1)

        else:
            X_game.append(fea[0] + fea[1])
            y_game.append(-1)

    total_turn = len(X_game)
    #res_X_game = X_game
    #res_y_game = y_game
    res_X_game = [X_game[2], X_game[int(0.5 * total_turn)], X_game[int(0.8 * total_turn)]]
    res_y_game = [y_game[2], y_game[int(0.5 * total_turn)], y_game[int(0.8 * total_turn)]]
    return res_X_game, res_y_game
            
        
def get_fea_label_discount(data_dict, game_hash, total_turn, winner):
    """提取完整一局的特征数据和对应标签
    考虑discounted reward"""
    gamma = 0.99
    X_game = []
    y_game = []
    data = data_dict[game_hash]
    for turn, fea in data.items():
        if winner == 0:
            X_game.append(fea[0] + fea[1])
            y_game.append(gamma**(total_turn-turn))

        else:
            X_game.append(fea[0] + fea[1])
            y_game.append(-gamma**(total_turn-turn))

    res_X_game = [X_game[2], X_game[int(0.5 * total_turn)], X_game[int(0.8 * total_turn)]]
    res_y_game = [y_game[2], y_game[int(0.5 * total_turn)], y_game[int(0.8 * total_turn)]]
#    y_game = [10*y for y in y_game]
    return res_X_game, res_y_game

def load_data(file_name, is_discounted):
    X, y = [], []
    data_dict = defaultdict(dict)
    with open(file_name, 'r') as f:
        for line in f:
            if line[0] != '{':
                continue            
            info = eval(line)           
            if 'winner' in info: #一局结束, 提出完整一局的特征数据和对应标签
                if is_discounted:
                    X_game, y_game = get_fea_label_discount(data_dict, info['GameHash'], info['Turn'], info['winner'])
#                    X_game, y_game = get_fea_label_HpDiff(data_dict, info['GameHash'], info['Turn'], info['winner'])  # Hpdiff
                else:
                    X_game, y_game = get_fea_label(data_dict, info['GameHash'], info['Turn'], info['winner'])
                X += X_game
                y += y_game
            else:
                fea0 = list(map(float, info['player0'].split('|')))
                fea1 = list(map(float, info['player1'].split('|')))
                data_dict[info['GameHash']][info['Turn']] = (fea0, fea1)

    X = np.stack(X, axis=0)
    y = np.array(y)   
    X = feature_filter(X)    
    total_data = np.hstack([X, y[..., None]])
    #print(total_data.shape)
    np.random.shuffle(total_data)
    X = total_data[:, :-1]
    y = total_data[:, -1]  
    print(X.shape, y.shape)   
    return X, y
                
if __name__ == '__main__':
    
    is_discounted = False
    X, y = load_data(data_file, is_discounted)
#    X[X<0] = 0
#    X = np.log(X + 1)  # log变换
    print(X[0, :])
    lr = LogisticRegression(max_iter=500, verbose=0)  
    #lr = LinearRegression()
    lr.fit(X, y)
    res = mean_squared_error(lr.predict(X), y)

    # 发现因为原始特征数据是对称构造的，训练得到的模型coef也是对称的，这样的话相当于特征是两个player的特征向量的对应差值
    coef = lr.coef_.flatten()
    print(','.join(['{:.3f}'.format(c) for c in coef]))
    print('res: ', res)
    print(np.sum(lr.predict(X) == y) / X.shape[0])
#    # 尝试MLP模型，并以PMML文件形式输出
#    mlp = MLPRegressor(hidden_layer_sizes=(5,), learning_rate_init=0.001, tol=1e-4, verbose=True, random_state=1)
#    mlp.fit(X, y)
#    print mean_squared_error(mlp.predict(X), y)
#
#    model_file = 'random1000_binary_discounted_mlp.model'
#    pmml_file = 'random1000_binary_discounted_mlp.pmml'    
#    pickle.dump(mlp, open(model_file, 'wb'))     
    # dump the model to PMML file
#    mlp_pipeline = PMMLPipeline([("MLP_model", mlp)])
#    sklearn2pmml(mlp_pipeline, pmml_file, with_repr = True)

    
    
