package net.demilich.metastone.game.behaviour.heuristic;

/**
 * Created by sjxn2423 on 2017/6/29.
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import net.demilich.metastone.game.Attribute;
import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.behaviour.GreedyOptimizeMoveLinear;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.cards.CardType;
import net.demilich.metastone.game.entities.heroes.Hero;
import net.demilich.metastone.game.entities.heroes.HeroClass;
import net.demilich.metastone.game.entities.minions.Minion;
import net.demilich.metastone.game.entities.minions.Summon;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SupervisedLinearHeuristic implements IGameStateHeuristic{

//    ################# Hunter #############################
//    正负1 binary 标签， 训练LogisticRegression模型得到权重
//    1. randomPlay_1000games数据，对称提取特征，胜率 49.5%
//    double[] coef = {0.099,-0.015,-0.104,0.093,0.127,0.049,0.052,0.063,0.072,0.028,0.026,0.005,0.000,-0.079,0.004,
//                    -0.099,0.015,0.104,-0.093,-0.127,-0.049,-0.052,-0.063,-0.072,-0.028,-0.026,-0.005,0.000,0.079,-0.004};

//    2. GameStateValue_300games数据，对称提取特征，胜率 41.1%
//    double[] coef = {0.135,0.028,-0.442,0.336,-0.529,0.431,-0.120,0.048,0.161,0.015,0.212,-0.013,0.000,-0.216,0.055,
//                    -0.135,-0.028,0.442,-0.336,0.529,-0.431,0.120,-0.048,-0.161,-0.015,-0.212,0.013,0.000,0.215,-0.055};

//    3. randomPlay_1000games数据，不对称提取特征，胜率 51.6%
//    double[] coef = {0.099,-0.004,-0.104,0.072,0.213,0.038,0.038,0.132,0.061,0.019,0.064,-0.010,0.000,-0.118,0.014,
//                    -0.098,0.025,0.110,-0.033,-0.038,-0.063,-0.063,0.009,-0.085,-0.037,0.017,-0.022,0.000,0.049,0.005};

//    4. GameStateValue_300games数据，不对称提取特征，胜率 43.5%
//    double[] coef ={0.133,-0.008,-0.429,0.344,-0.960,0.385,0.085,0.145,0.120,0.012,0.165,-0.022,0.000,-0.199,0.024,
//                    -0.138,-0.071,0.457,-0.075,0.186,-0.524,0.362,0.062,-0.203,-0.020,-0.228,-0.002,0.000,0.245,-0.083};

//    5. GreedyBestMove_1000games数据，不对称提取特征，胜率 54.2%
//    double[] coef = {0.116,0.142,-0.574,-0.088,-0.106,0.188,-0.053,-0.004,0.142,0.049,-0.181,-0.009,0.000,-0.581,0.059,
//                    -0.134,-0.157,0.467,0.640,-0.280,0.002,-0.034,-0.059,-0.136,-0.009,0.020,0.044,0.000,0.460,-0.056};

//    discounted real valued标签， 训练线性回归模型得到权重
//     1. randomPlay_1000games数据，对称提取特征，胜率 45.9%  gamma=0.9
//    double[] coef = {0.251,-0.012,-0.296,0.224,0.188,0.111,0.097,0.047,0.166,0.046,0.096,0.001,0.000,-0.066,0.012,
//                     -0.251,0.012,0.296,-0.224,-0.188,-0.111,-0.097,-0.047,-0.166,-0.046,-0.096,-0.001,-0.000,0.066,-0.012};

//    2. GameStateValue_300games数据，对称提取特征，胜率 42.7%   gamma=0.95
//    double[] coef = {0.349,0.095,-1.232,0.261,-0.463,0.587,-0.223,0.119,0.390,0.014,0.410,-0.025,-0.000,-0.578,0.127,
//                      -0.349,-0.095,1.232,-0.261,0.463,-0.587,0.223,-0.119,-0.390,-0.014,-0.410,0.025,0.000,0.578,-0.127};

//    3. randomPlay_1000games数据，不对称提取特征，胜率 43.8%  gamma=0.9
//    double[] coef = {0.243,0.001,-0.304,0.371,0.328,0.085,0.082,0.137,0.152,0.023,0.119,-0.014,0.000,-0.116,0.019,
//                    -0.259,0.023,0.292,0.076,-0.018,-0.142,-0.110,0.052,-0.181,-0.071,-0.064,-0.016,0.000,0.019,-0.006};
//    randomPlay_1000games数据，不对称提取特征，胜率 52.1%  gamma=0.99 （根据实验结果，gamma参数影响很大）
//    double[] coef = {0.031,-0.001,-0.030,0.035,0.079,0.009,0.008,0.051,0.017,0.004,0.018,-0.003,-0.000,-0.043,0.005,
//                    -0.030,0.008,0.034,-0.012,-0.021,-0.018,-0.017,-0.010,-0.025,-0.010,0.011,-0.007,0.000,0.021,0.001};

//    4. GameStateValue_300games数据，不对称提取特征，胜率 50.9%   gamma=0.95
//    double[] coef = {0.339,0.078,-1.220,0.330,-1.133,0.571,0.005,0.233,0.305,0.034,0.262,-0.037,0.000,-0.562,0.074,
//                    -0.355,-0.122,1.219,0.709,-0.310,-0.595,0.515,0.023,-0.470,-0.004,-0.505,0.007,0.000,0.615,-0.173};

//    discounted real valued标签，尝试不同的gamma （之前偶然设置成0.8后效果极差，胜率0.5%，不确定是不是偶然，对于gamma这么敏感？）
//     GreedyBestMove_1000games数据，discounted real valued标签，不对称提取特征
//    1. gamma=0.8，胜率 0.4%
//    double[] coef = {0.272,0.422,-1.220,-0.107,0.293,0.081,0.025,-0.259,0.222,0.099,-0.088,-0.005,-0.000,-0.482,0.077,
//                    -0.305,-0.484,1.080,0.346,-0.447,-0.025,-0.093,-0.032,-0.194,-0.022,-0.189,0.050,0.000,0.313,-0.055};

//    2. gamma=0.85，胜率 4.8%
//    double[] coef = {0.286,0.410,-1.338,-0.105,0.190,0.143,-0.001,-0.228,0.253,0.104,-0.178,-0.006,-0.000,-0.666,0.090,
//                    -0.322,-0.472,1.171,0.554,-0.462,-0.024,-0.095,-0.078,-0.221,-0.019,-0.148,0.062,0.000,0.465,-0.069};

//    3. gamma=0.9，胜率 26.6%
//    double[] coef = {0.296,0.400,-1.438,-0.222,0.050,0.227,-0.040,-0.175,0.288,0.109,-0.306,-0.009,-0.000,-0.943,0.109,
//                    -0.336,-0.456,1.233,0.922,-0.507,-0.017,-0.091,-0.137,-0.252,-0.014,-0.085,0.079,0.000,0.693,-0.090};

//    4. gamma=0.95，胜率 51.5%
//    double[] coef = {0.298,0.391,-1.491,-0.508,-0.140,0.346,-0.096,-0.085,0.325,0.114,-0.487,-0.015,-0.000,-1.369,0.137,
//                    -0.344,-0.433,1.225,1.541,-0.604,0.002,-0.077,-0.216,-0.286,-0.006,0.016,0.105,0.000,1.043,-0.123};

//    5. gamma=0.99，胜率 55.1%
//    double[] coef = {0.290,0.386,-1.463,-0.898,-0.341,0.474,-0.162,0.026,0.356,0.115,-0.690,-0.022,-0.000,-1.878,0.170,
//                    -0.341,-0.407,1.123,2.309,-0.742,0.031,-0.056,-0.303,-0.313,0.005,0.136,0.133,0.000,1.456,-0.165};

//    # 尝试增加训练数据
//    1. randomPlay_4000games数据，不对称提取特征，正负1标签，胜率 48.9% (效果相比 randomPlay_1000games数据反而下降了)
//    double[] coef = {0.098,-0.027,-0.089,0.077,0.058,0.068,0.057,0.041,0.078,0.040,0.085,0.006,0.000,-0.081,0.016,
//                    -0.103,0.028,0.078,-0.018,-0.046,-0.076,-0.052,-0.038,-0.082,-0.033,-0.103,-0.003,0.000,0.095,-0.008};

//    2. GreedyBestMove_8000games数据，不对称提取特征，正负1标签，胜率 58% (这个效果有所提升)
//    double[] coef = {0.139,0.130,-0.566,-0.052,0.045,0.114,-0.024,0.034,0.133,0.032,0.002,-0.028,0.000,-0.489,0.066,
//                    -0.134,-0.124,0.580,0.727,-0.135,-0.115,0.047,-0.035,-0.133,-0.035,0.015,0.024,0.000,0.516,-0.066};

//    3. GreedyBestMove_8000games数据，不对称提取特征，discounted标签，gamma=0.99，胜率 51%
//    double[] coef = {0.353,0.354,-1.485,-0.155,-0.003,0.265,-0.021,0.187,0.338,0.053,-0.069,-0.082,0.000,-1.567,0.201,
//                    -0.340,-0.338,1.526,1.742,-0.223,-0.272,0.093,-0.194,-0.332,-0.063,0.123,0.070,0.000,1.667,-0.203};

//    ##################### Warrior #############################
//    1. GreedyBestMove_1000games数据，不对称提取特征，正负1标签，胜率 51.7%
//    double[] coef = {0.098,-0.027,-0.089,0.077,0.058,0.068,0.057,0.041,0.078,0.040,0.085,0.006,0.000,-0.081,0.016,
//                    -0.103,0.028,0.078,-0.018,-0.046,-0.076,-0.052,-0.038,-0.082,-0.033,-0.103,-0.003,0.000,0.095,-0.008};

//    #####################  多英雄数据混合 (似乎大部分英雄变的更好了，如hunter，有些英雄相对变差了) #############################
//    1. GreedyBestMove, warrior, hunter 各1000局数据，不对称提取特征，正负1标签，胜率 hunter vs hunter  63.3%    warrior vs warrior  47.6%
//    double[] coef = {0.107,0.030,-0.368,-0.028,-0.056,0.099,0.008,0.027,0.109,0.031,-0.071,-0.006,0.000,-0.428,0.032,
//                    -0.116,-0.042,0.321,0.004,-0.099,0.009,-0.073,-0.033,-0.134,0.001,-0.174,0.036,0.000,0.256,-0.050};

//    2. GreedyBestMove, warrior + hunter + rogue + druid 各1000局数据，不对称提取特征，正负1标签，胜率：
//      hunter vs hunter  61.9%
//      warrior vs warrior  41.5%
//      Druid vs Druid      39.3%
//      rogue vs rogue      40.4%
//    double[] coef = {0.109,0.052,-0.224,-0.055,-0.145,0.091,0.023,0.008,0.113,0.020,-0.017,-0.011,0.000,-0.384,0.036,
//                    -0.122,-0.065,0.177,0.039,0.010,-0.052,-0.012,-0.020,-0.119,-0.006,-0.068,0.024,0.000,0.336,-0.061};

//    3. 单独使用Druid数据训练，然后测试Druid，胜率 24.4% （出乎意料的低）
//    double[] coef = {0.098,0.086,-0.009,-0.111,-0.279,0.092,0.045,-0.050,0.107,0.024,0.006,-0.007,0.000,-0.336,0.050,
//                    -0.105,-0.080,-0.036,0.087,0.096,-0.111,0.079,0.061,-0.105,-0.019,0.078,0.003,0.000,0.374,-0.070};

//    4. 单独使用Rogue数据训练，然后测试Rogue，胜率 27.4%
//    double[] coef = {0.139,0.090,-0.218,-0.154,-0.318,0.070,0.132,0.043,0.139,-0.018,0.053,-0.026,0.000,-0.406,0.047,
//                    -0.162,-0.126,0.124,-0.904,0.364,-0.174,-0.005,-0.120,-0.086,-0.004,-0.031,0.032,0.000,0.464,-0.067};

//    ################################################################################
//    1. 尝试使用训练出来的模型产生训练数据，循环迭代训练，看看能不能进一步改进模型效果
//    使用randomPlay_1000games数据，不对称提取特征，训练得到的模型（胜率 51.6% ）再生成1000局对局数据GreedyBestMoveLinear_1000games，再训练局面评估函数
//    胜率 10.7%，无法理解为什么这么差。。。
//    double[] coef = {0.147,0.212,-0.161,0.291,-0.093,-0.003,0.024,-0.038,0.130,0.004,-0.195,-0.021,0.000,-0.459,0.021,
//                    -0.151,-0.250,0.129,2.036,-0.210,-0.029,0.134,0.056,-0.119,-0.010,0.061,0.033,0.000,0.364,-0.045};

//    ####################################################################################
//    1. 尝试使用最后的Hp差值来设置标签，进行训练 (看coef，基本权重都在Hp维度上，直观感觉不会太好)
//     randomPlay_1000games数据，不对称提取特征，胜率 2.5%，果然效果很差
//    double[] coef = {0.945,0.007,-0.059,0.003,0.014,0.004,0.006,-0.019,0.013,0.001,0.008,-0.002,-0.000,0.007,-0.002,
//                    -0.947,-0.010,0.057,0.054,-0.012,-0.006,-0.004,0.025,-0.011,-0.008,-0.013,0.002,0.000,-0.008,0.002};

    // CEM直接训练 （结束时的HpDiff是评价指标）
    // 1. Hunter, 随机初始化， 胜率 60%， 简单有效 (这个参数对新构造的warrior的basic卡组效果也很好)
//    double[] coef = {-0.5599020978143998, -0.10678845255815499, -0.5060368963662478, 0.860109197731925, 2.1123828603053014, 0.6748960607386161, -0.32513191654973006, 0.8749736385951374, 4.271168821278315, 0.11435620316270807, -0.36339019448135423, 0.3292022976845038, -0.9752350022895511, -2.613927521735647, -0.6927209978674066, -1.0898216878945886, -2.462299355830384, 1.0329140566745134, 0.3548640655388687, 0.8995171116460702, 0.4771197381045891, 1.1243455865213696, -2.5538467503057314, -1.5563377743285818, -0.5151415484415016, 0.6429385956794124, -0.3643501955276073, -0.3916391257366842, 0.15450016298794692, -0.9919588895968282};
//    2. Warrior, basic deck, 胜率 57.4%
//    double[] coef = {2.3494133199429315, -0.7615906587168452, -2.4384364512470187, -0.9368315587989626, -2.1338391218099586, 0.2766773611662532, 0.1388061389206794, -2.8204320682980537, 1.9821513158072182, 0.9080322682980011, -0.5887226327604637, -1.488445673190608, -3.17454484211928, -6.500031136371627, 1.643412492696721, -1.0524469883439322, -0.4230946623121478, 0.20443704438104543, 1.6944867053690356, -2.7278619412516103, 0.2253514521762577, -1.7596786794500079, 1.8462647587760916, -4.464555188920525, -1.0982849749329433, 2.25501236925314, 3.1800554726099572, 0.4589838393472361, -0.813965389959278, 1.2866990054082552};
      // 使用上面Hunter参数初始化par mean进行训练的结果，胜率 62.2% （似乎并没有进一步提升）
//      double[] coef = {0.3349568456332215, -0.963450274742315, -0.5409124536665365, 0.4044185867589173, 1.384237318140199, 1.2303642480538852, -1.1759118655280163, 0.01123459093219008, 3.2345593729531, -0.3485486421428811, -0.0948463477064634, 0.4376530059270014, -0.6787503556058103, -2.0793790784136084, -0.4055290853366724, -0.9895335157579717, -1.535095372520873, 1.4922323153023829, 0.5025394252359925, 0.08295117802689919, 0.771983199717793, 1.3794411968833755, -3.0728225911681974, -0.2393005015218336, 0.043406049294212955, 0.597141874140535, -0.9073180790435322, -0.36414480492848145, -0.6551910897627793, -1.8767889305772225};

      //  Warrior, basic deck, Supervised 训练， random1000数据, Linear regression, discount 0.99, 胜率 53.9%
//      double[] coef = {0.017,-0.007,-0.054,0.021,-0.026,0.041,0.013,-0.069,0.057,0.010,-0.016,0.026,0.000,-0.077,0.000,-0.020,0.009,0.034,-0.010,0.042,-0.038,-0.017,0.074,-0.048,-0.009,0.008,-0.023,0.000,0.132,0.000};
     //  Warrior, basic deck, Supervised 训练， random1000数据, Logistic regression, 胜率 54.0%
//        double[] coef = {0.055,-0.030,-0.196,0.064,-0.093,0.133,0.046,-0.243,0.186,0.037,-0.021,0.083,0.000,-0.257,0.000,-0.065,0.032,0.138,-0.035,0.172,-0.136,-0.065,0.250,-0.162,-0.032,-0.020,-0.069,0.000,0.390,0.000};
        // 在上面54%参数的基础上进行CEM优化，胜率 50.4%， 反而下降了，说明现在CEM优化的方向有点问题，或者说现在使用的HpDiff这个优化目标可能需要调整
//        double[] coef = {-0.7347704027762162, -0.04189805555739064, 0.6021033270405626, -0.589421898496512, -0.46307469952144026, 0.15716627461074517, -0.1467122665026592, -0.6258889966512419, 1.1744789175133379, 0.891557258653844, -0.43775959801049863, -0.28628912264277706, -0.10397746949617213, -1.1121908492342252, -0.5798910743425613, 0.5880584765901816, 0.892629262873395, -0.9396100331627476, -0.2314059337667665, 0.18772220419562716, 0.3870115843663642, 0.2656611833382197, -0.8843380629976326, -1.6661561213646012, 0.17441567163482913, 0.21851977509006856, -0.17580313785104315, -0.5036564147389638, -0.509786450362303, 0.40041255288843713};

    // ##############################################################################
    // 改用Batch CEM直接优化一个batch的胜率
    // batchSize = 50; updateBatchSize = 20; topRatio = 0.25; 总局数 20000， 随机初始化N(0,1), 胜率 63%, 明显好于直接的CEM （但是这个参数还是很奇怪，为什么自身血量的权重是负数？）
//    double[] coef = {-2.0797226047257866, 0.830166599160172, 1.8125150255847366, -1.3327632490237755, 0.3080800968655299, -1.2463654599824516, 0.7081848405733783, 1.4123922529951056, 1.9429273449684543, 0.35813192746010436, -2.305584194425383, -1.044974502744587, 0.4603944578086973, -0.13885720604885737, 0.6076634157554632, -1.2792109333734534, 0.9990944578665518, 0.5890105559669363, -1.6318519194353063, 0.43820112484230683, 0.1688493669700329, 1.0934715486003266, 1.5895760517222293, -1.237926768085843, -0.5549145589811724, -0.06099650891291013, 0.557033447793615, -0.14392641846009047, 1.6375215167545705, -0.6573524120790898};
    // batchSize = 50; updateBatchSize = 20; topRatio = 0.25; 总局数 20000， 随机初始化N(0,1), 胜率 69.6%
//    double[] coef = {-1.3079282827328942, 0.58732182663696, -2.6324183416580507, 2.2297075162983617, 1.4766381888567601, -1.239992592665568, 0.16009216264780216, 2.4407369377995094, 1.4103622052230778, 0.9392209435529647, 0.6235191988969114, -1.132928168524416, 1.4088975530024825, -1.7068394255346213, -2.0378584172266794, -1.899556325115347, 0.7252582853479771, 0.2763803971411474, -0.3517244983780303, 1.0997846130225812, 0.7067738299797501, 0.7024497552923238, 0.30655644681996275, -2.472900226567996, 2.573351366872348, 0.6606831930154969, 1.7603264032746868, -0.5120515878455165, -0.24779314382277934, 0.1090970298623786};
    // 上面那次运行过程中，某一次batch胜率为42/50对应的参数，1000次测试胜率 70.4% （在这个特定的basic deck上，对GameStateValue的胜率也有54%了） 在随机deck上对GreedyBestMove还是只有30%+的胜率
//    double[] coef = {-1.0908019161141327, 0.6010612694539739, -2.5866502811370067, 2.22006504329619, 1.5848131785872193, -1.5805967137590908, 0.2941482345080195, 2.438458701747288, 1.6271195269493988, 0.7199948277761504, 0.506913009963485, -1.1279103773463786, 2.047988618707141, -1.57363758325508, -2.1533199260998135, -1.5920098329344945, 0.8910700740567333, 0.10653911094993662, -0.31931599483632694, 1.1156702570688315, 0.7425087545093341, 0.7080486811044453, 0.26080540244703476, -2.4334761063404704, 2.4431256407143143, 0.6575185973320499, 1.625164010147523, -0.45913914628255625, -0.23061990409221683, 0.08987186060534504};

    // 以70.4%的参数初始化，尝试使用batch CEM VS 这个参数对应的GreedyBestMoveLinear进行训练 （从结果来看，这种迭代训练的方式很难有提升）
    // 最终parMean， Vs GreedyBestMove 胜率 67.1%
//    double[] coef = {-1.1779447472496432, 0.989913365496526, -1.923300257554408, 2.150270674746591, 0.9874725242742034, -0.9210318595997492, 0.927361556423732, 2.3739523655350614, 1.7467134943676672, 1.1807599474436434, -0.8805590270924755, -0.875556160656964, 1.6515962151700971, -2.7603056181322154, -2.7646060346185863, -1.4118386225454607, 1.9328172295426165, 0.03203490307037743, 0.4189519496422004, 0.9347123387279597, 1.2367993159240882, 0.4808456258348186, -0.8227207743054281, -2.2979983746549584, 2.993189406815853, 0.6497145272444547, 1.6181769220479498, 1.6316882103479533, -1.1415837965071185, 2.0764396970290435};
    // 中间某次34/50的参数， Vs GreedyBestMove 胜率 70.1%
//    double[] coef = {-1.1737539281640201, 0.985564654084688, -1.9263432596811414, 2.1576339069793096, 0.9802833443773435, -0.9824179831887464, 0.8214640783251463, 2.4500829740492738, 1.2162843835358093, 1.155371421799991, -0.8714762565370807, -0.8739559857757228, 1.5821303085465122, -2.744535844897132, -2.7110213393818916, -1.366823761941784, 1.9353451693515558, 0.03205170265478041, 0.42112381331486015, 0.9367335879102747, 1.2858868821868543, 0.44767240347546405, -0.6675802176237489, -2.2934787695893846, 3.0084781471921325, 0.6494528403370364, 1.7407195489604637, 1.6090184330290742, -1.1731108979710196, 1.6941383984349139};

    // 使用Batch CEM训练之前的Hunter random deck， VS Greedy Best Move, 胜率 64.4% （证明了 Batch CEM算法的有效性，一个可能的局限是优化得到的参数只是对训练对手AI比较有效）
//    double[] coef = {0.07867269073028295, -0.05871635279257137, 0.2691386404321768, 0.7640731463926775, -0.09063329574356653, 0.25662717140918917, -0.24581516621266083, -0.26036248881523427, 1.5999024991493918, 0.6559015047862661, 0.026059784191931674, -0.5250603428635277, -0.9359161083555538, -0.2580164980340622, -1.0552901384820939, -0.832945406054766, 0.07667065041847747, 0.6566769736959431, -0.7466449682071832, -0.19558779292082473, -1.405546686916437, 0.027243120005678446, -0.9971595447179644, -1.2120900482148662, -0.19513836562191833, 0.9401871549477598, -0.7727314764509507, -0.2074126610813762, 0.13700488093337782, -1.546665587023834};

    // ##############################################################################
    // ######################## 特征大扩展到88个，增加随从的各种技能特征，如嘲讽、风怒、冻结等等，也增加武器攻击、耐久等特征 ###############################
    // iterNum=16, ParaMean  vs GreedyBestMove (使用 Wild pirate warrior卡组进行训练，针对默认GreedyBestMove，Greedy 模型 ##########################)
    //double[] coef = {-0.3197059078415641, 0.26563979510130475, -2.6596709517335033, 1.9381693797762174, 0.6615979464703083, 0.326881228128546, 0.5013003217711116, -0.1417382458779886, -1.249110064884316, -0.9707419068057191, 1.1420193284160278, 1.705627785179035, 0.11629496534549937, 0.4700783113900231, -1.1433090241292334, -0.04072748377562259, -0.4186661001353569, 0.6896225769927529, -0.4370041703948341, 1.4415685213996325, 0.8059970417270993, -0.7272289162112864, 0.6352391078496592, -0.12919708315277464, 0.08729507226726044, -0.249213119941941, 0.8192411978142655, 1.0288685625802263, -0.932218232240942, -0.17827228887057417, 0.4374783534013372, -0.7096717215021402, -0.15388043088891434, -1.514053836829151, -0.09243295748298855, -1.5912837382075113, -1.0889320783256706, -0.7497296829936259, -1.0270214061299254, -1.9990209494969282, -0.3824219991234162, 1.3045504644698778, 2.2173034209086784, -0.8885718984444004, -0.48764256395524075, -2.2087635807644337, 0.6129313043218516, 2.404797533663557, -0.4964562375427281, 0.1481320197449692, -1.0936918894553047, 0.5718305832578523, -1.111287525319979, -2.836698746652661, -1.2115196792224003, 0.7380790729956219, 0.15615711360515935, -1.251125139816367, 0.864219439601834, -0.3196240104387729, 2.420573609987938, -0.8993221687117616, -0.8059514109255506, -0.7327889243951451, 2.9460776349874633, 1.2355743315583965, 0.27505488183592347, -1.0950708046039126, 0.911800892880857, -0.4288366493029427, 1.3884339331997009, 2.1900266565503914, -0.17654313783475634, -0.27674873680050194, -0.12714653121389805, -1.2662627624095197, -1.0761261964304734, 1.7158581590553794, 0.5591984173601761, -0.9174071137303029, 2.7297059308061455, 0.12566296423398887, 0.5025516966134067, 0.03148595175713782, 0.09772115759829057, -0.33374034291813126, 0.12430774520421281, 1.5777412996091424};

    /* fh start */
    /* 这里的胜率一般是指GreedyBestMove */
    // 具体信息请查看issue
    // 实验1
    //double[] coef = {0.103,-0.044,0.026,-0.015,0.153,0.052,0.073,0.106,0.076,0.053,0.137,0.003,0.000,-0.062,0.010,-0.098,0.035,-0.022,0.022,-0.157,-0.044,-0.074,-0.088,-0.072,-0.060,-0.140,-0.004,0.000,0.051,-0.008};

    //实验2
    //double[] coef = {0.101,-0.024,-0.092,0.019,0.113,0.040,0.065,0.074,0.063,0.046,0.106,0.006,0.000,-0.094,0.018,-0.099,0.019,0.094,-0.062,-0.099,-0.038,-0.068,-0.058,-0.064,-0.047,-0.094,-0.008,0.000,0.101,-0.018};

    //实验3
    //double[] coef = {0.031,-0.014,0.010,-0.005,0.063,0.011,0.021,0.049,0.015,0.018,0.040,0.000,-0.000,-0.029,0.004,-0.029,0.012,-0.008,0.013,-0.063,-0.011,-0.020,-0.045,-0.013,-0.020,-0.041,-0.001,0.000,0.027,-0.003};

    //实验4
    //double[] coef = {0.106,0.783,-0.036,0.003,0.008,0.201,0.017,0.057,0.083,0.018,0.063,0.053,0.069,0.061,-0.045,0.109,0.426,-0.706,0.311,-0.123,0.051,-0.010,0.403,-0.404,-1.156,0.432,-0.456,0.274,0.062,-0.177,0.553,-0.095,-0.037,-0.313,0.126,0.081,0.006,0.000,-0.692,0.138,-0.087,0.036,0.012,-0.090,0.252,0.050,-0.131,0.232,-0.966,-0.080,-0.073,-0.023,0.017,-0.069,-0.056,-0.410,0.078,0.025,0.259,0.205,-0.427,0.320,-0.092,-0.034,-0.249,0.069,-0.112,0.493,-0.214,0.142,2.136,-0.634,-0.130,0.299,-0.012,-0.028,-0.258,0.148,-0.104,-0.009,0.000,0.828,-0.163,-0.067,-0.001,-0.113};

    //实验5
    //double[] coef = {0.101,0.047,-0.026,0.033,-0.061,0.256,0.091,0.044,0.062,0.071,0.061,0.046,-0.049,0.044,-0.020,0.090,-0.026,0.073,-0.101,0.049,-0.011,-0.136,0.090,-0.006,-0.014,0.014,0.026,0.098,0.027,-0.042,-0.146,0.042,-0.000,-0.122,0.017,0.128,0.004,0.000,-0.292,0.031,-0.087,0.020,0.052,-0.103,-0.039,0.025,-0.032,0.052,-0.107,-0.085,-0.048,-0.062,-0.077,-0.059,-0.047,0.108,-0.056,0.023,-0.015,0.040,-0.156,0.022,-0.030,0.035,-0.017,-0.037,-0.010,0.128,-0.049,-0.005,-0.011,-0.036,0.041,-0.005,-0.076,0.082,0.088,-0.017,-0.103,-0.008,0.000,0.267,-0.034,0.094,-0.020,-0.071};

    //实验6
    //double[] coef = {0.101,0.024,-0.040,0.042,-0.071,0.282,0.152,0.043,0.082,0.114,0.075,0.050,-0.064,0.044,-0.024,-0.132,-0.082,0.260,-0.071,0.040,0.004,-0.190,0.092,0.019,-0.004,-0.002,0.048,0.149,0.027,-0.055,-0.165,-0.001,0.047,-0.087,-0.006,0.153,0.001,0.000,-0.285,0.027,-0.070,0.017,0.066,-0.104,-0.053,0.040,-0.036,0.067,-0.184,-0.120,-0.057,-0.074,-0.108,-0.071,-0.059,0.099,-0.060,0.031,0.010,-0.214,0.110,0.138,-0.050,0.010,-0.090,0.002,-0.051,0.099,-0.044,-0.004,-0.052,0.019,0.009,-0.165,-0.078,0.115,0.145,-0.033,-0.135,-0.005,0.000,0.273,-0.033,0.078,-0.015,-0.095};

    //实验7
    //double[] coef = {0.030,0.006,-0.012,0.011,-0.023,0.076,0.060,0.010,0.021,0.047,0.018,0.015,0.001,0.005,-0.007,-0.147,0.006,0.062,-0.030,0.014,0.001,-0.046,0.026,0.002,-0.014,0.007,0.013,0.068,0.005,-0.016,-0.047,-0.001,0.013,-0.032,-0.002,0.045,-0.000,-0.000,-0.105,0.010,-0.031,0.006,0.023,-0.031,-0.008,0.013,-0.011,0.024,-0.041,-0.056,-0.011,-0.022,-0.046,-0.017,-0.017,0.025,-0.016,0.009,0.051,-0.048,-0.004,0.053,-0.022,0.006,-0.028,0.001,-0.012,0.150,-0.038,-0.007,0.008,-0.002,0.001,-0.031,-0.024,0.031,0.051,-0.011,-0.037,-0.001,-0.000,0.104,-0.012,0.035,-0.006,-0.034};

    //实验8
    //double[] coef = {0.103,-0.044,0.026,0.153,0.052,0.073,0.106,0.076,0.053,0.137,0.003,-0.062,0.010,-0.098,0.035,-0.022,-0.157,-0.044,-0.074,-0.088,-0.072,-0.060,-0.140,-0.004,0.051,-0.008};

    //实验9
    //double[] coef = {0.100,-0.050,0.021,0.015,0.108,0.053,0.079,0.086,0.075,0.055,0.136,0.003,-0.074,0.011,-0.099,0.041,-0.018,-0.082,-0.140,-0.047,-0.080,-0.111,-0.073,-0.052,-0.151,-0.002,0.059,-0.011};

    //实验10
    //double[] coef = {0.100,-0.033,0.035,0.070,0.051,-0.103,0.126,0.046,0.085,0.092,0.078,0.052,0.127,0.005,-0.070,0.012,-0.099,0.036,-0.031,0.029,-0.054,0.088,-0.189,-0.032,-0.085,-0.089,-0.075,-0.062,-0.136,-0.004,0.030,-0.003};

    //实验11
    //double[] coef = {0.100,-0.045,0.031,-0.003,0.077,-0.111,0.144,0.056,0.069,0.101,0.076,0.051,0.137,0.002,-0.081,0.010,0.020,-0.008,0.045,-0.099,0.041,-0.030,-0.078,-0.035,0.065,-0.131,-0.039,-0.090,-0.099,-0.075,-0.054,-0.161,0.008,0.041,0.004,-0.010,-0.015,-0.096};

    //实验12
    //double[] coef = {0.102,-0.037,0.024,0.012,0.033,-0.061,0.133,0.050,0.070,0.100,0.068,0.049,0.020,0.051,0.162,0.125,0.004,-0.055,0.002,-0.100,0.037,-0.016,-0.014,-0.066,0.102,-0.174,-0.040,-0.063,-0.122,-0.062,-0.050,0.005,-0.092,-0.210,-0.139,-0.003,0.060,-0.005};

    //以下是尝试启发式算法

    /*
    LinearSA
    double[] coef = {-4.241073713745082, -12.056120425097753, 4.2185345379921255, -9.049399838472766,
            10.477805645979194, -3.589006214920987, -2.277197573376644, -0.7925255431493199,
            -5.708000499250177, 5.024858464544529, 3.656592529453385, 4.79938803413826, -5.183288814707212,
            0.6422598257279797, 2.868247844677586, 3.888589482315878, 5.012405025286813, -8.323822417107433,
            -8.423225836228985, 17.50508993283151, 1.9978079385323113, 1.3431108511950665, 16.05600063138508,
            2.1460495761556952, 4.060469204855834, 9.5602209520546, -7.015596217402916, 3.4974168526887133,
            0.4089335353358605, -14.332540872123298, 2.1241728989149884, 7.844471131041297};
    */

    //LinearBatchSA
    /*
    double[] coef = {3.644135023663773, -0.9615609081122801, 2.024042805418184, -2.839589239851455, -0.36083944016761743,
            3.1327423817680935, -1.6006532244133544, 1.5240364333068965, 0.10292412063785437, 2.20493486624309,
            -0.8002076094203605, -1.4462089461575431, 0.20253547230204938, -3.5594085170230834, 0.8112617837864082,
            0.8455193120957427, -5.317338916640265, -6.578989199200123, 4.895846727447919, 3.760148320983327, -1.0517862238133977,
            1.5308359227121806, 3.036194214080256, 1.5169217468285965, 4.001465499296524, -0.663429102626834, -0.2916637130431557,
            -2.0998714957091718, -5.175849327504254, 2.494277902378437, -1.6480248758549747, 3.0551952941271994};
    */
    /*
    double[] coef = {3.931197209815702, -1.1987419278250058, 1.7813530501731873, -2.6530103893196255,
            0.1045643445905135, 3.3835100814176977, -1.6872816196473608, 0.6180272742028643, -0.08781454711684403,
            2.4885640534543323, -0.9814830260583718, -1.5966585437914833, 0.8093754845122078, -3.1052856774628834,
            1.0289840197579336, 0.943278708307815, -5.292217750225227, -6.057237083483911, 3.9490369305445374,
            3.9480979469017092, -0.6805164659088052, 1.921111685264891, 3.3727837239615757, 1.5834713562755862,
            4.039336188586661, -0.22644941382138234, -0.492971458231964, -1.577418337558726, -6.14427430085533,
            2.5380283988310586, -1.7302123155336135, 3.026029918611831};
    */

    /* WPW Deck */
    // 32 features, simple way
    //double[] coef = {0.115,-0.009,-0.220,0.075,0.098,0.037,0.008,0.123,0.146,-0.019,0.145,0.145,0.422,0.018,0.097,-0.002,-0.117,0.002,0.221,-0.085,-0.103,-0.039,-0.040,-0.118,-0.123,0.023,-0.136,-0.154,-0.428,-0.013,-0.082,-0.003};

    // 30 feature Batch
    double[] coef = {0.7125495137949671, 0.5091471257647046, -0.8231712701706321, 2.5264078113855093, 1.0375569288864837, 1.1640239998359707, 2.5425126505564686, 0.7063599637874528, -0.29630062482053854,
            -0.24730433469129723, 0.7666541346755109, 1.292402294962682, -0.07295217499663155, -1.6413819705409334, -0.12633182222585249, -1.3502586312638063, 1.2183994391481012, 1.3965305729519653, 1.2816231247747116, 0.4688391343487325, 1.4597273721196413, 0.7071357244516756, -0.10755065286791628, 0.015267351890178027, -1.6574929534344691, 0.22991348602785133,
            -0.1871396612178135, -0.3758821541493397, 0.15470755202242886, -0.599345569739484, 0.5149263468550491, -1.4364981485255408, 0.13497871391131866, -0.3281306772562238, -0.4717226509938427, -1.582171489519275, -0.38941887040583034, -1.6958656462691009, -0.03349791215206621, -0.7632922771274688, -0.1769969519740584,
            0.3396109880632417, 2.6328553088999187, -1.386774393843911, -0.7475120698794813, -0.2679980876940002, -1.1957724946391273, 1.3811444534964463, 0.23879211550347912, 1.6930478948170542, 0.9915371795640486, 0.6347449313827516, -1.2922332153901501, -2.6363283905917556, -1.9474001555621225, 1.3182799215367556, -1.509731805873174,
            0.5869939417244485, 0.5534111891123048, -1.7796893946050971, 1.9718815938941388, -0.7755856847194096, -0.14549631439286065, -0.08231247510396461, 0.19227576178672498, -1.1423045385267052, -1.0977362110952111, 0.4477935553039135, 0.3135016119378379, -0.2712803445476216, -1.7965867943882368,
            0.1596496339401956, -1.0193698457189149, 0.9329734279902431, -0.9466294488186886, -1.2670569354925694, -0.016736280523102998, 1.2178189706537492, -0.7794069372738947,
            -0.909663114866837, 1.538847772237225, 0.1936418994057872, -0.0049351466810061695, -1.2522135632076028, 0.38960028015360737, -1.0453184890254854, 0.00469731243404066, 2.1131931013655354};
    /* WPW Deck End */
    /* fh end */

    private final static Logger logger = LoggerFactory.getLogger(GreedyOptimizeMoveLinear.class);

    @Override
    public double getScore(GameContext context, int playerId) {
        float score = 0;
        Player player = context.getPlayer(playerId);
        Player opponent = context.getOpponent(player);
        if (player.getHero().isDestroyed()) {   // 己方被干掉，得分 负无穷
            return Float.NEGATIVE_INFINITY;
        }
        if (opponent.getHero().isDestroyed()) {  // 对方被干掉，得分 正无穷
            return Float.POSITIVE_INFINITY;
        }

        List<Integer> envState = player.getPlayerState();
        //List<Integer> envState = player.getPlayerStatefh0();
        //logger.info("Origin Size: {}", envState.size());
        envState.addAll(opponent.getPlayerState());
        //envState.addAll(opponent.getPlayerStatefh0());
        //logger.info("Total Size: {}", envState.size());
        //logger.info("Coef Size: {}", coef.length);
        // 威胁等级标识特征
        /*
        int threatLevelHigh= 0;
        int threatLevelMiddle = 0;
        int threatLevel = calculateThreatLevel(context, playerId);
        if(threatLevel == 2){
            threatLevelHigh = 1;
        }else if(threatLevel == 1){
            threatLevelMiddle = 1;
        }
        envState.add(threatLevelHigh);
        envState.add(threatLevelMiddle);
        */
        assert(envState.size() == coef.length);
        int j = 0;
        for (int i = 0; i < envState.size(); i++){
            score += coef[i]*envState.get(i);
            j++;
        }

        return score;
    }

    private static int calculateThreatLevel(GameContext context, int playerId) {
        int damageOnBoard = 0;
        Player player = context.getPlayer(playerId);
        Player opponent = context.getOpponent(player);
        for (Minion minion : opponent.getMinions()) {
            damageOnBoard += minion.getAttack(); // * minion.getAttributeValue(Attribute.NUMBER_OF_ATTACKS);
        }
        damageOnBoard += getHeroDamage(opponent.getHero());  //对方随从 + 英雄的攻击力

        int remainingHp = player.getHero().getEffectiveHp() - damageOnBoard;  // 根据减去对方伤害后我方剩余血量来确定威胁等级
        if (remainingHp < 1) {
            return 2;
        } else if (remainingHp < 15) {
            return 1;
        }
        return 0;
    }

    private static int getHeroDamage(Hero hero) {
        int heroDamage = 0;
        if (hero.getHeroClass() == HeroClass.MAGE) {
            heroDamage += 1;
        } else if (hero.getHeroClass() == HeroClass.HUNTER) {
            heroDamage += 2;
        } else if (hero.getHeroClass() == HeroClass.DRUID) {
            heroDamage += 1;
        } else if (hero.getHeroClass() == HeroClass.ROGUE) {
            heroDamage += 1;
        }
        if (hero.getWeapon() != null) {
            heroDamage += hero.getWeapon().getWeaponDamage();
        }
        return heroDamage;
    }


    @Override
    public void onActionSelected(GameContext context, int playerId) {
    }


}
