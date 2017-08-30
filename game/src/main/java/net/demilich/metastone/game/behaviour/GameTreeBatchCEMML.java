package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.Attribute;
import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.ActionType;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.entities.heroes.Hero;
import net.demilich.metastone.game.entities.heroes.HeroClass;
import net.demilich.metastone.game.entities.minions.Minion;
import net.demilich.metastone.game.logic.GameLogic;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

// 尝试直接使用CEM (Noisy Cross Entropy Method) 方法优化Linear value function中的参数, 但现在会往前多步后评估局面，而不是GreedyBest

public class GameTreeBatchCEMML extends Behaviour {

    private final static Logger logger = LoggerFactory.getLogger(GameTreeBatchCEMML.class);
    private Random random = new Random();
    private final static int feaNum = 88;
    private static int depth = 2;
    private static INDArray parMean = Nd4j.zeros(depth, feaNum);
    private static INDArray parVar= Nd4j.valueArrayOf(depth, feaNum, 0.25);
    private static INDArray parWeight = Nd4j.zeros(depth, feaNum);

    private static ArrayList<INDArray> paraList = new ArrayList<>();
    private static SortedMap<Integer, List<Integer>> rewardMap = new TreeMap<>(Comparator.reverseOrder());
    private static int gameCount = 0;
    private static int batchCount = 0;
    private static int batchWinCnt = 0;
    private static int iterNum = 0;
    private final static int batchSize = 50;
    private final static int updateBatchSize = 20;
    private final static double topRatio = 0.25;

    // 初始化par均值
//	double[] coef0 = {-0.4026557850528921, -1.4777779620148956, 0.4807519990112861, 0.17256270698849252, -3.0048219754608025, 1.2471351008867102, 0.0984930660725887, 0.10428128567415333, 2.305958663224414, 1.1694585122714134, 1.6161794705279806, -1.2488627310061495, 0.7849320841015818, -0.44585583054203787, -0.5739923937489321, -0.15309219515362008, 0.10443397417923186, -1.6386810620082035, -1.5720069467333566, 1.142923451103324, -1.8273642125940401, 3.9607177425623874, -1.5590593357541482, -1.8584667661464103, -0.7176115097989136, -1.9578732692028225, -1.555219474560269, 1.260854506957219, -2.110174979376178, 0.6617734109303619};

    public GameTreeBatchCEMML() {
        logger.info("parVar: {}", parVar);
        updateParWeight();
    }

    @Override
    public String getName() {
        return "CEM-Batch-Tree-ML";
    }

    @Override
    public List<Card> mulligan(GameContext context, Player player, List<Card> cards) {
        List<Card> discardedCards = new ArrayList<Card>();
        for (Card card : cards) {
            if (card.getBaseManaCost() >= 4 || card.getCardId()=="minion_patches_the_pirate") {  //耗法值>=4的不要, Patches the Pirate这张牌等他被触发召唤
                discardedCards.add(card);
            }
        }
        return discardedCards;
    }

    @Override
    public GameAction requestAction(GameContext context, Player player, List<GameAction> validActions) {

        if (validActions.size() == 1) {  //只剩一个action一般是 END_TURN
            return validActions.get(0);
        }

        depth = 2;
        // when evaluating battlecry and discover actions, only optimize the immediate value （两种特殊的action）
        if (validActions.get(0).getActionType() == ActionType.BATTLECRY) {
            depth = 0;

        } else if (validActions.get(0).getActionType() == ActionType.DISCOVER) {  // battlecry and discover actions一定会在第一个么？
            return validActions.get(0);
        }

        GameAction bestAction = validActions.get(0);
        double bestScore = Double.NEGATIVE_INFINITY;

        for (GameAction gameAction : validActions) {
            double score = alphaBeta(context, player.getId(), gameAction, depth);  // 对每一个可能action，使用alphaBeta递归计算得分
            if (score > bestScore) {
                bestAction = gameAction;
                bestScore = score;
            }
        }
        return bestAction;
    }

    private double alphaBeta(GameContext context, int playerId, GameAction action, int depth) {
        GameContext simulation = context.clone();  // clone目前环境
        simulation.getLogic().performGameAction(playerId, action);  // 在拷贝环境中执行action
        if (depth == 0 || simulation.getActivePlayerId() != playerId || simulation.gameDecided()) {  // depth层递归结束、发生玩家切换（我方这轮打完了）或者比赛结果已定时，返回score
            return evaluateContext(simulation, playerId);
        }

        List<GameAction> validActions = simulation.getValidActions();  //执行完一个action之后，获取接下来可以执行的action

        double score = Float.NEGATIVE_INFINITY;

        for (GameAction gameAction : validActions) {
            score = Math.max(score, alphaBeta(simulation, playerId, gameAction, depth - 1));  // 递归调用alphaBeta，取评分较大的
            if (score >= 100000) {
                break;
            }
        }
        return score;
    }

    private static int calculateThreatLevel(GameContext context, int playerId) {
        int damageOnBoard = 0;
        Player player = context.getPlayer(playerId);
        Player opponent = context.getOpponent(player);
        for (Minion minion : opponent.getMinions()) {
            damageOnBoard += minion.getAttack();
        }
        damageOnBoard += getHeroDamage(opponent.getHero());  //对方随从 + 英雄的攻击力 (暂时没有考虑风怒、冻结等的影响，因为 之前 minion.getAttributeValue(Attribute.NUMBER_OF_ATTACKS)经常得到0)

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

    private double evaluateContext(GameContext context, int playerId) {
        Player player = context.getPlayer(playerId);
        Player opponent = context.getOpponent(player);
        if (player.getHero().isDestroyed()) {   // 己方被干掉，得分 负无穷
            return Float.NEGATIVE_INFINITY;  // 正负无穷会影响envState的解析，如果要加的话可以改成 +-100之类的
        }
        if (opponent.getHero().isDestroyed()) {  // 对方被干掉，得分 正无穷
            return Float.POSITIVE_INFINITY;
        }
        List<Integer> envState = player.getPlayerState();
        envState.addAll(opponent.getPlayerState());

        // 威胁等级标识特征
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

        double score = 0;
//        for (int i = 0; i < parWeight.length; i++){
//            score += parWeight[i]*envState.get(i);
//        }
        return score;
    }

    private void updateParWeight(){
        // 根据参数的均值和方差，按正态分布生成parWeight
//        for(int i=0; i<parWeight.length; i++){
//            parWeight[i] = parMean[i] + Math.sqrt(parVar[i])*random.nextGaussian();
//        }
    }

    private double calcMean(double[] paras){
        double mean = 0;
        for(double para : paras){
            mean += para;
        }
        mean /= paras.length;
        return mean;
    }

    private double calcVar(double[] paras){
        double mean = calcMean(paras);
        double var = 0;
        for(double para : paras){
            var += (para - mean)*(para - mean);
        }
        var /= paras.length;
        return var;
    }

    private void updateMeanVar(){
        int k = 0;
        int topNum = (int)(updateBatchSize*topRatio);
        double[][] topPara = new double[feaNum][topNum];
        double[] bestPara = new double[feaNum];
        double meanTopReward = 0;
        // 取出reward最好的若干次的参数
        for(Integer reward: rewardMap.keySet()){
            for(Integer ind: rewardMap.get(reward)){
//                double[] para = paraList.get(ind);
//                for(int i=0; i<para.length; i++){
//                    topPara[i][k] = para[i];
//                }
//                k++;
//                if(k == 1){
//                    bestPara = para.clone();
//                }
                meanTopReward += reward;
                if(k >= topNum){
                    meanTopReward /= topNum;
                    logger.info("################# iterNum: {}, meanTopReward: {}, bestPara: {} ##################", iterNum, meanTopReward, bestPara);
                    // 更新均值和方差
                    for(int i=0; i<feaNum; i++){
//                        this.parMean[i] = calcMean(topPara[i]);
//                        this.parVar[i] = calcVar(topPara[i]) + Math.max(0.1 - 0.01*iterNum, 0);  // 添加逐渐减小的Noise， 可调整
                    }
                    logger.info("########## rewardMap: {}, parMean: {}, parVar: {}", rewardMap, parMean, parVar);
                    // 清空这一个batch的数据
                    paraList.clear();
                    rewardMap.clear();
                    return;
                }
            }
        }
    }

    @Override
    public void onGameOver(GameContext context, int playerId, int winningPlayerId) {
        // GameOver的时候会跳入这个函数
        gameCount++;
        if(playerId == winningPlayerId){
            batchWinCnt += 1;
        }

        // 一个Batch结束
        if(gameCount == batchSize){
            logger.info("batchCount: {}, batchWinCnt: {}", batchCount, batchWinCnt);
            int reward = batchWinCnt;
            // 保存这一Batch的最终batchWinCnt和对应batchCount编号（从0开始编号）
            if(rewardMap.containsKey(reward)){
                rewardMap.get(reward).add(batchCount);
            }else{
                rewardMap.put(reward, new ArrayList<>(Arrays.asList(batchCount)));
            }
            // 保存这一Batch使用的模型参数
//            paraList.add(parWeight.clone());
            batchCount++;
            gameCount = 0;
            batchWinCnt = 0;
            // 根据均值和方差，随机生成下一Batch使用的权重参数
            updateParWeight();
        }

        // 执行一个updateBatchSize之后, 更新参数均值和方差
        if(batchCount == updateBatchSize){
            iterNum++;
//			logger.info("rewardMap: {}, para: {}, parMean: {}, parVar: {}", rewardMap, parWeight, parMean, parVar);
            // 更新参数均值和方差, 并清空这一个batch的数据
            updateMeanVar();
            batchCount = 0;
//			logger.info("########## rewardMap: {}, para: {}, parMean: {}, parVar: {}", rewardMap, parWeight, parMean, parVar);
        }
    }
}
