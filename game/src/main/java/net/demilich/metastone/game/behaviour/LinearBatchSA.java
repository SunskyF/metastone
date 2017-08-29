package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.logic.GameLogic;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

public class LinearBatchSA extends Behaviour{

    private final static Logger logger = LoggerFactory.getLogger(LinearBatchCEM.class);
    private Random random = new Random();
    private final static int feaNum = 32;

    private double[] parWeight = new double[feaNum];
    private double[] bestParWeight = new double[feaNum];
    private double bestWin = 0;

    private double[] preParWeight = new double[feaNum];
    private static ArrayList<Float> rewardRecord = new ArrayList<>();
    private static int gameCount = 0;
    private static int batchCount = 0;
    private static int batchWinCnt = 0;
    private final static int batchSize = 50;
    private final static int updateBatchSize = 20;
    private static int preReward = -1;

    private static double T = 10;// 初始化温度
    private static final double delta = 0.98;// 温度的下降率


    private double[] coef = {-0.02720876,  1.36188549,  0.11338756,  0.65414721,  1.58445377,
                    0.37074123, -1.25416617, -0.39662372,  1.1987753 , -0.15725062,
                    -2.75032035, -0.00563931,  0.31177766, -0.21569988,  1.60154003,
                    -0.88498952, -0.10140752,  0.01068558,  0.80651281,  1.48053426,
                    -1.34961829,  1.25708946,  0.35363696, -1.14820727,  0.222536  ,
                    0.38519943, -1.06496087,  0.54320185, -0.09864067,  0.94326665,
                    0.45094126, -0.83054591}; // 32, N(0, 1), np.random.normal(0, 1, 32)

    /*
    double[] coef = {3.644135023663773, -0.9615609081122801, 2.024042805418184, -2.839589239851455, -0.36083944016761743,
            3.1327423817680935, -1.6006532244133544, 1.5240364333068965, 0.10292412063785437, 2.20493486624309,
            -0.8002076094203605, -1.4462089461575431, 0.20253547230204938, -3.5594085170230834, 0.8112617837864082,
            0.8455193120957427, -5.317338916640265, -6.578989199200123, 4.895846727447919, 3.760148320983327, -1.0517862238133977,
            1.5308359227121806, 3.036194214080256, 1.5169217468285965, 4.001465499296524, -0.663429102626834, -0.2916637130431557,
            -2.0998714957091718, -5.175849327504254, 2.494277902378437, -1.6480248758549747, 3.0551952941271994};
    */
    public LinearBatchSA(){
        for(int i=0; i < feaNum; i++){
            parWeight[i] = coef[i] + 0.01 * random.nextGaussian();
            preParWeight[i] = coef[i];
        }
        bestParWeight = parWeight.clone();
    }

    @Override
    public String getName() {
        return "SSA-Linear-Batch";
    }

    @Override
    public List<Card> mulligan(GameContext context, Player player, List<Card> cards) {
        List<Card> discardedCards = new ArrayList<Card>();
        for (Card card : cards) {
            if (card.getBaseManaCost() >= 4) {  //耗法值>=4的不要
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

        // get best action at the current state and the corresponding Q-score
        GameAction bestAction = validActions.get(0);
        double bestScore = Double.NEGATIVE_INFINITY;
        for (GameAction gameAction : validActions) {
            GameContext simulationResult = simulateAction(context.clone(), player, gameAction);  //假设执行gameAction，得到之后的game context
            double gameStateScore = evaluateContext(simulationResult, player.getId());  //heuristic.getScore(simulationResult, player.getId());	     //heuristic评估执行gameAction之后的游戏局面的分数
            if (gameStateScore > bestScore) {		// 记录得分最高的action
                bestScore = gameStateScore;
                bestAction = gameAction;
            }
            simulationResult.dispose();  //GameContext环境每次仿真完销毁
        }

        return bestAction;
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
        List<Integer> envState = player.getPlayerStatefh0(false);
        envState.addAll(opponent.getPlayerStatefh0(false));

        double score = 0;
        assert (envState.size() == parWeight.length);

        for (int i = 0; i < parWeight.length; i++){
            score += parWeight[i]*envState.get(i);
        }
        return score;
    }

    private GameContext simulateAction(GameContext simulation, Player player, GameAction action) {
        simulation.getLogic().performGameAction(player.getId(), action);   // 在simulation GameContext中执行action，似乎是获取logic模块来执行action的
        return simulation;
    }

    private void RandomupdateParWeight(){
        // 根据参数的均值和方差，按正态分布生成parWeight
        for(int i=0; i<parWeight.length; i++){
            preParWeight[i] = parWeight[i];
            parWeight[i] = parWeight[i] + T * random.nextGaussian() / 100;
        }
    }

    private void RecoverParWeight(){
        for(int i=0; i<parWeight.length; i++){
            parWeight[i] = preParWeight[i];
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
            logger.info("batchCount: {}, batchWinCnt: {}, batchWinRate: {}", batchCount, batchWinCnt, batchWinCnt*1.0/batchSize);

            if (batchWinCnt > bestWin){
                //bestParWeight = parWeight.clone();
                bestWin = batchWinCnt;
            }

            int reward = 2 * batchWinCnt - batchSize;
            int deltaReward = preReward - reward;
            logger.info("reward: {}, preReward: {}, Delta Reward: {}", reward, preReward, deltaReward);

            if(deltaReward < 0 ){
                RandomupdateParWeight();
                preReward = reward;
            }else {
                double p = 1 / (1 + Math
                        .exp(-deltaReward / T));
                logger.info("Prob: {}", p);
                if (Math.random() < p) {
                    RandomupdateParWeight();
                    preReward = reward;
                }else{
                    RecoverParWeight();
                    RandomupdateParWeight();
                }
            }
            T *= delta;
            T = Math.max(T, 0.01);

            batchCount++;
            gameCount = 0;
            batchWinCnt = 0;
        }

        // 执行一个updateBatchSize之后, 更新参数均值和方差
        if(batchCount == updateBatchSize){
            batchCount = 0;
            logger.info("Batch count: {}", batchCount);
            logger.info("Para: {}", parWeight);
            logger.info("Prepara: {}", preParWeight);
            logger.info("Temperature: {}", T);
            logger.info("Best Win: {}", bestWin);
            logger.info("Best Para: {}", bestParWeight);
        }
    }
}
