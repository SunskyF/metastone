package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.logic.GameLogic;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

public class LinearSA extends Behaviour{

    private final static Logger logger = LoggerFactory.getLogger(LinearBatchCEM.class);
    private Random random = new Random();
    private final static int feaNum = 32;
    //private static double[] parMean = new double[feaNum];
    //private static double[] parVar = new double[feaNum];
    private double[] parWeight = new double[feaNum];
    private double[] preParWeight = new double[feaNum];
    private static int gameCount = 0;
    private static int batchCount = 0;
    private static int batchWinCount = 0;
    private final static int batchSize = 50;

    private static double T = 10000;// 初始化温度
    public static final double Tmin = 1e-8;// 温度的下界
    private static final double delta = 0.99;// 温度的下降率
    private static int preReward = -1;

    double[] coef = {-0.02720876,  1.36188549,  0.11338756,  0.65414721,  1.58445377,
            0.37074123, -1.25416617, -0.39662372,  1.1987753 , -0.15725062,
            -2.75032035, -0.00563931,  0.31177766, -0.21569988,  1.60154003,
            -0.88498952, -0.10140752,  0.01068558,  0.80651281,  1.48053426,
            -1.34961829,  1.25708946,  0.35363696, -1.14820727,  0.222536  ,
            0.38519943, -1.06496087,  0.54320185, -0.09864067,  0.94326665,
            0.45094126, -0.83054591}; // 32, N(0, 1), np.random.normal(0, 1, 32)

    public LinearSA(){
        for(int i=0; i < feaNum; i++){
            parWeight[i] = coef[i] + 0.01 * random.nextGaussian();
            preParWeight[i] = coef[i];
        }
    }

    @Override
    public String getName() {
        return "SSA-Linear";
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
        List<Integer> envState = player.getPlayerStatefh0();
        envState.addAll(opponent.getPlayerStatefh0());

        double score = 0;
        assert (envState.size() == parWeight.length);
        //logger.info("EnvState Size: {}", envState.size());

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
        //logger.info("Para: {}", parWeight);

        for(int i=0; i<parWeight.length; i++){
            preParWeight[i] = parWeight[i];
            parWeight[i] = parWeight[i] + T * random.nextGaussian() / 10000.0;
            //logger.info("Random Gaussian {}", random.nextGaussian());
        }
    }

    private void RecoverParWeight(){
        parWeight = preParWeight.clone();
    }

    @Override
    public void onGameOver(GameContext context, int playerId, int winningPlayerId) {
        // GameOver的时候会跳入这个函数
        Player player = context.getPlayer(playerId);
        Player opponent = context.getOpponent(player);
        int reward = player.getHero().getHp() - opponent.getHero().getHp();

        if(playerId == winningPlayerId){
            batchWinCount += 1;
        }

        gameCount++;
        logger.info("T: {}", T);
        logger.info("gameCount: {}, winner: {}, HpDiff: {}", gameCount, winningPlayerId, reward);  // 可以根据HpDiff来判断胜负和设定reward

        // 执行一个batchSize之后
        int deltaReward = preReward - reward;
        logger.info("Delta Reward: {}", deltaReward);

        if (gameCount % batchSize == 0){

            logger.info("Batch count: {}", batchCount);
            logger.info("Para: {}", parWeight);
            logger.info("Prepara: {}", preParWeight);
            logger.info("Win prob: {}", batchWinCount * 1.0 / batchSize);
            batchCount++;
            batchWinCount = 0;
            gameCount = 0;
        }

        if(deltaReward < 0 ){
            RandomupdateParWeight();
            preReward = reward;
            //gameCount = 0;
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
    }
}
