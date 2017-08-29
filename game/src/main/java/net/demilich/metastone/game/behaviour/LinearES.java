package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.logic.GameLogic;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

public class LinearES extends Behaviour {

    private final static Logger logger = LoggerFactory.getLogger(LinearCEM.class);
    private Random random = new Random();

    private final static int feaNum = 32;
    private static double[] parWeight = new double[feaNum];
    private static double[] randomWeight = new double[feaNum];
    private static ArrayList<double[]> paraList = new ArrayList<>();
    private static ArrayList<Float> rewardList = new ArrayList<>();

    private final static int N_KID = 10;                 // half of the training population
    private final int N_GENERATION = 5000;         // training step
    private final static double LR = 0.05;                    // learning rate
    private final static double momentum = 0.9;
    private final double SIGMA = 0.05;                 // mutation strength or step size

    private static int gameCount = 0;
    private final int batchSize = 2 * N_KID;
    private static int batchCount = 0;
    private static int batchWinCount = 0;
    private static int bestBatchWinCnt = 0;
    private static double[] bestPara = new double[feaNum];

    private static SGD optimizer = new SGD(feaNum, LR, momentum);

    private double[] coef = {3.966110419559154, -2.4169705502104564, 8.747155488295219, 5.823024410074601, -3.4503442021459647, -8.587312082950975, -0.11506715680773065, 0.0792109105317904, -3.8571690123302087, 0.02598850156605708, -1.544141907557514,
            2.7989451868164696, 2.1288287762202995, -4.386831292592509, 5.4812869663731245, 1.8220898841735311, -2.728902203672761, -5.575826746582367, 1.7498977161809357,
            5.422017794523841, -6.578209778944436, -5.31992256597322, 3.081707581032312, -6.209864088735404, 1.6256958912945998, -7.134837503154441, 3.8747361360708683, -2.306484497059981,
            2.124463004503334, -7.341767865516911, -1.6965733781488535, -5.892826746972325}; // 32, N(0, 1), np.random.normal(0, 1, 32)

    double[] util_ = new double[N_KID * 2];
    private static double[] utility = new double[N_KID * 2];

    public LinearES(){
        // parWeight = coef.clone();
        assert(parWeight.length == coef.length);
        for(int i=0; i<feaNum; i++){
            parWeight[i] = coef[i];
        }
        updateParWeight();
        double base = N_KID * 2;
        double sum = 0;
        for(int i = 0; i < base;++i){
            util_[i] = Math.max(0, Math.log(base / 2 + 1) - Math.log(i + 1));
            sum += util_[i];
        }
        for (int i = 0; i < base; ++i){
            utility[i] = util_[i] / sum - 1.0 / base;
        }
        logger.info("Utility: {}", utility);
    }

    @Override
    public String getName() {
        return "Linear-ES";
    }

    private double Sign(int id){
        return id % 2 == 0 ? -1.0 : 1.0;
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

    private GameContext simulateAction(GameContext simulation, Player player, GameAction action) {
        simulation.getLogic().performGameAction(player.getId(), action);   // 在simulation GameContext中执行action，似乎是获取logic模块来执行action的
        return simulation;
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

        assert (envState.size() == feaNum);
        double score = 0;
        for (int i = 0; i < parWeight.length; i++){
//            score +=  (parWeight[i] + SIGMA * randomWeight[i]) * envState.get(i);
            score += parWeight[i] * envState.get(i);
        }
        return score;
    }

    private void updateParWeight(){
        for(int i=0; i<parWeight.length; i++){
            randomWeight[i] = random.nextGaussian();
            parWeight[i] += Sign(gameCount) * SIGMA * randomWeight[i];
        }
    }

    public static <T extends Number> int[] asArray(final T... a) {
        int[] b = new int[a.length];
        for (int i = 0; i < b.length; i++) {
            b[i] = a[i].intValue();
        }
        return b;
    }

    public static int[] argsort(final Float[] a, final boolean ascending) {
        Integer[] indexes = new Integer[a.length];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return (ascending ? 1 : -1) * Float.compare(a[i1], a[i2]);
            }
        });
        return asArray(indexes);
    }

    @Override
    public void onGameOver(GameContext context, int playerId, int winningPlayerId) {
        // GameOver的时候会跳入这个函数
        Player player = context.getPlayer(playerId);
        Player opponent = context.getOpponent(player);
        float reward = player.getHero().getHp() - opponent.getHero().getHp();
//        float reward = playerId == winningPlayerId ? 1 : 0;
        rewardList.add(reward);
        // 保存这一局使用的模型参数
        paraList.add(randomWeight.clone());
        gameCount++;
        if (reward > 0){
            batchWinCount++;
        }

//        logger.info("Reward: {}", reward);
        // 执行一个batchSize之后
        if(gameCount%batchSize == 0){
            logger.info("BatchCount: {}, BatchWinCount: {}, BatchSize: {}", batchCount, batchWinCount, batchSize);
            logger.info("Best Win Cnt: {}, best Para: {}", bestBatchWinCnt, bestPara);
            if (batchWinCount > bestBatchWinCnt){
                bestBatchWinCnt = batchWinCount;
                bestPara = parWeight.clone();
            }
            batchCount++;
            batchWinCount = 0;
            gameCount = 0;

//            logger.info("Reward: {}, Para: {}", rewardList.size(), paraList.size());
            double[] cumulative_update = new double[feaNum];
            Float[] rewardArray = new Float[rewardList.size()];
            int[] kids_rank = argsort(rewardList.toArray(rewardArray), true);
//            logger.info("Kids Rank: {}", kids_rank);
            for (int i = 0; i < rewardList.size(); ++i){ // number n
                for(int j = 0; j < feaNum; ++j){
                    cumulative_update[j] += utility[i] * Sign(kids_rank[i]) * paraList.get(i)[j];
                }
            }
            for(int j = 0; j < feaNum; ++j){
                cumulative_update[j] /= 2 * N_KID * SIGMA;
            }
            double[] gradients = optimizer.get_gradient(cumulative_update);
            for(int j = 0; j < feaNum; ++j){
                parWeight[j] += gradients[j];
            }
//            logger.info("Para: {}", parWeight);
            paraList.clear();
            rewardList.clear();
        }
        updateParWeight();
    }
}
