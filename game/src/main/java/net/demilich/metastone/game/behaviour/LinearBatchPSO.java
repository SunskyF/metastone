package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.behaviour.pso.Agent;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.behaviour.pso.PSO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class LinearBatchPSO extends Behaviour {
    private final static Logger logger = LoggerFactory.getLogger(LinearBatchPSO.class);

    PSO pso = new PSO();
    // Game Para
    int nFeature = 89;
    // Game Para End

    private static double[] kidRewards = new double[Agent.iPOSNum];


    // Simulation count
    private static int gameCount = 0;
    private static int batchCount = 0;
    private static int batchWinCnt = 0;
    private final static int batchSize = 50;
    private static int epoch = 0;
    // Sim End


    public LinearBatchPSO(){
    }

    @Override
    public String getName() {
        return "Linear-PSO";
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
        List<Integer> envState = player.getPlayerState();
        envState.addAll(opponent.getPlayerState());
        envState.add(context.getTurn());

        double score = 0;
        for (int i=0;i<envState.size();++i){
            score += envState.get(i) * this.pso.agent[batchCount].pos[i];
        }
        return score;
    }

    @Override
    public GameAction requestAction(GameContext context, Player player, List<GameAction> validActions) {
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

    @Override
    public void onGameOver(GameContext context, int playerId, int winningPlayerId) {
        gameCount++;
        if (playerId == winningPlayerId) {
            batchWinCnt += 1;
        }

        if (gameCount == batchSize) { // 第一层，计算一个batch的胜率作为Reward
            kidRewards[batchCount] = batchWinCnt;
            logger.info("BatchCount: {}, BatchWinCount: {}", batchCount, batchWinCnt);
            this.pso.agent[batchCount].UpdateFitness(batchWinCnt);
            batchCount++;
            gameCount = 0;
            batchWinCnt = 0;
        }

        if (batchCount == Agent.iPOSNum) { // 第二层，train one step
            // Agent.iPOSNum: 20
            logger.info("Epoch: {}, reward: {}", epoch, kidRewards);
            double rewardSum = 0;
            for (int i =0;i<kidRewards.length;++i){
                rewardSum += kidRewards[i];
            }
            logger.info("Mean Reward: {}", rewardSum / kidRewards.length);
            logger.info("Best Para: {}", Agent.gbest);
            pso.update();
            batchCount = 0;
            epoch++;
        }
    }
}
