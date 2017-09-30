package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.ActionType;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.behaviour.pso.Agent;
import net.demilich.metastone.game.behaviour.pso.PSO;
import net.demilich.metastone.game.cards.Card;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class GameTreeBatchPSO extends Behaviour {
    private final static Logger logger = LoggerFactory.getLogger(GameTreeBatchPSO.class);

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


    public GameTreeBatchPSO(){
    }

    @Override
    public String getName() {
        return "GameTree-PSO";
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
        if (validActions.size() == 1) {  //只剩一个action一般是 END_TURN
            return validActions.get(0);
        }

        int depth = 2;
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
