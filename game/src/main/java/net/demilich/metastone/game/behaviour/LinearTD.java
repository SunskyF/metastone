package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.logic.GameLogic;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

// Reference cs221 game2
// And https://github.com/stober/td/blob/master/src/__init__.py

public class LinearTD extends Behaviour { // TD(1)
    private final static Logger logger = LoggerFactory.getLogger(LinearTD.class);

    private Random random = new Random();
    private int feaNum = 30; // start from origin 88 features
    public INDArray parWeight;

    private INDArray e = Nd4j.zeros(feaNum);

    // Hyper Para
    private double lr = 0.01; // learing rate
    private double gamma = 0.9; // discount
    private double ld = 0.5; // lambda

    public LinearTD(){
        parWeight = Nd4j.rand(new int[]{1, feaNum}); // vector as row
//        logger.info("Weight: {}", parWeight);
    }

    @Override
    public String getName() {
        return "Linear-TD";
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

    private INDArray getFeature(GameContext context, int playerId){
        Player player = context.getPlayer(playerId);
        Player opponent = context.getOpponent(player);

        List<Integer> envState = player.getPlayerStateBasic();
        envState.addAll(opponent.getPlayerStateBasic());

//        logger.info("EnvState: {}", envState.size()); // 30

        INDArray feature = Nd4j.zeros(envState.size(), 1); // vector as col
        for (int i=0; i < envState.size(); ++i){
            feature.putScalar(i, 0, envState.get(i).intValue());
        }
        return feature;
    }

    private double value(GameContext context, int playerId){

        double score;
//        logger.info("parWeight: {} {}", parWeight.columns(), parWeight.rows()); // 30 1
        INDArray feature = getFeature(context, playerId);

//        logger.info("Feature Size: {} {}", feature.columns(), feature.rows()); // 1 30
        score = parWeight.mmul(feature).getDouble(0,0);
//        logger.info("Score Mul: {}", parWeight.mmul(feature).shape());
        return score;
    }

    private double delta(GameContext feature, double reward, GameContext pfeature, int playerId){
        return reward + (gamma * value(feature, playerId)) - value(pfeature, playerId);
    }

    private GameContext simulateAction(GameContext simulation, Player player, GameAction action) {
        simulation.getLogic().performGameAction(player.getId(), action);
        return simulation;
    }

    @Override
    public GameAction requestAction(GameContext context, Player player, List<GameAction> validActions) {
        if (validActions.size() == 1) {  //只剩一个action一般是 END_TURN
            return validActions.get(0);
        }
        GameAction bestAction = validActions.get(0);
        double bestScore = Double.NEGATIVE_INFINITY;

        for (GameAction gameAction : validActions) {
            GameContext simulationResult = simulateAction(context.clone(), player, gameAction);  //假设执行gameAction，得到之后的game context

            double gameStateScore = value(simulationResult, player.getId());	     //heuristic评估执行gameAction之后的游戏局面的分数

            //logger.info("Score: {}", gameStateScore);
            logger.info("Action {} gains score of {}", gameAction, gameStateScore);
            if (gameStateScore > bestScore) {		// 记录得分最高的action
                bestScore = gameStateScore;
                bestAction = gameAction;
            }
            simulationResult.dispose();  //GameContext环境每次仿真完销毁
        }
        GameContext simulationResult = simulateAction(context.clone(), player, bestAction);  //假设执行gameAction，得到之后的game context
        double reward = 0;
        if (simulationResult.gameDecided()){
            if (simulationResult.getWinningPlayerId() == player.getId())
                reward = 1;
            else
                reward = -1;
        }
        logger.info("Reward: {}", reward);
        double delta = delta(simulationResult, reward, context, player.getId());
        e = e.mul(gamma).mul(ld).add(getFeature(context, player.getId()).transpose());
        parWeight = parWeight.add(e.mul(lr).mul(delta));
        return bestAction;
    }

}
