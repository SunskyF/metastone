package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.ActionType;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Randomk extends Behaviour {
    private final static Logger logger = LoggerFactory.getLogger(net.demilich.metastone.game.behaviour.Randomk.class);

    private Random random = new Random();
    private int k;
    double[] coef = {-1.3079282827328942, 0.58732182663696, -2.6324183416580507, 2.2297075162983617, 1.4766381888567601, -1.239992592665568, 0.16009216264780216, 2.4407369377995094, 1.4103622052230778, 0.9392209435529647, 0.6235191988969114, -1.132928168524416, 1.4088975530024825, -1.7068394255346213, -2.0378584172266794, -1.899556325115347, 0.7252582853479771, 0.2763803971411474, -0.3517244983780303, 1.0997846130225812, 0.7067738299797501, 0.7024497552923238, 0.30655644681996275, -2.472900226567996, 2.573351366872348, 0.6606831930154969, 1.7603264032746868, -0.5120515878455165, -0.24779314382277934, 0.1090970298623786};

    public Randomk(int k){
        this.k = k;
    }

    @Override
    public String getName() {
        return "Random k";
    }

    @Override
    public List<Card> mulligan(GameContext context, Player player, List<Card> cards) {
        return new ArrayList<>();
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
        List<Integer> envState = player.getPlayerStateBasic();
        envState.addAll(opponent.getPlayerStateBasic());
        double score = 0;
        for (int i = 0; i < coef.length; i++){
            score += coef[i]*envState.get(i);
        }
        return score;
    }

    @Override
    public GameAction requestAction(GameContext context, Player player, List<GameAction> validActions) {
        if (validActions.size() == 1) {
            return validActions.get(0);
        }

        if (validActions.get(0).getActionType() == ActionType.BATTLECRY) {
            return validActions.get(random.nextInt(validActions.size()));
        }
        if (validActions.get(0).getActionType() == ActionType.DISCOVER) {
            return validActions.get(random.nextInt(validActions.size()));
        }

        GameAction bestAction = validActions.get(0);
        double bestScore = Float.NEGATIVE_INFINITY;
        int highBound = Math.min(this.k, validActions.size());
        for (int i = 0;i < highBound; ++i){
            int choice = random.nextInt(validActions.size());
            GameContext simulationResult = simulateAction(context.clone(), player.getId(), validActions.get(choice));  //假设执行gameAction，得到之后的game context
            double gameStateScore = evaluateContext(simulationResult, player.getId());

            if (gameStateScore > bestScore){
                bestScore = gameStateScore;
                bestAction = validActions.get(choice);
            }
            validActions.remove(choice);
            simulationResult.dispose();
        }
        return bestAction;
    }

    private GameContext simulateAction(GameContext simulation, int playerId, GameAction action) {
        simulation.getLogic().performGameAction(playerId, action);   // 在simulation GameContext中执行action，似乎是获取logic模块来执行action的
        return simulation;
    }

}
