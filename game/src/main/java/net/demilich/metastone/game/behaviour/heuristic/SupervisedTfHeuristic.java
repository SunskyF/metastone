package net.demilich.metastone.game.behaviour.heuristic;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.behaviour.GreedyOptimizeMoveLinear;
import net.demilich.metastone.game.behaviour.tf_util.Model_loader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.List;

public class SupervisedTfHeuristic implements IGameStateHeuristic{

    private final static Logger logger = LoggerFactory.getLogger(GreedyOptimizeMoveLinear.class);
    private static Model_loader tf_model = new Model_loader("E:\\workspace\\metastone\\game\\model", "frozen_model_0.pb");

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

        List<Integer> envState = player.getPlayerStateBasic();
        envState.addAll(opponent.getPlayerStateBasic());//30 features

        float matrix[][] = new float[1][26];

        for (int i=0, j=0; i<envState.size(); i++){
            if (isIgnore(i))
                continue;
            matrix[0][j] = envState.get(i);
            j++;
        }

        try (Tensor input = Tensor.create(matrix); // 将数组转化为tensor，使其可以作为输入
             Tensor result = this.tf_model.s.runner().feed("INPUT/input", input).fetch("OUTPUT/output").run().get(0)) { //将其输入前向网络，得到输出结果

            score = result.copyTo(new float[1][1])[0][0];
            
        }
        return score;
    }

    private boolean isIgnore(int i) {
        int[] ignoreIdx = {3, 12, 18, 27};
        for (int idx: ignoreIdx){
            if (idx == i)
                return true;
        }
        return false;
    }

    @Override
    public void onActionSelected(GameContext context, int playerId) {

    }
}
