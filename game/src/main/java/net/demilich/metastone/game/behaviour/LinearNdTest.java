package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LinearNdTest extends Behaviour{
    private final static Logger logger = LoggerFactory.getLogger(LinearNdTest.class);

    INDArray para; // parameters
    List<INDArray> p = new ArrayList<>();
    List<int[]> shapes = new ArrayList<>();

    // Game Para
    int nFeature = 28;
    // Game Para End

    String paraFile = "NdModel/network/NdPara_CEM_28fea.data";

    public LinearNdTest(){
        this.para = buildNetwork();
        try{
            logger.info("Loading... {}", paraFile);
            File readFile = new File(paraFile);
            this.para = Nd4j.readBinary(readFile);
//            logger.info("Para: {}", this.para.get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
            logger.info("Loaded...");
        }
        catch (IOException e){
            e.printStackTrace();
        }
//        logger.info("Shape: {}", this.para.shape());
        this.p = paramReshape(this.para);
    }

    @Override
    public String getName() {
        return "Linear-Nd-Test";
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
        List<Double> envState = player.getPlayerStatefh0(false);
        envState.addAll(opponent.getPlayerStatefh0(true));
//        List<Integer> envState = player.getPlayerState();
//        envState.addAll(opponent.getPlayerState());

        double[] tmp = new double[this.nFeature];
        for (int i=0;i<envState.size();++i){
            tmp[i] = envState.get(i);
        }
        INDArray featureIND = Nd4j.create(tmp);
        return getValue(this.p, featureIND);
    }

    INDArray linear(int nIn, int nOut){
        INDArray w = Nd4j.randn(nIn * nOut, 1);
        INDArray b = Nd4j.randn(nOut, 1);
        shapes.add(new int[]{nIn, nOut});
        return Nd4j.concat(0, w, b);
    }

    INDArray buildNetwork(){
//        INDArray p0 = linear(this.nFeature, 1);
        INDArray p0 = linear(this.nFeature, 30);
        INDArray p1 = linear(30, 1);
        return Nd4j.concat(0, p0, p1);
//        return Nd4j.concat(0, p0);
    }

    double getValue(List<INDArray> p, INDArray x){
//        x = x.mmul(p.get(0)).add(p.get(1));
        x = Transforms.tanh(x.mmul(p.get(0)).add(p.get(1)));
//        x = Transforms.tanh(x.mmul(p.get(2)).add(p.get(3)));
        x = x.mmul(p.get(2)).add(p.get(3));
        return x.getDouble(0,0);
    }

    List<INDArray> paramReshape(INDArray param){
        List<INDArray> params = new ArrayList<>();
        int start = 0;
        for(int i =0; i < shapes.size(); ++i){
            int[] shape = shapes.get(i);
            int nW = shape[0] * shape[1];
            int nb = shape[1];
            params.add(param.get(NDArrayIndex.interval(start, start+nW), NDArrayIndex.all()).reshape(shape));
            params.add(param.get(NDArrayIndex.interval(start+nW, start+nW+nb), NDArrayIndex.all()).reshape(1, shape[1]));
            start += nW + nb;
        }
        return params;
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
            double gameStateScore = evaluateContext(simulationResult, player.getId());  //heuristic.getScore(simulationResult, player.getId());	     //heuristic评估执行gameAction之后的游戏局面的分数
            if (gameStateScore > bestScore) {		// 记录得分最高的action
                bestScore = gameStateScore;
                bestAction = gameAction;
            }
            simulationResult.dispose();  //GameContext环境每次仿真完销毁
        }

        return bestAction;
    }
}
