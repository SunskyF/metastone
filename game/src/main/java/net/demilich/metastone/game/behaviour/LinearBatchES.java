package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.logic.GameLogic;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;


public class LinearBatchES extends Behaviour {

    public class SGD{
        INDArray v;
        double lr;
        double momentum;
        public SGD(INDArray params, double lr, double momentum){
            this.lr = lr;
            this.momentum = momentum;
            this.v = Nd4j.zeros(params.shape());
        }

        public INDArray get_gradient(INDArray grad){
//            this.v = this.v.mul(this.momentum).add(grad.mul(1 - this.momentum));
//            return this.v.mul(this.lr);
            return grad.mul(this.lr);
        }
    }

    private final static Logger logger = LoggerFactory.getLogger(LinearBatchES.class);

    private INDArray para; // parameters
    private static INDArray utility;
    private static int NKid = 10;
    private static double LR = 1; //0.001 0.01 0.1 1.0
    double SIGMA = 0.5;
    SGD optimizer;
    private List<int[]> shapes = new ArrayList<>();
    private static List<INDArray> p = new ArrayList<>();
    private static INDArray noise;
    private static double[] kidRewards = new double[NKid * 2];

    // Game Para
    int nFeature = 28;
    // Game Para End

    // Simulation count
    private static int gameCount = 0;
    private static int batchCount = 0;
    private static int batchWinCnt = 0;
    private final static int batchSize = 50;
    private static int epoch = 0;
    private static double totalBestReward = -1;
    // Sim End

    public LinearBatchES(){
//        logger.info("Constructor");
        this.para = buildNetwork(); // value network
        // readBinary
//        try{
//            logger.info("Loading...");
//            File readFile = new File("NdPara_3layerNetwork.data");
//            Nd4j.readBinary(readFile);
//        }
//        catch (IOException e){
//            e.printStackTrace();
//        }
        int base = NKid * 2;
        INDArray rank = Nd4j.arange(1, base+1);
        INDArray util_ = Transforms.max(Transforms.log(rank).sub(Math.log(base / 2.0 + 1)).neg(), 0);
//        utility = util_.div(util_.sumNumber().doubleValue()).sub(1.0 / base);
        this.utility = util_.div(util_.sumNumber().doubleValue());
        this.optimizer = new SGD(this.para, LR, 0.9);
        this.noise = Nd4j.randn(new int[]{NKid, this.para.rows()}).repeat(0, 2);
//        logger.info("Origin Noise: {}", this.noise.getRow(batchCount).transpose().get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
//        logger.info("Normal Noise: {}", this.noise.getRow(batchCount).transpose().mul(sign(batchCount)*this.SIGMA).get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
//        logger.info("Origin Para: {}", this.para.get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
//        logger.info("Normal Para: {}", this.para.add(this.noise.getRow(batchCount).transpose().mul(sign(batchCount)*this.SIGMA)).get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
        this.p = paramReshape(this.para.add(this.noise.getRow(batchCount).transpose().mul(sign(batchCount)*this.SIGMA)));
    }

    INDArray linear(int nIn, int nOut){
        INDArray w = Nd4j.randn(nIn * nOut, 1);
        INDArray b = Nd4j.randn(nOut, 1);
        shapes.add(new int[]{nIn, nOut});
        return Nd4j.concat(0, w, b);
    }

    INDArray buildNetwork(){
        INDArray p0 = linear(this.nFeature, 1);
//        INDArray p1 = linear(30, 20);
//        INDArray p2 = linear(20, 1);
        return Nd4j.concat(0, p0);
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
        envState.addAll(opponent.getPlayerStatefh0(true));
        double[] tmp = new double[this.nFeature];
        for (int i=0;i<envState.size();++i){
            tmp[i] = envState.get(i);
        }
        INDArray featureIND = Nd4j.create(tmp);
//        logger.info("{}", featureIND);
        return getValue(this.p, featureIND);
    }

    double getValue(List<INDArray> p, INDArray x){
        x = x.mmul(p.get(0)).add(p.get(1));
//        x = Transforms.tanh(x.mmul(p.get(0)).add(p.get(1)));
//        x = Transforms.tanh(x.mmul(p.get(2)).add(p.get(3)));
//        x = x.mmul(p.get(4)).add(p.get(5));
//        logger.info("value: {}", x);
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

    public static <T extends Number> int[] asArray(final T... a) {
        int[] b = new int[a.length];
        for (int i = 0; i < b.length; i++) {
            b[i] = a[i].intValue();
        }
        return b;
    }

    @Override
    public String getName() {
        return "Linear-Batch-ES";
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
        if(playerId == winningPlayerId){
            batchWinCnt += 1;
        }
//        logger.info("Player: {}, Winner: {}", playerId, winningPlayerId);

        if (gameCount == batchSize){ // 第一层，计算一个batch的胜率作为Reward
            kidRewards[batchCount] = batchWinCnt;
            logger.info("BatchCount: {}, BatchWinCount: {}", batchCount, batchWinCnt);
            batchCount++; // KId
            if (batchCount < NKid * 2){
                this.p = paramReshape(this.para.add(this.noise.getRow(batchCount).transpose().mul(sign(batchCount)*this.SIGMA)));
            }
            gameCount = 0;
            batchWinCnt = 0;
        }

        if (batchCount == NKid * 2){ // 第二层，train one step
            INDArray rewards = Nd4j.create(kidRewards);
            double meanReward = rewards.meanNumber().doubleValue();
            logger.info("Epoch: {}, reward: {}, BestReward: {}", epoch, meanReward, totalBestReward);

            Integer[] kidRank = argsort(kidRewards, false);
            INDArray cumu = Nd4j.zeros(this.para.shape());
            double flag = 0;
            for (int i=0;i<kidRank.length;++i){
                int kId = kidRank[i];
                if (i < 5){
                    flag = 1;
                }else
                    flag = 0;
//                cumu = cumu.add(noise.getRow(kId).transpose().mul(sign(kId) * utility.getDouble(0, i)));
                cumu = cumu.add(noise.getRow(kId).transpose().mul(sign(kId) * flag));
            }
            logger.info("Rewards: {}", rewards);
            logger.info("Best Para: {}", this.para.add(noise.getRow(kidRank[0]).transpose().mul(sign(batchCount)*this.SIGMA))
                    .get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));

            if (kidRewards[kidRank[0]] > totalBestReward){
                try{
                    totalBestReward = kidRewards[kidRank[0]];
                    logger.info("Saving... {}", totalBestReward);
                    File saveFile = new File("NdPara_3layerNetwork.data");
                    Nd4j.saveBinary(this.para.add(noise.getRow(kidRank[0]).transpose()), saveFile);
                }
                catch (IOException e){
                    e.printStackTrace();
                }
            }
//            logger.info("Cumu: {}", cumu.div(2 * NKid * SIGMA));
            INDArray grad = optimizer.get_gradient(cumu.div(2 * NKid * SIGMA));
            this.para.addi(grad);
            logger.info("Grad: {}", grad.get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
            logger.info("Para: {}", this.para.get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
            noise = Nd4j.randn(new int[]{NKid, this.para.rows()}).repeat(0, 2);
            batchCount = 0;
            this.p = paramReshape(this.para.add(this.noise.getRow(batchCount).transpose().mul(sign(batchCount)*this.SIGMA)));
            epoch++;
        }
    }

    public static Integer[] argsort(final double[] a, final boolean ascending) {
        Integer[] indexes = new Integer[a.length];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return (ascending ? 1 : -1) * Double.compare(a[i1], a[i2]);
            }
        });
        return indexes;
    }

    double sign(int i){
        if (i % 2 == 0)
            return 1;
        else
            return -1;
    }
}
