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

    class SGD{
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

    class Adam{
        double lr;
        double beta1, beta2;
        double eps;
        INDArray m;
        INDArray v;
        int t = 0;
        public Adam(INDArray params, double lr){
            this.m = Nd4j.zeros(params.rows()).transpose();
            this.v = Nd4j.zeros(params.rows()).transpose();
            this.beta1 = 0.9;
            this.beta2 = 0.999;
            this.eps = 1e-8;
            this.t = 0;
            this.lr = lr;
        }

        public INDArray get_gradient(INDArray grad) {
            t++;
            logger.info("t: {}", t);
            logger.info("2: {}, 1: {}, div: {}", Math.sqrt(1 - Math.pow(this.beta2, this.t)), 1 - Math.pow(this.beta1, this.t),
                    Math.sqrt(1 - Math.pow(this.beta2, this.t)) / (1 - Math.pow(this.beta1, this.t)));
            double a = this.lr * Math.sqrt(1 - Math.pow(this.beta2, this.t)) / (1 - Math.pow(this.beta1, this.t));
            logger.info("a: {}, lr: {}", a, this.lr);
            this.m = this.m.mul(this.beta1).add(grad.mul(1-this.beta1));
            this.v = this.v.mul(this.beta2).add(grad.mul(grad).mul(1-this.beta2));
            INDArray step = this.m.mul(-a).div(Transforms.sqrt(this.v).add(this.eps));
            logger.info("m: {}", m.get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
            logger.info("v: {}", m.get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
            return step;
        }
    }

    private final static Logger logger = LoggerFactory.getLogger(LinearBatchES.class);

    private INDArray para; // parameters
    private static INDArray utility;
    private static int NKid = 10;
    private static double LR = 1; //0.001 0.01 0.1 1.0
    double SIGMA = 0.1;
    SGD optimizer;
    private List<int[]> shapes = new ArrayList<>();
    private static List<INDArray> p = new ArrayList<>();
    private static INDArray noise;
    private static double[] kidRewards = new double[NKid * 2];

    // Game Para
    int nFeature = 89;
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
        utility = util_.div(util_.sumNumber().doubleValue()).sub(1.0 / base);
        this.optimizer = new SGD(this.para, LR, 0.9);
//        this.optimizer = new Adam(this.para, LR);
        this.noise = Nd4j.randn(new int[]{NKid, this.para.rows()}).repeat(0, 2);
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
//        INDArray p0 = linear(this.nFeature, 30);
//        INDArray p1 = linear(30, 1);
//        return Nd4j.concat(0, p0, p1);
        return Nd4j.concat(0, p0);
    }

    double getValue(List<INDArray> p, INDArray x){
        x = x.mmul(p.get(0)).add(p.get(1));
//        x = Transforms.tanh(x.mmul(p.get(0)).add(p.get(1)));
//        x = Transforms.tanh(x.mmul(p.get(2)).add(p.get(3)));
//        x = x.mmul(p.get(2)).add(p.get(3));
        return x.getDouble(0,0);
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

        double[] tmp = new double[this.nFeature];
        for (int i=0;i<envState.size();++i){
            tmp[i] = envState.get(i);
        }
        INDArray featureIND = Nd4j.create(tmp);
//        logger.info("Feature: {}", featureIND);
        return getValue(this.p, featureIND);
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
            for (int i=0;i<kidRank.length;++i){
                int kId = kidRank[i];
                // CEM方法
//                cumu = cumu.add(noise.getRow(kId).transpose().mul(sign(kId) * kidRewards[kidRank[i]])); // 原论文中
//                cumu = cumu.add(noise.getRow(kId).transpose().mul(sign(kId) * utility.getDouble(0, i))); // 莫凡
                if (i > 4)
                    break;
                cumu = cumu.add(noise.getRow(kId).transpose().mul(sign(kId)));
            }
            logger.info("Rewards: {}", rewards);
            logger.info("Best Para: {}", this.para.add(noise.getRow(kidRank[0]).transpose().mul(sign(batchCount)*this.SIGMA))
                    .get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));

            if (kidRewards[kidRank[0]] > totalBestReward){
                try{
                    totalBestReward = kidRewards[kidRank[0]];
                    String fileName = "NdModel/network/NdPara_CEM_89fea.data";
                    logger.info("Saving... {} {}", totalBestReward, fileName);
                    File saveFile = new File(fileName);
                    Nd4j.saveBinary(this.para.add(noise.getRow(kidRank[0]).transpose().mul(sign(batchCount)*this.SIGMA)), saveFile);
                }
                catch (IOException e){
                    e.printStackTrace();
                }
            }
            INDArray grad = optimizer.get_gradient(cumu.div(2 * NKid));
//            INDArray grad = optimizer.get_gradient(cumu.div(2 * NKid * SIGMA));
            this.para.addi(grad);
            logger.info("Grad: {}", grad.get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
            logger.info("Para: {}", this.para.get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
            logger.info("LR: {}", LR);
            noise = Nd4j.randn(new int[]{NKid, this.para.rows()}).repeat(0, 2);
            batchCount = 0;
            this.p = paramReshape(this.para.add(this.noise.getRow(batchCount).transpose().mul(sign(batchCount)*this.SIGMA)));
            epoch++;
            if (epoch % 10 == 0){
                LR /= 2;
            }
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
