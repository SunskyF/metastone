package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.ActionType;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.entities.heroes.Hero;
import net.demilich.metastone.game.entities.heroes.HeroClass;
import net.demilich.metastone.game.entities.minions.Minion;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class GameTreeBatchES extends Behaviour{

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

    private final static Logger logger = LoggerFactory.getLogger(GameTreeBatchES.class);

    private INDArray para; // parameters
    private static INDArray utility;
    private static int NKid = 10;
    private static double LR = 1;
    double SIGMA = 0.5;
    SGD optimizer;
    private List<int[]> shapes = new ArrayList<>();
    private static List<INDArray> p = new ArrayList<>();
    private static INDArray noise;
    private static double[] kidRewards = new double[NKid * 2];

    // Game Para
    int nFeature = 96;
    // Game Para End

    // Simulation count
    private static int gameCount = 0;
    private static int batchCount = 0;
    private static int batchWinCnt = 0;
    private final static int batchSize = 50;
    private static int epoch = 0;
    private static double totalBestReward = -1;
    private static double totalMeanReward = -1;
    private static HashMap<List<Double>, Double> store = new HashMap<>();
    // Sim End

    public GameTreeBatchES(){
//        logger.info("Constructor");
        this.para = buildNetwork(); // value network
        // readBinary
        try{
            logger.info("Loading...");
            File readFile = new File("NdModel/linear/96feaGameTree_mean_para_ES.data");
            this.para = Nd4j.readBinary(readFile);
        }
        catch (IOException e){
            e.printStackTrace();
        }
        int base = NKid * 2;
        INDArray rank = Nd4j.arange(1, base+1);
        INDArray util_ = Transforms.max(Transforms.log(rank).sub(Math.log(base / 2.0 + 1)).neg(), 0);
        utility = util_.div(util_.sumNumber().doubleValue()).sub(1.0 / base);
        this.optimizer = new SGD(this.para, LR, 0.9);
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
//        INDArray p1 = linear(30, 20);
//        INDArray p2 = linear(30, 1);
//        return Nd4j.concat(0, p0, p2);
        return Nd4j.concat(0, p0);
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
        List<Double> playerFeature = player.getPlayerStateDouble();
        List<Double> opponentFeature  = opponent.getPlayerStateDouble();
        List<Double> envState = new ArrayList<>();
        envState.addAll(playerFeature);
        envState.addAll(opponentFeature);

        int threatLevelHigh= 0;
        int threatLevelMiddle = 0;
        int threatLevel = calculateThreatLevel(context, playerId);
        if(threatLevel == 2){
            threatLevelHigh = 1;
        }else if(threatLevel == 1){
            threatLevelMiddle = 1;
        }
        envState.add(threatLevelHigh / 1.0);
        envState.add(threatLevelMiddle / 1.0);
        envState.add(context.getTurn() / 1.0); // 89
        // 增加比例特征
        envState.add((playerFeature.get(0) + 1.0) / (opponentFeature.get(0) + 1.0)); // HP 比值
        envState.add((playerFeature.get(35) + playerFeature.get(38) + playerFeature.get(40) + 1.0) /
                (opponentFeature.get(35) + opponentFeature.get(38) + opponentFeature.get(40) + 1.0)); // 手牌数目 比值
        envState.add((playerFeature.get(6) + playerFeature.get(9) + 1.0) / (opponentFeature.get(6) + opponentFeature.get(9) + 1.0)); // 场上随从数目 比值
        envState.add((playerFeature.get(7) + 1.0) / (opponentFeature.get(7) + 1.0)); // 场上可攻击随从攻击力 比值
        envState.add((playerFeature.get(8) + 1.0) / (opponentFeature.get(8) + 1.0)); // 场上随从血量 比值
        envState.add((playerFeature.get(10) + 1.0) / (opponentFeature.get(10) + 1.0)); // 场上不可攻击随从攻击力 比值
        envState.add((playerFeature.get(11) + 1.0) / (opponentFeature.get(11) + 1.0)); // 场上不可攻击随从血量 比值
        // 96

        double[] tmp = new double[this.nFeature];
        for (int i=0;i<envState.size();++i){
            tmp[i] = envState.get(i);
        }
//        logger.info("tmp: {}, envState: {}", tmp.length, envState.size());
        INDArray featureIND = Nd4j.create(tmp);
        return getValue(this.p, featureIND);
    }

    double getValue(List<INDArray> p, INDArray x){
        x = x.mmul(p.get(0)).add(p.get(1));
//        x = Transforms.tanh(x.mmul(p.get(0)).add(p.get(1)));
//        x = Transforms.tanh(x.mmul(p.get(2)).add(p.get(3)));
//        x = x.mmul(p.get(2)).add(p.get(3));
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
    public String getName() {
        return "GameTree-Batch-ES";
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

        int depth = 2;
        // when evaluating battlecry and discover actions, only optimize the immediate value （两种特殊的action）
        if (validActions.get(0).getActionType() == ActionType.BATTLECRY) {
            depth = 0;
        } else if (validActions.get(0).getActionType() == ActionType.DISCOVER) {  // battlecry and discover actions一定会在第一个么？
            return validActions.get(0);
        }
        GameAction bestAction = validActions.get(0);

        double bestScore = Double.NEGATIVE_INFINITY;
        store.clear();
        for (GameAction gameAction : validActions) {
            double score = alphaBeta(context, player.getId(), gameAction, depth);  // 对每一个可能action，使用alphaBeta递归计算得分
            if (score > bestScore) {
                bestAction = gameAction;
                bestScore = score;
            }
        }
//        if (bestAction.getActionType() == ActionType.END_TURN){
//            store.clear();//每次都是只对一次搜索
//        }
        return bestAction;
    }

    private double alphaBeta(GameContext context, int playerId, GameAction action, int depth) {
        GameContext simulation = context.clone();  // clone目前环境
        simulation.getLogic().performGameAction(playerId, action);  // 在拷贝环境中执行action
        if (depth == 0 || simulation.gameDecided() || simulation.getActivePlayerId() != playerId) {  // depth层递归结束、发生玩家切换（我方这轮打完了）或者比赛结果已定时，返回score
            return evaluateContext(simulation, playerId); //  || simulation.gameDecided()
        }

//        if (simulation.getActivePlayerId() != playerId){
//            simulation.startTurn(simulation.getActivePlayerId(), true);
//            List<GameAction> validAct = simulation.getValidActions();
//            simulation.getLogic().performGameAction(simulation.getActivePlayerId(), validAct.get(validAct.size()-1));
//            double sum = 0;
//            for (int i =0;i<5;++i){
//                double bestScore = Double.NEGATIVE_INFINITY;
//                GameContext temp = simulation.clone();
//                temp.startTurn(simulation.getActivePlayerId());
//                for (GameAction gameAction : temp.getValidActions()) {
//                    double score = alphaBeta(temp, temp.getActivePlayerId(), gameAction, depth-1);
//                    if (score > bestScore) {
//                        bestScore = score;
//                    }
//                }
//                sum += bestScore;
//                temp.dispose();
//            }
//            return sum / 5.0;
//        }

        List<GameAction> validActions = simulation.getValidActions();  //执行完一个action之后，获取接下来可以执行的action

        double score = Float.NEGATIVE_INFINITY;
        Player player = simulation.getPlayer(playerId);
        Player opponent = simulation.getOpponent(player);
        if (player.getHero().isDestroyed()) {   // 己方被干掉，得分 负无穷
            return Float.NEGATIVE_INFINITY;  // 正负无穷会影响envState的解析，如果要加的话可以改成 +-100之类的
        }
        if (opponent.getHero().isDestroyed()) {  // 对方被干掉，得分 正无穷
            return Float.POSITIVE_INFINITY;
        }
        List<Double> playerFeature = player.getPlayerStateDouble();
        List<Double> opponentFeature  = opponent.getPlayerStateDouble();
        List<Double> envState = new ArrayList<>();
        envState.addAll(playerFeature);
        envState.addAll(opponentFeature);

        int threatLevelHigh= 0;
        int threatLevelMiddle = 0;
        int threatLevel = calculateThreatLevel(simulation, playerId);
        if(threatLevel == 2){
            threatLevelHigh = 1;
        }else if(threatLevel == 1){
            threatLevelMiddle = 1;
        }
        envState.add(threatLevelHigh / 1.0);
        envState.add(threatLevelMiddle / 1.0);
        envState.add(simulation.getTurn() / 1.0); // 89
        // 增加比例特征
        envState.add((playerFeature.get(0) + 1.0) / (opponentFeature.get(0) + 1.0)); // HP 比值
        envState.add((playerFeature.get(35) + playerFeature.get(38) + playerFeature.get(40) + 1.0) /
                (opponentFeature.get(35) + opponentFeature.get(38) + opponentFeature.get(40) + 1.0)); // 手牌数目 比值
        envState.add((playerFeature.get(6) + playerFeature.get(9) + 1.0) / (opponentFeature.get(6) + opponentFeature.get(9) + 1.0)); // 场上随从数目 比值
        envState.add((playerFeature.get(7) + 1.0) / (opponentFeature.get(7) + 1.0)); // 场上可攻击随从攻击力 比值
        envState.add((playerFeature.get(8) + 1.0) / (opponentFeature.get(8) + 1.0)); // 场上随从血量 比值
        envState.add((playerFeature.get(10) + 1.0) / (opponentFeature.get(10) + 1.0)); // 场上不可攻击随从攻击 比值
        envState.add((playerFeature.get(11) + 1.0) / (opponentFeature.get(11) + 1.0)); // 场上不可攻击随从血量 比值
        // 96

        if (store.containsKey(envState)){
            return store.get(envState); // 多线程模拟会报错
        }


        for (GameAction gameAction : validActions) {
            score = Math.max(score, alphaBeta(simulation, playerId, gameAction, depth - 1));  // 递归调用alphaBeta，取评分较大的
            if (score >= 100000) {
                break;
            }
        }
        store.put(envState, score);
        return score;
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
//                cumu = cumu.add(noise.getRow(kId).transpose().mul(sign(kId) * kidRewards[kidRank[i]])); // 原论文中
//                cumu = cumu.add(noise.getRow(kId).transpose().mul(sign(kId) * utility.getDouble(0, i))); // 莫凡
                if (i > 4)
                    break;
                cumu = cumu.add(noise.getRow(kId).transpose().mul(sign(kId)));
            }
            logger.info("Rewards: {}", rewards);
//            logger.info("Best Para: {}", this.para.add(noise.getRow(kidRank[0]).transpose().mul(sign(batchCount)*this.SIGMA))
//                    .get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
            logger.info("Best Para: {}", this.para.get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));

            if (kidRewards[kidRank[0]] > totalBestReward){
                try{
                    totalBestReward = kidRewards[kidRank[0]];
                    logger.info("Saving... {}", totalBestReward);
                    File saveFile = new File("NdModel/linear/96feaGameTree_best_para_ES_control_warrior.data");
                    Nd4j.saveBinary(this.para.add(noise.getRow(kidRank[0]).transpose().mul(sign(batchCount)*this.SIGMA)), saveFile);
                }
                catch (IOException e){
                    e.printStackTrace();
                }
            }
            if (meanReward > totalMeanReward){
                try{
                    totalMeanReward = meanReward;
                    logger.info("Saving... {}", totalMeanReward);
                    File saveFile = new File("NdModel/linear/96feaGameTree_mean_para_ES_control_warrior.data");
                    Nd4j.saveBinary(this.para, saveFile);
//                    File saveFileTop = new File("NdModel/linear/98feaGameTree_topmean_para_ES_basic.data");
//                    Nd4j.saveBinary(this.para.add(cumu.div(5)), saveFileTop);
                }
                catch (IOException e){
                    e.printStackTrace();
                }
            }
            INDArray grad = optimizer.get_gradient(cumu.div(2 * NKid * SIGMA));
            this.para.addi(grad);
            logger.info("Grad: {}", grad.get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
            logger.info("Para: {}", this.para.get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
            noise = Nd4j.randn(new int[]{NKid, this.para.rows()}).repeat(0, 2);
            batchCount = 0;
            this.p = paramReshape(this.para.add(this.noise.getRow(batchCount).transpose().mul(sign(batchCount)*this.SIGMA)));
            epoch++;
            if (epoch % 10 == 0){
                LR = LR / 2;
                SIGMA = SIGMA / 2;
            }
        }
    }

    private static int calculateThreatLevel(GameContext context, int playerId) {
        int damageOnBoard = 0;
        Player player = context.getPlayer(playerId);
        Player opponent = context.getOpponent(player);
        for (Minion minion : opponent.getMinions()) {
            damageOnBoard += minion.getAttack();
        }
        damageOnBoard += getHeroDamage(opponent.getHero());  //对方随从 + 英雄的攻击力 (暂时没有考虑风怒、冻结等的影响，因为 之前 minion.getAttributeValue(Attribute.NUMBER_OF_ATTACKS)经常得到0)

        int remainingHp = player.getHero().getEffectiveHp() - damageOnBoard;  // 根据减去对方伤害后我方剩余血量来确定威胁等级
        if (remainingHp < 1) {
            return 2;
        } else if (remainingHp < 15) {
            return 1;
        }
        return 0;
    }

    private static int getHeroDamage(Hero hero) {
        int heroDamage = 0;
        if (hero.getHeroClass() == HeroClass.MAGE) {
            heroDamage += 1;
        } else if (hero.getHeroClass() == HeroClass.HUNTER) {
            heroDamage += 2;
        } else if (hero.getHeroClass() == HeroClass.DRUID) {
            heroDamage += 1;
        } else if (hero.getHeroClass() == HeroClass.ROGUE) {
            heroDamage += 1;
        }
        if (hero.getWeapon() != null) {
            heroDamage += hero.getWeapon().getWeaponDamage();
        }
        return heroDamage;
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
