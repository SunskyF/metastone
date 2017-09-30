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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

public class GameTreeCrossTurn extends Behaviour{
    private final static Logger logger = LoggerFactory.getLogger(net.demilich.metastone.game.behaviour.GameTreeCrossTurn.class);
    private static HashMap<List<Double>, Double> store = new HashMap<>();

    INDArray para; // parameters
    List<INDArray> p = new ArrayList<>();
    List<int[]> shapes = new ArrayList<>();
    // Game Para
    int nFeature = 96;
    // Game Para End

    String paraFile = "NdModel/linear/96feaGameTree_mean_para_ES.data";

    public GameTreeCrossTurn(){
        this.para = buildNetwork();
//        logger.info("shape: {}", this.para.shape());
        try{
            File readFile = new File(paraFile);
            this.para = Nd4j.readBinary(readFile);
//            logger.info("shape: {}", this.para.shape());
//            logger.info("Para: {}", this.para.get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
        }
        catch (IOException e){
            e.printStackTrace();
        }
//        logger.info("Shape: {}", this.para.shape());
        this.p = paramReshape(this.para);
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
//        INDArray p2 = linear(20, 1);
//        return Nd4j.concat(0, p0, p1, p2);
        return Nd4j.concat(0, p0);
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
        return "GameTreeCrossTurn";
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

    private static int calculateThreatLevel(GameContext context, int playerId) {
        int damageOnBoard = 0;
        Player player = context.getPlayer(playerId);
        Player opponent = context.getOpponent(player);
        for (Minion minion : opponent.getMinions()) {
            damageOnBoard += minion.getAttack(); // * minion.getAttributeValue(Attribute.NUMBER_OF_ATTACKS); (暂时没有考虑风怒、冻结等的影响)
        }
        damageOnBoard += getHeroDamage(opponent.getHero());  //对方随从 + 英雄的攻击力

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

    double getValue(List<INDArray> p, INDArray x){
        x = x.mmul(p.get(0)).add(p.get(1));
//        x = Transforms.tanh(x.mmul(p.get(0)).add(p.get(1)));
//        x = Transforms.tanh(x.mmul(p.get(2)).add(p.get(3)));
//        x = x.mmul(p.get(4)).add(p.get(5));
        return x.getDouble(0,0);
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

        // 威胁等级标识特征
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
        envState.add(context.getTurn() / 1.0);

        envState.add((playerFeature.get(0) + 1.0) / (opponentFeature.get(0) + 1.0)); // HP 比值
        envState.add((playerFeature.get(35) + playerFeature.get(38) + playerFeature.get(40) + 1.0) /
                (opponentFeature.get(35) + opponentFeature.get(38) + opponentFeature.get(40) + 1.0)); // 手牌数目 比值
        envState.add((playerFeature.get(6) + playerFeature.get(9) + 1.0) / (opponentFeature.get(6) + opponentFeature.get(9) + 1.0)); // 场上随从数目 比值
        envState.add((playerFeature.get(7) + 1.0) / (opponentFeature.get(7) + 1.0)); // 场上可攻击随从攻击力 比值
        envState.add((playerFeature.get(8) + 1.0) / (opponentFeature.get(8) + 1.0)); // 场上随从血量 比值
        envState.add((playerFeature.get(10) + 1.0) / (opponentFeature.get(10) + 1.0)); // 场上随从血量 比值
        envState.add((playerFeature.get(11) + 1.0) / (opponentFeature.get(11) + 1.0)); // 场上随从血量 比值
        // 96

        double[] tmp = new double[this.nFeature];
        for (int i=0;i<envState.size();++i){
            tmp[i] = envState.get(i);
        }

        INDArray featureIND = Nd4j.create(tmp);
        return getValue(this.p, featureIND);
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
        store.clear();//每次都是只对一次搜索
        GameAction bestAction = validActions.get(0);
        double bestScore = Double.NEGATIVE_INFINITY;

//        logger.info("ValidAct 1: {}", context.getValidActions());

        for (GameAction gameAction : validActions) {
            double score = alphaBeta(context, player.getId(), gameAction, depth);  // 对每一个可能action，使用alphaBeta递归计算得分
            logger.info("Act: {}, score: {}", gameAction, score);
            if (score > bestScore) {
                bestAction = gameAction;
                bestScore = score;
            }
        }
        logger.info("Choosed: {}", bestAction);
        return bestAction;
    }

    private double alphaBeta(GameContext context, int playerId, GameAction action, int depth) {
        GameContext simulation = context.clone();  // clone目前环境
        simulation.getLogic().performGameAction(playerId, action);  // 在拷贝环境中执行action
        if (depth == 0 || simulation.gameDecided()) {  // depth层递归结束、发生玩家切换（我方这轮打完了）或者比赛结果已定时，返回score
            return evaluateContext(simulation, playerId);
        }
        if (simulation.getActivePlayerId() != playerId){
//            logger.info("Opponent Turn, depth: {}, action: {}", depth, action);
            GameContext simulationOppo = simulation.clone();
            simulationOppo.startTurn(simulation.getActivePlayerId(), true); // 对手回合

            List<GameAction> validAct = simulationOppo.getValidActions();
            GameAction act = validAct.get(new Random().nextInt(validAct.size()));
            while(act.getActionType() != ActionType.END_TURN){
                simulationOppo.getLogic().performGameAction(simulationOppo.getActivePlayerId(), act);
                if (simulationOppo.gameDecided()){
                    return Float.NEGATIVE_INFINITY;
                }
                validAct = simulationOppo.getValidActions();
//                logger.info("act: {}, size: {}", act, validAct.size());
                act = validAct.get(new Random().nextInt(validAct.size()));
            }
            simulationOppo.getLogic().performGameAction(simulationOppo.getActivePlayerId(), act);// 对手回合结束
            double sum = 0;
            for (int i =0;i<5;++i){
                double bestScore = Double.NEGATIVE_INFINITY;
                GameContext temp = simulationOppo.clone();
                temp.startTurn(temp.getActivePlayerId()); // 我方回合开始
                for (GameAction gameAction : temp.getValidActions()) {
                    double score = alphaBeta(temp, temp.getActivePlayerId(), gameAction, depth-1);
                    if (score > bestScore) {
                        bestScore = score;
                    }
                }
                sum += bestScore;
                temp.dispose();
            }
            return sum / 5.0;
        }

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
        envState.add((playerFeature.get(10) + 1.0) / (opponentFeature.get(10) + 1.0)); // 场上不可攻击随从血量 比值
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
}
