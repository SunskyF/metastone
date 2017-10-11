package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.ActionType;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.behaviour.heuristic.IGameStateHeuristic;
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
import java.util.*;

public class GameTreeBestMoveND extends Behaviour{

    private final static Logger logger = LoggerFactory.getLogger(net.demilich.metastone.game.behaviour.GameTreeBestMoveND.class);
    private static HashMap<List<Double>, Double> store = new HashMap<>();

    INDArray para; // parameters
    List<INDArray> p = new ArrayList<>();
    List<int[]> shapes = new ArrayList<>();
    private int phase = 0;

    // Game Para
    int nFeature = 96; // 当特征改变时，需要更改这里的特征数目
    // Game Para End

    String paraFile = "NdModel/linear/96feaGameTree_mean_para_ES.data";

    // Simulation Count
    private static int evaluateCnt = 0;
    private static int requestCnt = 0;
    private int localEvaluateCnt = 0;
    // Sim Count End
    public GameTreeBestMoveND(){
        this.para = buildNetwork(); // 这步是必须的，需要用来获得网络的shape
//        logger.info("shape: {}", this.para.shape());
        try{ // 读取模型文件
            logger.info("Loading... {}", paraFile);
            File readFile = new File(paraFile);
            this.para = Nd4j.readBinary(readFile);
//            logger.info("shape: {}", this.para.shape());
//            logger.info("Para: {}", this.para.get(NDArrayIndex.interval(0, 10), NDArrayIndex.all()));
            logger.info("Loaded...");
        }
        catch (IOException e){
            e.printStackTrace();
        }
//        logger.info("Shape: {}", this.para.shape());
        this.p = paramReshape(this.para); // 将模型reshape成我们需要的模样，主要是将w和b分离
    }

    INDArray linear(int nIn, int nOut){
        INDArray w = Nd4j.randn(nIn * nOut, 1);
        INDArray b = Nd4j.randn(nOut, 1);
        shapes.add(new int[]{nIn, nOut});
        return Nd4j.concat(0, w, b);
    }

    INDArray buildNetwork(){
        // 更换模型时需要同时修改buildNetwork和getValue
        // 线性模型
        INDArray p0 = linear(this.nFeature, 1);
        return Nd4j.concat(0, p0);
        // 线性模型 End
        // 网络模型
//        INDArray p0 = linear(this.nFeature, 30);
//        INDArray p1 = linear(30, 20);
//        INDArray p2 = linear(20, 1);
//        return Nd4j.concat(0, p0, p1, p2);
        // 网络模型 End
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
        return "GameTreeBestMoveND";
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

    private static int calculateThreatLevel(Player player, Player opponent) {
        int damageOnBoard = 0;
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
        // 线性模型
        x = x.mmul(p.get(0)).add(p.get(1));
        return x.getDouble(0,0);
        // 线性模型 End
        // 网络模型
//        x = Transforms.tanh(x.mmul(p.get(0)).add(p.get(1)));
//        x = Transforms.tanh(x.mmul(p.get(2)).add(p.get(3)));
//        x = x.mmul(p.get(4)).add(p.get(5));
//        return x.getDouble(0,0);
        // 网络模型 End

    }

    private List<Double> getFeature(Player player, Player opponent, double turn){

        List<Double> playerFeature = player.getPlayerStateDouble();
        List<Double> opponentFeature  = opponent.getPlayerStateDouble();
        List<Double> envState = new ArrayList<>();
        envState.addAll(playerFeature);
        envState.addAll(opponentFeature);

        // 威胁等级标识特征
        int threatLevelHigh= 0;
        int threatLevelMiddle = 0;
        int threatLevel = calculateThreatLevel(player, opponent);
        if(threatLevel == 2){
            threatLevelHigh = 1;
        }else if(threatLevel == 1){
            threatLevelMiddle = 1;
        }
        envState.add(threatLevelHigh / 1.0);
        envState.add(threatLevelMiddle / 1.0);
        envState.add(turn);

        envState.add((playerFeature.get(0) + 1.0) / (opponentFeature.get(0) + 1.0)); // HP 比值
        envState.add((playerFeature.get(35) + playerFeature.get(38) + playerFeature.get(40) + 1.0) /
                (opponentFeature.get(35) + opponentFeature.get(38) + opponentFeature.get(40) + 1.0)); // 手牌数目 比值
        envState.add((playerFeature.get(6) + playerFeature.get(9) + 1.0) / (opponentFeature.get(6) + opponentFeature.get(9) + 1.0)); // 场上随从数目 比值
        envState.add((playerFeature.get(7) + 1.0) / (opponentFeature.get(7) + 1.0)); // 场上可攻击随从攻击力 比值
        envState.add((playerFeature.get(8) + 1.0) / (opponentFeature.get(8) + 1.0)); // 场上可攻击随从血量 比值
        envState.add((playerFeature.get(10) + 1.0) / (opponentFeature.get(10) + 1.0)); // 场上不可攻击随从攻击力 比值
        envState.add((playerFeature.get(11) + 1.0) / (opponentFeature.get(11) + 1.0)); // 场上不可攻击随从血量 比值
        // 96
        return envState;
    }

    private double evaluateContext(GameContext context, int playerId) {
        evaluateCnt++;
        localEvaluateCnt++;
        Player player = context.getPlayer(playerId);
        Player opponent = context.getOpponent(player);
        if (player.getHero().isDestroyed()) {   // 己方被干掉，得分 负无穷
            return Float.NEGATIVE_INFINITY;  // 正负无穷会影响envState的解析，如果要加的话可以改成 +-100之类的
        }
        if (opponent.getHero().isDestroyed()) {  // 对方被干掉，得分 正无穷
            return Float.POSITIVE_INFINITY;
        }

        List<Double> envState = getFeature(player, opponent, context.getTurn());
        double[] tmp = new double[this.nFeature];
        for (int i=0;i<envState.size();++i){ // 使得特征能转为nd4j的形式
            tmp[i] = envState.get(i);
        }

        INDArray featureIND = Nd4j.create(tmp);
        return getValue(this.p, featureIND);
    }

    @Override
    public GameAction requestAction(GameContext context, Player player, List<GameAction> validActions) {
        requestCnt++;
        localEvaluateCnt = 0;
        if (requestCnt % 100 == 0)
            logger.info("requestCnt: {}, evaluateCnt: {}", requestCnt, evaluateCnt);

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

        store.clear();//每次都是只对一次搜索，清空哈希表
        GameAction bestAction = validActions.get(0);
        double bestScore = Double.NEGATIVE_INFINITY;

        // 当修改剪枝操作版本时，记得修改alphabeta函数中的剪枝操作
        // 基础版本
        for (GameAction gameAction: validActions){ //基础的方法
            double score = alphaBeta(context, player.getId(), gameAction, depth);
            if (score > bestScore){
                bestAction = gameAction;
                bestScore = score;
            }
        }
        // 基础版本 End

        // 分阶段，区分出牌和平A
//        phase = 1;
//        for (GameAction gameAction : validActions){
//            if ((gameAction.getActionType() != ActionType.PHYSICAL_ATTACK) && (gameAction.getActionType() != ActionType.END_TURN)){
//                phase = 0;
//                break;
//            }
//        }
//        for (GameAction gameAction : validActions) {
//            if (phase == 0 && (gameAction.getActionType() != ActionType.PHYSICAL_ATTACK) && (gameAction.getActionType() != ActionType.END_TURN)){
//                double score = alphaBeta(context, player.getId(), gameAction, depth);  // 对每一个可能action，使用alphaBeta递归计算得分
//                if (score > bestScore) {
//                    bestAction = gameAction;
//                    bestScore = score;
//                }
//            }
//            else if (phase == 1 && (gameAction.getActionType() == ActionType.PHYSICAL_ATTACK || gameAction.getActionType() == ActionType.END_TURN)){
//                double score = alphaBeta(context, player.getId(), gameAction, depth);  // 对每一个可能action，使用alphaBeta递归计算得分
//                if (score > bestScore) {
//                    bestAction = gameAction;
//                    bestScore = score;
//                }
//            }
//        }
        // 分阶段，区分出牌和平A End

        // 选取最优k个剪枝
//        SortedMap<Double, GameAction> scoreActionMap = new TreeMap<>(Comparator.reverseOrder());
//        for (GameAction gameAction : validActions) {  // 遍历validactions，使用Linear评估函数评估得到的局面，并按得分降序排列
//            GameContext simulationResult = simulateAction(context.clone(), player.getId(), gameAction);  //假设执行gameAction，得到之后的game context
//            double gameStateScore = evaluateContext(simulationResult, player.getId());	//heuristic.getScore(simulationResult, player.getId());     //heuristic评估执行gameAction之后的游戏局面的分数
//            if(!scoreActionMap.containsKey(gameStateScore)){  // 注意：暂时简单的认为gameStateScore相同的两个simulationResult context一样，只保留第一个simulationResult对应的action
//                scoreActionMap.put(gameStateScore, gameAction);
//            }
//            simulationResult.dispose();  //GameContext环境每次仿真完销毁
//        }
//
//        int k = 0;
//        for(GameAction gameAction: scoreActionMap.values()){
//            double score = alphaBeta(context, player.getId(), gameAction, depth);  // 对每一个可能action，使用alphaBeta递归计算得分
//            if (score > bestScore) {
//                bestAction = gameAction;
//                bestScore = score;
//            }
//            k += 1;
//            if(k >= 2){
//                break;
//            }
//        }
        // 选取最优k个剪枝 End

        logger.info("Local Evaluate Cnt: {}", localEvaluateCnt);
        return bestAction;
    }

    private GameContext simulateAction(GameContext simulation, int playerId, GameAction action) {
        simulation.getLogic().performGameAction(playerId, action);   // 在simulation GameContext中执行action，似乎是获取logic模块来执行action的
        return simulation;
    }

    private double alphaBeta(GameContext context, int playerId, GameAction action, int depth) {
        GameContext simulation = context.clone();  // clone目前环境
        simulation.getLogic().performGameAction(playerId, action);  // 在拷贝环境中执行action
        if (depth == 0 || simulation.getActivePlayerId() != playerId || simulation.gameDecided()) {  // depth层递归结束、发生玩家切换（我方这轮打完了）或者比赛结果已定时，返回score
            return evaluateContext(simulation, playerId);
        }

        List<GameAction> validActions = simulation.getValidActions();  //执行完一个action之后，获取接下来可以执行的action

        double score = Float.NEGATIVE_INFINITY;

        // 合并状态  需要在开始搜索前查找一次哈希表和结束搜索后将结果保存入哈希表
//        Player player = simulation.getPlayer(playerId);
//        Player opponent = simulation.getOpponent(player);
//        if (player.getHero().isDestroyed()) {   // 己方被干掉，得分 负无穷
//            return Float.NEGATIVE_INFINITY;  // 正负无穷会影响envState的解析，如果要加的话可以改成 +-100之类的
//        }
//        if (opponent.getHero().isDestroyed()) {  // 对方被干掉，得分 正无穷
//            return Float.POSITIVE_INFINITY;
//        }
//        List<Double> envState = getFeature(player, opponent, simulation.getTurn());

//        if (store.containsKey(envState)) // 如果哈希表中有特征向量，则直接从哈希表中获得
//            return getStoredValue(envState);
        // 合并状态 End

        // 基础版本
        for (GameAction gameAction: validActions){
            score = Math.max(score, alphaBeta(simulation, playerId, gameAction, depth - 1));  // 递归调用alphaBeta，取评分较大的
            if (score >= 100000) {
                break;
            }
        }
        // 基础版本 End

        // 选取最优k个剪枝
//        SortedMap<Double, GameAction> scoreActionMap = new TreeMap<>(Comparator.reverseOrder());
//        for (GameAction gameAction : validActions) {  // 遍历validactions，使用Linear评估函数评估得到的局面，并按得分降序排列
//            GameContext simulationResult = simulateAction(simulation.clone(), playerId, gameAction);  //假设执行gameAction，得到之后的game context
//            double gameStateScore = evaluateContext(simulationResult, playerId);  //heuristic.getScore(simulationResult, playerId);  //heuristic评估执行gameAction之后的游戏局面的分数
//            if(!scoreActionMap.containsKey(gameStateScore)){  // 注意：暂时简单的认为gameStateScore相同的两个simulationResult context一样，只保留第一个simulationResult对应的action
//                scoreActionMap.put(gameStateScore, gameAction);
//            }
//            simulationResult.dispose();  //GameContext环境每次仿真完销毁
//        }
//
//        int k = 0;
//        for(GameAction gameAction: scoreActionMap.values()){
//            score = Math.max(score, alphaBeta(simulation, playerId, gameAction, depth - 1));  // 递归调用alphaBeta，取评分较大的
//            k += 1;
//            if (score >= 100000 || k >= 2) {
//                break;
//            }
//        }
        // 选取最优k个剪枝 End

        // 分阶段，区分出牌和平A
//        for (GameAction gameAction : validActions) {
//            if (phase == 0 && (gameAction.getActionType() != ActionType.PHYSICAL_ATTACK) && (gameAction.getActionType() != ActionType.END_TURN)){
//                score = Math.max(score, alphaBeta(simulation, playerId, gameAction, depth - 1));  // 递归调用alphaBeta，取评分较大的
//                if (score >= 100000) {
//                    break;
//                }
//            }
//            else if (phase == 1 && (gameAction.getActionType() == ActionType.PHYSICAL_ATTACK || gameAction.getActionType() == ActionType.END_TURN)){
//                score = Math.max(score, alphaBeta(simulation, playerId, gameAction, depth - 1));  // 递归调用alphaBeta，取评分较大的
//                if (score >= 100000) {
//                    break;
//                }
//            }
//        }
//        if (score == Float.NEGATIVE_INFINITY){ // 这步很重要，当这里没有符合条件的action时，就直接进行局面评估
//            score = evaluateContext(simulation, playerId);
//        }
        // 分阶段，区分出牌和平A End

        // 合并状态
//        putStoredValue(envState, score);
        // 合并状态 End
        return score;
    }

    private double getStoredValue(List<Double> envState){
        return store.get(envState); // 多线程模拟会报错
    }

    private void putStoredValue(List<Double> envState, double score){
        store.put(envState, score);
    }
}
