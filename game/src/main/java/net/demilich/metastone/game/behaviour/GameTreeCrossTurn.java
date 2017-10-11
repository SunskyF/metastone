package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.ActionType;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.entities.heroes.Hero;
import net.demilich.metastone.game.entities.heroes.HeroClass;
import net.demilich.metastone.game.entities.minions.Minion;
import net.demilich.metastone.game.logic.GameLogic;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class GameTreeCrossTurn extends Behaviour{
    private final static Logger logger = LoggerFactory.getLogger(net.demilich.metastone.game.behaviour.GameTreeCrossTurn.class);
    private static HashMap<List<Double>, Double> store = new HashMap<>();

    INDArray para; // parameters
    List<INDArray> p = new ArrayList<>();
    List<int[]> shapes = new ArrayList<>();
    double[] coef = {-0.3197059078415641, 0.26563979510130475, -2.6596709517335033, 1.9381693797762174, 0.6615979464703083, 0.326881228128546, 0.5013003217711116, -0.1417382458779886, -1.249110064884316, -0.9707419068057191, 1.1420193284160278, 1.705627785179035, 0.11629496534549937, 0.4700783113900231, -1.1433090241292334, -0.04072748377562259, -0.4186661001353569, 0.6896225769927529, -0.4370041703948341, 1.4415685213996325, 0.8059970417270993, -0.7272289162112864, 0.6352391078496592, -0.12919708315277464, 0.08729507226726044, -0.249213119941941, 0.8192411978142655, 1.0288685625802263, -0.932218232240942, -0.17827228887057417, 0.4374783534013372, -0.7096717215021402, -0.15388043088891434, -1.514053836829151, -0.09243295748298855, -1.5912837382075113, -1.0889320783256706, -0.7497296829936259, -1.0270214061299254, -1.9990209494969282, -0.3824219991234162, 1.3045504644698778, 2.2173034209086784, -0.8885718984444004, -0.48764256395524075, -2.2087635807644337, 0.6129313043218516, 2.404797533663557, -0.4964562375427281, 0.1481320197449692, -1.0936918894553047, 0.5718305832578523, -1.111287525319979, -2.836698746652661, -1.2115196792224003, 0.7380790729956219, 0.15615711360515935, -1.251125139816367, 0.864219439601834, -0.3196240104387729, 2.420573609987938, -0.8993221687117616, -0.8059514109255506, -0.7327889243951451, 2.9460776349874633, 1.2355743315583965, 0.27505488183592347, -1.0950708046039126, 0.911800892880857, -0.4288366493029427, 1.3884339331997009, 2.1900266565503914, -0.17654313783475634, -0.27674873680050194, -0.12714653121389805, -1.2662627624095197, -1.0761261964304734, 1.7158581590553794, 0.5591984173601761, -0.9174071137303029, 2.7297059308061455, 0.12566296423398887, 0.5025516966134067, 0.03148595175713782, 0.09772115759829057, -0.33374034291813126};
    private int maxBestActions = 6;
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

        // 基础版本
//        for (GameAction gameAction : validActions) {
//            double score = alphaBeta(context, player.getId(), gameAction, depth);  // 对每一个可能action，使用alphaBeta递归计算得分
//            if (score > bestScore) {
//                bestAction = gameAction;
//                bestScore = score;
//            }
//        }
        // 基础版本 End
        // 选取最优k个剪枝
        SortedMap<Double, GameAction> scoreActionMap = new TreeMap<>(Comparator.reverseOrder());
        for (GameAction gameAction : validActions) {  // 遍历validactions，使用Linear评估函数评估得到的局面，并按得分降序排列
            GameContext simulationResult = simulateAction(context.clone(), player.getId(), gameAction);  //假设执行gameAction，得到之后的game context
            double gameStateScore = evaluateContext(simulationResult, player.getId());	//heuristic.getScore(simulationResult, player.getId());     //heuristic评估执行gameAction之后的游戏局面的分数
            if(!scoreActionMap.containsKey(gameStateScore)){  // 注意：暂时简单的认为gameStateScore相同的两个simulationResult context一样，只保留第一个simulationResult对应的action
                scoreActionMap.put(gameStateScore, gameAction);
            }
            simulationResult.dispose();  //GameContext环境每次仿真完销毁
        }

        int k = 0;
        for(GameAction gameAction: scoreActionMap.values()){
            double score = alphaBeta(context, player.getId(), gameAction, depth);  // 对每一个可能action，使用alphaBeta递归计算得分
            if (score > bestScore) {
                bestAction = gameAction;
                bestScore = score;
            }
            k += 1;
            if(k >= maxBestActions){
                break;
            }
        }
        // 选取最优k个剪枝 End
        return bestAction;
    }

    // Greedy的评估函数
    double getScore(GameContext context, int playerId){


        Player player = context.getPlayer(playerId);
        Player opponent = context.getOpponent(player);
        if (player.getHero().isDestroyed()) {   // 己方被干掉，得分 负无穷
            return 0;
        }
        if (opponent.getHero().isDestroyed()) {  // 对方被干掉，得分 正无穷
            return 1;
        }

        float score = 0;

        List<Integer> envState = player.getPlayerState();
        envState.addAll(opponent.getPlayerState());

        for (int i = 0; i < envState.size(); i++){
            score += coef[i] * envState.get(i);
        }

        return score;
    }

    private double alphaBeta(GameContext context, int playerId, GameAction action, int depth) {
        GameContext simulation = context.clone();  // clone目前环境
        simulation.getLogic().performGameAction(playerId, action);  // 在拷贝环境中执行action
        if (depth == 0 || simulation.gameDecided()) {  // depth层递归结束、(发生玩家切换（我方这轮打完了）)或者比赛结果已定时，返回score
            return evaluateContext(simulation, playerId);
        }
        if (simulation.getActivePlayerId() != playerId){ // 发生玩家切换
            GameContext simulationOppo = simulation.clone();
            simulationOppo.startTurn(simulation.getActivePlayerId()); // 对手回合

            List<GameAction> validAct = simulationOppo.getValidActions();
            GameAction act = validAct.get(0);
            double bestScore = Double.NEGATIVE_INFINITY;
            // --------------------------很奇怪--------------------------------------------
            // IndexOutOfBoundsException: Index: 0, Size: 0 报错
//            do{
//                double bestScore = Double.NEGATIVE_INFINITY;
//
//                act = validAct.get(0); //在这行报错
//
//                for (GameAction gameAction : validAct){
//                    GameContext simulationResult = simulateAction(simulationOppo.clone(), simulationOppo.getActivePlayerId(), gameAction);  //假设执行gameAction，得到之后的game context
//                    double gameStateScore = getScore(simulationResult, simulationOppo.getActivePlayerId());	     //heuristic评估执行gameAction之后的游戏局面的分数
//                    if (gameStateScore > bestScore) {		// 记录得分最高的action
//                        bestScore = gameStateScore;
//                        act = gameAction;
//                    }
//                    simulationResult.dispose();  //GameContext环境每次仿真完销毁
//                }
//                simulationOppo.getLogic().performGameAction(simulationOppo.getActivePlayerId(), act);
//                if (act.getActionType() != ActionType.END_TURN)
//                    break;
//                if (simulationOppo.gameDecided()){
//                    return Float.NEGATIVE_INFINITY;
//                }
//                validAct = simulationOppo.getValidActions();
//            }while(true);
            // ----------------------------------------------------------------------
            for (GameAction gameAction : validAct){
                GameContext simulationResult = simulateAction(simulationOppo.clone(), simulationOppo.getActivePlayerId(), gameAction);  //假设执行gameAction，得到之后的game context
                double gameStateScore = getScore(simulationResult, simulationOppo.getActivePlayerId());	     //heuristic评估执行gameAction之后的游戏局面的分数
                if (gameStateScore > bestScore) {		// 记录得分最高的action
                    bestScore = gameStateScore;
                    act = gameAction;
                }
                simulationResult.dispose();  //GameContext环境每次仿真完销毁
            }
            while(act.getActionType() != ActionType.END_TURN){
                simulationOppo.getLogic().performGameAction(simulationOppo.getActivePlayerId(), act);
                if (simulationOppo.gameDecided()){
                    return Float.NEGATIVE_INFINITY;
                }
                validAct = simulationOppo.getValidActions();
//                act = validAct.get(new Random().nextInt(validAct.size())); // 对手随机选择
                // 对手Greedy选择
                act = validAct.get(0);
                bestScore = Double.NEGATIVE_INFINITY;
                for (GameAction gameAction : validAct){
                    GameContext simulationResult = simulateAction(simulationOppo.clone(), simulationOppo.getActivePlayerId(), gameAction);  //假设执行gameAction，得到之后的game context
                    double gameStateScore = getScore(simulationResult, simulationOppo.getActivePlayerId());	     //heuristic评估执行gameAction之后的游戏局面的分数
                    if (gameStateScore > bestScore) {		// 记录得分最高的action
                        bestScore = gameStateScore;
                        act = gameAction;
                    }
                    simulationResult.dispose();  //GameContext环境每次仿真完销毁
                }
                // 对手选择结束
            }
            simulationOppo.getLogic().performGameAction(simulationOppo.getActivePlayerId(), act);// 对手回合结束，执行End Turn

            double sum = 0;
            for (int i =0; i < 1; ++i){ // 搜索我方action
                bestScore = Double.NEGATIVE_INFINITY;
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
            return sum / 1.0;
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
        List<Double> envState = getFeature(player, opponent, simulation.getTurn());

        if (store.containsKey(envState)) // 如果哈希表中有特征向量，则直接从哈希表中获得
            return getStoredValue(envState);

        // 基础版本
//        for (GameAction gameAction : validActions) {
//            score = Math.max(score, alphaBeta(simulation, playerId, gameAction, depth - 1));  // 递归调用alphaBeta，取评分较大的
//            if (score >= 100000) {
//                break;
//            }
//        }
        // 基础版本 End
        // 选取最优k个剪枝
        SortedMap<Double, GameAction> scoreActionMap = new TreeMap<>(Comparator.reverseOrder());
        for (GameAction gameAction : validActions) {  // 遍历validactions，使用Linear评估函数评估得到的局面，并按得分降序排列
            GameContext simulationResult = simulateAction(simulation.clone(), playerId, gameAction);  //假设执行gameAction，得到之后的game context
            double gameStateScore = evaluateContext(simulationResult, playerId);  //heuristic.getScore(simulationResult, playerId);  //heuristic评估执行gameAction之后的游戏局面的分数
            if(!scoreActionMap.containsKey(gameStateScore)){  // 注意：暂时简单的认为gameStateScore相同的两个simulationResult context一样，只保留第一个simulationResult对应的action
                scoreActionMap.put(gameStateScore, gameAction);
            }
            simulationResult.dispose();  //GameContext环境每次仿真完销毁
        }

        int k = 0;
        for(GameAction gameAction: scoreActionMap.values()){
            score = Math.max(score, alphaBeta(simulation, playerId, gameAction, depth - 1));  // 递归调用alphaBeta，取评分较大的
            k += 1;
            if (score >= 100000 || k >= maxBestActions) {
                break;
            }
        }
        // 选取最优k个剪枝 End
        putStoredValue(envState, score);
        return score;
    }
    private GameContext simulateAction(GameContext simulation, int player, GameAction action) {
        simulation.getLogic().performGameAction(player, action);   // 在simulation GameContext中执行action，似乎是获取logic模块来执行action的
        return simulation;
    }

    private double getStoredValue(List<Double> envState){
        return store.get(envState); // 多线程模拟会报错
    }

    private void putStoredValue(List<Double> envState, double score){
        store.put(envState, score);
    }
}
