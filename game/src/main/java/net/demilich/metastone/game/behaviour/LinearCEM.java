package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.logic.GameLogic;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import java.io.*;
import java.util.*;

// 尝试直接使用CEM (Noisy Cross Entropy Method) 方法优化Linear value function中的参数

public class LinearCEM extends Behaviour {

	private final static Logger logger = LoggerFactory.getLogger(LinearCEM.class);
	private Random random = new Random();
	private final static int feaNum = 28;
	private static double[] parMean = new double[feaNum];
	private static double[] parVar = new double[feaNum];
	private double[] parWeight = new double[feaNum];
	private static ArrayList<double[]> paraList = new ArrayList<>();
	private static SortedMap<Integer, List<Integer>> rewardMap = new TreeMap<>(Comparator.reverseOrder());
	private static int gameCount = 0;
	private static int iterNum = 0;
	private final static int batchSize = 50;
	private final static double topRatio = 0.2;

	// 初始化par均值
	double[] coef0 = {0.055,-0.030,-0.196,0.064,-0.093,0.133,0.046,-0.243,0.186,0.037,-0.021,0.083,0.000,-0.257,0.000,-0.065,0.032,0.138,-0.035,0.172,-0.136,-0.065,0.250,-0.162,-0.032,-0.020,-0.069,0.000,0.390,0.000};

	public LinearCEM() {
		for(int i=0; i<feaNum; i++){
			parMean[i] = coef0[i]; //2*random.nextDouble() - 1;
			parVar[i] = 0.01;
		}
		updateParWeight();
	}

	@Override
	public String getName() {
		return "CEM-Linear";
	}

	@Override
	public List<Card> mulligan(GameContext context, Player player, List<Card> cards) {
		List<Card> discardedCards = new ArrayList<Card>();
		for (Card card : cards) {
			if (card.getBaseManaCost() >= 4) {  //耗法值>=4的不要
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

		// get best action at the current state and the corresponding Q-score
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

	private GameContext simulateAction(GameContext simulation, Player player, GameAction action) {
		GameLogic.logger.debug("");
		GameLogic.logger.debug("********SIMULATION starts********** " + simulation.getLogic().hashCode());
		simulation.getLogic().performGameAction(player.getId(), action);   // 在simulation GameContext中执行action，似乎是获取logic模块来执行action的
		GameLogic.logger.debug("********SIMULATION ends**********");
		GameLogic.logger.debug("");
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
		envState.addAll(opponent.getPlayerStatefh0(false));
//		logger.info("Par Weight: {}, envState: {}", parWeight.length, envState.size());
		double score = 0;
		for (int i = 0; i < parWeight.length; i++){
			score += parWeight[i]*envState.get(i);
		}
		return score;
	}

	private void updateParWeight(){
		// 根据参数的均值和方差，按正态分布生成parWeight
		for(int i=0; i<parWeight.length; i++){
			parWeight[i] = parMean[i] + Math.sqrt(parVar[i])*random.nextGaussian();
		}
	}

	private double calcMean(double[] paras){
		double mean = 0;
		for(double para : paras){
			mean += para;
		}
		mean /= paras.length;
		return mean;
	}

	private double calcVar(double[] paras){
		double mean = calcMean(paras);
		double var = 0;
		for(double para : paras){
			var += (para - mean)*(para - mean);
		}
		var /= paras.length;
		return var;
	}

	private void updateMeanVar(){
		int k = 0;
		int topNum = (int)(batchSize*topRatio);
		double[][] topPara = new double[feaNum][topNum];
		double meanTopReward = 0;
		// 取出reward最好的若干次的参数
		for(Integer reward: rewardMap.keySet()){
			for(Integer ind: rewardMap.get(reward)){
				double[] para = paraList.get(ind);
				for(int i=0; i<para.length; i++){
					topPara[i][k] = para[i];
				}
				k++;
				meanTopReward += reward;
				if(k >= topNum){
					meanTopReward /= topNum;
					logger.info("################# iterNum: {}, meanTopReward: {} ##################", iterNum, meanTopReward);
					// 更新均值和方差
					for(int i=0; i<feaNum; i++){
						this.parMean[i] = calcMean(topPara[i]);
						this.parVar[i] = calcVar(topPara[i]) + Math.max(0.01 - 0.0001*iterNum, 0);  // 添加逐渐减小的Noise， 可调整
					}
					// 清空这一个batch的数据
					paraList.clear();
					rewardMap.clear();
					return;
				}
			}
		}
	}

	@Override
	public void onGameOver(GameContext context, int playerId, int winningPlayerId) {
		// GameOver的时候会跳入这个函数
		Player player = context.getPlayer(playerId);
		Player opponent = context.getOpponent(player);
//		List<Integer> playerState = player.getPlayerState();
//		List<Integer> opponentState = opponent.getPlayerState();
		int reward = player.getHero().getHp() - opponent.getHero().getHp();

		// 保存这一局的最终reward和对应gameCount编号（从0开始编号）
		if(rewardMap.containsKey(reward)){
			rewardMap.get(reward).add(gameCount);
		}else{
			rewardMap.put(reward, new ArrayList<>(Arrays.asList(gameCount)));
		}
		// 保存这一局使用的模型参数
		paraList.add(parWeight.clone());

		gameCount++;
		logger.info("gameCount: {}, winner: {}, HpDiff: {}", gameCount, winningPlayerId, reward);  // 可以根据HpDiff来判断胜负和设定reward
//		logger.info("rewardMap: {}, para: {}, parMean: {}, parVar: {}", rewardMap, parWeight, parMean, parVar);
		// 执行一个batchSize之后
		if(gameCount%batchSize == 0){
			iterNum++;
			logger.info("rewardMap: {}, para: {}, parMean: {}, parVar: {}", rewardMap, parWeight, parMean, parVar);
			// 更新参数均值和方差, 并清空这一个batch的数据
			updateMeanVar();
			gameCount = 0;
			logger.info("########## rewardMap: {}, para: {}, parMean: {}, parVar: {}", rewardMap, parWeight, parMean, parVar);
		}
		// 根据均值和方差，随机生成下一局使用的权重参数
		updateParWeight();
	}
}
