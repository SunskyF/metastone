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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

// 尝试直接使用CEM (Noisy Cross Entropy Method) 方法优化Linear value function中的参数, 但现在会往前多步后评估局面，而不是GreedyBest

public class GameTreePruneBatchCEM extends Behaviour {

	private final static Logger logger = LoggerFactory.getLogger(GameTreePruneBatchCEM.class);
	private Random random = new Random();
	private final static int feaNum = 120;//88
	private static double[] parMean = new double[feaNum];
	private static double[] parVar = new double[feaNum];
	private double[] parWeight = new double[feaNum];
	private static ArrayList<double[]> paraList = new ArrayList<>();
	private static SortedMap<Integer, List<Integer>> rewardMap = new TreeMap<>(Comparator.reverseOrder());
	private static int gameCount = 0;
	private static int batchCount = 0;
	private static int batchWinCnt = 0;
	private static int iterNum = 0;
	private final static int batchSize = 40;
	private final static int updateBatchSize = 20;
	private final static double topRatio = 0.25;
	private final static int maxNumMoveSearched = 6;
	private final IGameStateHeuristic heuristic;
	private static HashMap<List<Integer>, Double> store = new HashMap<>();

	// 初始化par均值
	double[] coef0 = {0.760326412267397, 1.0589083919017865, -0.9400597870303837, 1.8744045092345258, 0.40236541366111334, 0.7418310137683413, 0.484131665708331, -0.7552493510082674, -0.49584676055695726, -0.16698138431239073, 0.36754468394420575, 1.4864556790810401, 1.1277694490216241, -0.19385547308776607, -1.0543152353607872, -0.2839165182591483, -1.4812832356356582, 0.3152121620740715, 0.0576844177240588, 1.1548436693125355, 1.9800036654926079, -0.3901963327635122, -0.7253225619942502, 0.18436967723674624, -0.6734730405335783, 0.5670641396548124, 0.639198399863008, -2.323129891333293, -1.0595513754902268, -0.804768708781494, 0.049672910287548194, 0.45954962674589844, -1.2431053666129253, 1.453612258307805, 0.26218920504452126, -2.0395371675289904, 0.18812695989268322, -1.328556982125309, -0.09499834083064819, -1.3747489091708807, 0.4488316541321531, 1.557497817592364, 1.0913026067029423, -0.597424707768485, -0.8361789076762033, -0.30777376598675443, -0.12238371097680156, 1.5841665240946965, -1.6267115776236394, -0.8061619411443898, -0.6975022521655243, 0.8245686938375283, -0.1718025408001051, -1.965065224429499, -1.59218011905894, 0.5600873334680981, -0.1825971189270253, -0.584501980650917, 2.416352484932763, -0.994680624848772, 1.9001580351986664, 0.11200461958090441, -1.149550315376586, 0.08082039517078825, -0.16580058883653237, -1.571984435874137, -2.4570940338800336, -0.15228687237889565, -0.1436000925996575, -0.06952984168734147, 0.8184547686840771, 0.6615721068181535, -0.2850074919127527, 0.16685275047289128, -0.01777378273904684, -1.705839086129454, -0.22775080851185345, 1.740876345330731, 0.2765548960828773, -0.3556647399330023, 0.4369754630045146, 0.679533140719245, 0.7003181586567979, 0.3695026344221172, -0.45705065513923954, -0.5747965912920181, 1.4643969249437774, 3.3167090234141243};

	public GameTreePruneBatchCEM(IGameStateHeuristic heuristic) {
		this.heuristic = heuristic;
		for(int i=0; i<feaNum; i++){
			parMean[i] = 2*random.nextDouble() - 1; //2*random.nextDouble() - 1;
			parVar[i] = 0.25;
		}
		updateParWeight();
	}

	@Override
	public String getName() {
		return "CEM-Batch-Tree-Prune";
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

	private GameContext simulateAction(GameContext simulation, int playerId, GameAction action) {
		simulation.getLogic().performGameAction(playerId, action);   // 在simulation GameContext中执行action，似乎是获取logic模块来执行action的
		return simulation;
	}

	@Override
	public GameAction requestAction(GameContext context, Player player, List<GameAction> validActions) {

		if (validActions.size() == 1) {  //只剩一个action一般是 END_TURN
			return validActions.get(0);
		}

		int depth = 8;
		// when evaluating battlecry and discover actions, only optimize the immediate value （两种特殊的action）
		if (validActions.get(0).getActionType() == ActionType.BATTLECRY) {
			depth = 0;
		} else if (validActions.get(0).getActionType() == ActionType.DISCOVER) {  // battlecry and discover actions一定会在第一个么？
			return validActions.get(0);
		}

		store.clear();//每次都是只对一次搜索

		SortedMap<Double, GameAction> scoreActionMap = new TreeMap<>(Comparator.reverseOrder());
		for (GameAction gameAction : validActions) {  // 遍历validactions，使用Linear评估函数评估得到的局面，并按得分降序排列
			GameContext simulationResult = simulateAction(context.clone(), player.getId(), gameAction);  //假设执行gameAction，得到之后的game context
			double gameStateScore = evaluateContext(simulationResult, player.getId()); //heuristic.getScore(simulationResult, player.getId());	     //heuristic评估执行gameAction之后的游戏局面的分数
			if(!scoreActionMap.containsKey(gameStateScore)){  // 注意：暂时简单的认为gameStateScore相同的两个simulationResult context一样，只保留第一个simulationResult对应的action
				scoreActionMap.put(gameStateScore, gameAction);
			}
			simulationResult.dispose();  //GameContext环境每次仿真完销毁
		}

		GameAction bestAction = validActions.get(0);
		double bestScore = Double.NEGATIVE_INFINITY;
		int k = 0;
		for(GameAction gameAction: scoreActionMap.values()){
			double score = alphaBeta(context, player.getId(), gameAction, depth);  // 对每一个可能action，使用alphaBeta递归计算得分
			if (score > bestScore) {
				bestAction = gameAction;
				bestScore = score;
			}
			k += 1;
			if(k >= maxNumMoveSearched){
				break;
			}
		}
		return bestAction;
	}

	private double alphaBeta(GameContext context, int playerId, GameAction action, int depth) {
		GameContext simulation = context.clone();  // clone目前环境
		simulation.getLogic().performGameAction(playerId, action);  // 在拷贝环境中执行action
		if (depth == 0 || simulation.getActivePlayerId() != playerId || simulation.gameDecided()) {  // depth层递归结束、发生玩家切换（我方这轮打完了）或者比赛结果已定时，返回score
			return evaluateContext(simulation, playerId);
		}

		List<GameAction> validActions = simulation.getValidActions();  //执行完一个action之后，获取接下来可以执行的action

		SortedMap<Double, GameAction> scoreActionMap = new TreeMap<>(Comparator.reverseOrder());
		for (GameAction gameAction : validActions) {  // 遍历validactions，使用Linear评估函数评估得到的局面，并按得分降序排列
			GameContext simulationResult = simulateAction(simulation.clone(), playerId, gameAction);  //假设执行gameAction，得到之后的game context
			double gameStateScore = evaluateContext(simulationResult, playerId); //heuristic.getScore(simulationResult, playerId);	     //heuristic评估执行gameAction之后的游戏局面的分数
			if(!scoreActionMap.containsKey(gameStateScore)){  // 注意：暂时简单的认为gameStateScore相同的两个simulationResult context一样，只保留第一个simulationResult对应的action
				scoreActionMap.put(gameStateScore, gameAction);
			}
			simulationResult.dispose();  //GameContext环境每次仿真完销毁
		}

		Player player = simulation.getPlayer(playerId);
		Player opponent = simulation.getOpponent(player);

		List<Integer> envState = player.getPlayerStatefh1(false);
		envState.addAll(opponent.getPlayerStatefh1(false));

		if (store.containsKey(envState)){
			return store.get(envState);
		}

		double score = Float.NEGATIVE_INFINITY;
		int k = 0;
		for(GameAction gameAction: scoreActionMap.values()){
			score = Math.max(score, alphaBeta(simulation, playerId, gameAction, depth - 1));  // 递归调用alphaBeta，取评分较大的
			k += 1;
			if (score >= 100000 || k >= maxNumMoveSearched) {
				break;
			}
		}
		store.put(envState, score);

		return score;
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

	private double evaluateContext(GameContext context, int playerId) {
		Player player = context.getPlayer(playerId);
		Player opponent = context.getOpponent(player);
		if (player.getHero().isDestroyed()) {   // 己方被干掉，得分 负无穷
			return Float.NEGATIVE_INFINITY;  // 正负无穷会影响envState的解析，如果要加的话可以改成 +-100之类的
		}
		if (opponent.getHero().isDestroyed()) {  // 对方被干掉，得分 正无穷
			return Float.POSITIVE_INFINITY;
		}
		List<Integer> envState = player.getPlayerStatefh1(false);
		envState.addAll(opponent.getPlayerStatefh1(false));

//		logger.info("Feature Number: {}", envState.size());

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
		int topNum = (int)(updateBatchSize*topRatio);
		double[][] topPara = new double[feaNum][topNum];
		double[] bestPara = new double[feaNum];
		double meanTopReward = 0;
		// 取出reward最好的若干次的参数
		for(Integer reward: rewardMap.keySet()){
			for(Integer ind: rewardMap.get(reward)){
				double[] para = paraList.get(ind);
				for(int i=0; i<para.length; i++){
					topPara[i][k] = para[i];
				}
				k++;
				if(k == 1){
					bestPara = para.clone();
				}
				meanTopReward += reward;
				if(k >= topNum){
					meanTopReward /= topNum;
					logger.info("################# iterNum: {}, meanTopReward: {}, bestPara: {} ##################", iterNum, meanTopReward, bestPara);
					// 更新均值和方差
					for(int i=0; i<feaNum; i++){
						this.parMean[i] = calcMean(topPara[i]);
						this.parVar[i] = calcVar(topPara[i]) + Math.max(0.1 - 0.01*iterNum, 0);  // 添加逐渐减小的Noise， 可调整
					}
					logger.info("########## rewardMap: {}, parMean: {}, parVar: {}", rewardMap, parMean, parVar);
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
		gameCount++;
		if(playerId == winningPlayerId){
			batchWinCnt += 1;
		}
//		logger.info("Win: {}", winningPlayerId);
		// 一个Batch结束
		if(gameCount == batchSize){
			logger.info("batchCount: {}, batchWinCnt: {}", batchCount, batchWinCnt);
			int reward = batchWinCnt;
			// 保存这一Batch的最终batchWinCnt和对应batchCount编号（从0开始编号）
			if(rewardMap.containsKey(reward)){
				rewardMap.get(reward).add(batchCount);
			}else{
				rewardMap.put(reward, new ArrayList<>(Arrays.asList(batchCount)));
			}
			// 保存这一Batch使用的模型参数
			paraList.add(parWeight.clone());
			batchCount++;
			gameCount = 0;
			batchWinCnt = 0;
			// 根据均值和方差，随机生成下一Batch使用的权重参数
			updateParWeight();
		}

		// 执行一个updateBatchSize之后, 更新参数均值和方差
		if(batchCount == updateBatchSize){
			iterNum++;
//			logger.info("rewardMap: {}, para: {}, parMean: {}, parVar: {}", rewardMap, parWeight, parMean, parVar);
			// 更新参数均值和方差, 并清空这一个batch的数据
			updateMeanVar();
			batchCount = 0;
//			logger.info("########## rewardMap: {}, para: {}, parMean: {}, parVar: {}", rewardMap, parWeight, parMean, parVar);
		}
	}
}
