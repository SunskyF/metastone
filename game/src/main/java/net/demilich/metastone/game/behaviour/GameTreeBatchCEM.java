package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.Attribute;
import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.ActionType;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.entities.heroes.Hero;
import net.demilich.metastone.game.entities.heroes.HeroClass;
import net.demilich.metastone.game.entities.minions.Minion;
import net.demilich.metastone.game.logic.GameLogic;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

// 尝试直接使用CEM (Noisy Cross Entropy Method) 方法优化Linear value function中的参数, 但现在会往前多步后评估局面，而不是GreedyBest

public class GameTreeBatchCEM extends Behaviour {

	private final static Logger logger = LoggerFactory.getLogger(GameTreeBatchCEM.class);
	private Random random = new Random();
	private final static int stage = 2;
	private final static int feaNum = 88; // 88
	private final static int totalFea = feaNum * stage;
	private static double[] parMean = new double[totalFea];
	private static double[] parVar = new double[totalFea];
	private double[] parWeight = new double[totalFea];
	private static ArrayList<double[]> paraList = new ArrayList<>();
	private static SortedMap<Integer, List<Integer>> rewardMap = new TreeMap<>(Comparator.reverseOrder());
	private static int gameCount = 0;
	private static int batchCount = 0;
	private static int batchWinCnt = 0;
	private static int iterNum = 0;
	private final static int batchSize = 50;
	private final static int updateBatchSize = 20;
	private final static double topRatio = 0.25;
	private int nowStage = 0;

	// 初始化par均值
	// 30
//	double[] coef0 = {-0.4026557850528921, -1.4777779620148956, 0.4807519990112861, 0.17256270698849252, -3.0048219754608025, 1.2471351008867102, 0.0984930660725887, 0.10428128567415333, 2.305958663224414, 1.1694585122714134, 1.6161794705279806, -1.2488627310061495, 0.7849320841015818, -0.44585583054203787, -0.5739923937489321, -0.15309219515362008, 0.10443397417923186, -1.6386810620082035, -1.5720069467333566, 1.142923451103324, -1.8273642125940401, 3.9607177425623874, -1.5590593357541482, -1.8584667661464103, -0.7176115097989136, -1.9578732692028225, -1.555219474560269, 1.260854506957219, -2.110174979376178, 0.6617734109303619};
	// 88
	double[] coef0 = {-0.042435191603752365, 0.40616696103829747, -0.6076077508888231, 0.7272720043082649, 0.20202105394258138, 0.6313194790542382, 0.018128759542628426, -0.6714877445764238, -1.4053075565764823, -0.5645665894497521, 0.7847370830575832, 0.6027815061309918, 0.11944932690773696, 0.16639617889671887, -0.596884142191235, -1.303442871303662, -0.7689279203049311, 0.08684843915912507, 0.2194776555521321, 0.3016746685481717, 1.2937229089380362, -0.3867675424274616, 0.1076755527545435, -0.2732232995785765, -1.2185303120526827, 0.44953477632896055, 0.4381595843767112, -0.42740268057811853, -0.7593509953652382, -0.25454178333555133, -0.06777467174623754, -0.29690084326922533, 0.139006279981216, -0.19757466435610388, -0.10281368164119024, -1.0318842806816224, 0.6697091092076944, -1.5504887549148079, -0.46513049548185065, -0.224168675085325, 0.3363897344845124, 1.0077572813673379, 1.790712443198513, -0.7974532011373017, -0.034379835513210874, -0.53728431299419, 0.27911211300832656, 1.77296275442259, -0.49676110743592145, 0.1543295718827667, -0.8465960481972349, 0.7616942260912617, -0.09467554276581093, -2.0242808280175053, -0.42044884895501194, 0.22098060325157623, -0.7124548934850738, -0.46505479523846677, 0.48943013064063173, -1.0814450906779396, 1.494202436696506, -0.2253465154462635, -0.45375551865466784, 0.23376970921025383, -0.31330715778322343, -0.36119276257174077, -0.643335795514266, -9.750616593995213E-4, 0.38910933337215503, 0.09884268820871578, 0.37335447500509883, 0.8047764493080031, -0.4720077472112548, 0.264490464098162, 0.4104466046033848, -1.3686731855897478, -0.3840621668768126, 1.0668114179340098, 0.2664341808467372, -0.9317865799520276, 0.7754792723230315, 0.23662829519630627, -0.4460817116862948, -0.010734604936549205, 0.08007887984653188, -0.7490589136711456, 0.32186845493246313, 1.4714419204754288};

	public GameTreeBatchCEM() {
		logger.info("{}", coef0.length);
		for (int s = 0; s < stage; ++s){
			for(int i=0; i<feaNum; i++){
				parMean[s * feaNum + i] = coef0[i]; //2*random.nextDouble() - 1;
				parVar[s * feaNum + i] = 0.25;
			}
		}

		updateParWeight();
	}

	@Override
	public String getName() {
		return "CEM-Batch-Tree";
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
		if (context.getTurn() < 10)
			nowStage = 0;
		else
			nowStage = 1;

		double bestScore = Double.NEGATIVE_INFINITY;

		for (GameAction gameAction : validActions) {
			double score = alphaBeta(context, player.getId(), gameAction, depth);  // 对每一个可能action，使用alphaBeta递归计算得分
			if (score > bestScore) {
				bestAction = gameAction;
				bestScore = score;
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

		double score = Float.NEGATIVE_INFINITY;

		for (GameAction gameAction : validActions) {
			score = Math.max(score, alphaBeta(simulation, playerId, gameAction, depth - 1));  // 递归调用alphaBeta，取评分较大的
			if (score >= 100000) {
				break;
			}
		}
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
//		List<Integer> envState = player.getPlayerStatefh1(false);
//		envState.addAll(opponent.getPlayerStatefh1(false));
		List<Integer> envState = player.getPlayerState();
		envState.addAll(opponent.getPlayerState());

//		logger.info("Feature Num: {}", envState.size());
		// 威胁等级标识特征
		int threatLevelHigh= 0;
		int threatLevelMiddle = 0;
		int threatLevel = calculateThreatLevel(context, playerId);
		if(threatLevel == 2){
			threatLevelHigh = 1;
		}else if(threatLevel == 1){
			threatLevelMiddle = 1;
		}
		envState.add(threatLevelHigh);
		envState.add(threatLevelMiddle);

		double score = 0;
		int startI = this.nowStage * this.feaNum;
		for (int i = startI; i < (this.nowStage + 1)* this.feaNum; i++){
//			logger.info("{} {} {} {}", i, this.feaNum, this.totalFea, envState.size());
			score += parWeight[i]*envState.get(i - startI);
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
		double[][] topPara = new double[totalFea][topNum];
		double[] bestPara = new double[totalFea];
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
					for(int i=feaNum; i<totalFea; i++){
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

		// 一个Batch结束
		if(gameCount == batchSize){
			logger.info("batchCount: {}, batchWinCnt: {} / {}", batchCount, batchWinCnt, batchSize);
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
