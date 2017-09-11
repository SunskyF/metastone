package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.Attribute;
import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.entities.heroes.Hero;
import net.demilich.metastone.game.entities.heroes.HeroClass;
import net.demilich.metastone.game.entities.minions.Minion;
import net.demilich.metastone.game.logic.GameLogic;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

// 尝试直接使用CEM (Noisy Cross Entropy Method) 方法优化Linear value function中的参数
// 相对于Linear CEM中优化一局结束时的HpDiff， Linear Batch CEM中以优化一个batch的对局的胜率为目标
// 也就是说一组参数下跑batch局，以这组参数和这batch局的胜率作为一个样本

public class LinearBatchCEM extends Behaviour {

	private final static Logger logger = LoggerFactory.getLogger(LinearBatchCEM.class);
	private Random random = new Random();
	private final static int stage = 2;
	private final static int feaSignle = 28;
	private final static int feaNum = stage * feaSignle;
	private static double[] parMean = new double[feaNum];
	private static double[] parVar = new double[feaNum];
	private double[] parWeight = new double[feaNum];
	private static ArrayList<double[]> paraList = new ArrayList<>();
	private static SortedMap<Integer, List<Integer>> rewardMap = new TreeMap<>(Comparator.reverseOrder());
	private static int gameCount = 0;
	private static int batchCount = 0;
	private static int batchWinCnt = 0;
	private static int iterNum = 0;
	private final static int batchSize = 50;
	private final static int updateBatchSize = 20;
	private final static double topRatio = 0.25;
	private int nowTurn = 0;

	// 初始化par均值
//	double[] coef0 = {-0.042435191603752365, 0.40616696103829747, -0.6076077508888231, 0.7272720043082649, 0.20202105394258138, 0.6313194790542382, 0.018128759542628426, -0.6714877445764238, -1.4053075565764823, -0.5645665894497521, 0.7847370830575832, 0.6027815061309918, 0.11944932690773696, 0.16639617889671887, -0.596884142191235, -1.303442871303662, -0.7689279203049311, 0.08684843915912507, 0.2194776555521321, 0.3016746685481717, 1.2937229089380362, -0.3867675424274616, 0.1076755527545435, -0.2732232995785765, -1.2185303120526827, 0.44953477632896055, 0.4381595843767112, -0.42740268057811853, -0.7593509953652382, -0.25454178333555133, -0.06777467174623754, -0.29690084326922533, 0.139006279981216, -0.19757466435610388, -0.10281368164119024, -1.0318842806816224, 0.6697091092076944, -1.5504887549148079, -0.46513049548185065, -0.224168675085325, 0.3363897344845124, 1.0077572813673379, 1.790712443198513, -0.7974532011373017, -0.034379835513210874, -0.53728431299419, 0.27911211300832656, 1.77296275442259, -0.49676110743592145, 0.1543295718827667, -0.8465960481972349, 0.7616942260912617, -0.09467554276581093, -2.0242808280175053, -0.42044884895501194, 0.22098060325157623, -0.7124548934850738, -0.46505479523846677, 0.48943013064063173, -1.0814450906779396, 1.494202436696506, -0.2253465154462635, -0.45375551865466784, 0.23376970921025383, -0.31330715778322343, -0.36119276257174077, -0.643335795514266, -9.750616593995213E-4, 0.38910933337215503, 0.09884268820871578, 0.37335447500509883, 0.8047764493080031, -0.4720077472112548, 0.264490464098162, 0.4104466046033848, -1.3686731855897478, -0.3840621668768126, 1.0668114179340098, 0.2664341808467372, -0.9317865799520276, 0.7754792723230315, 0.23662829519630627, -0.4460817116862948, -0.010734604936549205, 0.08007887984653188, -0.7490589136711456, 0.32186845493246313, 1.4714419204754288};
	double[] coef0 = {0.08705327830248306, -0.028473563039054184, -1.6699125490135103, 2.9011900823998165, 0.8647260485505271,
			0.7221744400149696, -0.30756153183604185, 1.1021736417978183, -0.8879105036608483, 1.0335600542736803, 1.3044616971327883,
			0.9101027869544048, 0.29860997457933763, -0.3657844932384716, -0.5466927677624636, 0.711730691451666, -0.7361602428437026,
			1.1818377110431368, 1.0704007929412764, -0.2064146454138593, 1.9878480155773197, -0.8253328675428739, 1.1834213985741755,
			-0.3760740401509485, -0.33734255974364963, -0.9240615103962506, 0.9203271831613986, -0.443300771273455, -0.40074564266083584,
			-0.45672345146959076, 0.3808070861110021, -0.4090303175940698, -0.19206389806072244, 0.3255206609301137, 1.405210659961705,
			-2.6873856123383613, -1.023422665515519, -3.0383722774792075, 0.32875601500723844, -0.619512015411133, -0.34070661622088355,
			0.8522964036093238, 3.2099373464187844, -1.6381360109233647, -0.6531607670830637, -0.2054973577247289, 1.5457313376551602, 1.6648618579791086,
			-0.42440107525812865, -0.763084983554154, -0.5981607269354156, 1.3251487555327612, -1.938737987160963, -3.1216973637091576,
			-0.9462451444497259, -0.19266968278073918, -1.175385017717137, -0.20191081615537404, 0.9528575558896453, -1.6122103991632808, 2.0802897709894017,
			-0.8500308871161788, -2.6485908816419963, 0.6850375931229098, -0.007820691502555849, 0.4742530909080602, -1.096332103285889, -1.7851375058268617,
			0.21067765165006042, -0.18961021567895225, 0.27990602338417636, 0.10684188821057261, 0.8237212146173462,
			0.5051825720702025, 1.0394910094595908, -1.5909722243464166, 1.0790644610395934, 1.0371159605479374, 1.8617525620206137,
			-1.6610380467389887, 1.2930522217664526, -0.13163915919616986, -1.3977396292212636, 0.04488550408763463,
			-0.8906171654108572, 0.6581315471904368, 2.351949347693711, 1.3359697270037687};
	// iterNum=16, bestPara 50/50 vs GreedyBestMove (使用 Wild pirate warrior卡组进行训练，针对默认GreedyBestMove，Greedy 模型 ##########################)
//	double[] coef0 = {-0.292840715614615, 0.20101831923661426, -3.39143378499341, 1.9875259555880256, 0.38243094977445563, 0.19916580778656684, 0.39681845708944546, -0.1516207831858801, -1.3270806070142345, -0.8516608373607158, 1.1691499563044714, 1.7037675238382644, 0.26375688433025735, 0.6041754222325952, -1.2228269800621305, -0.04841372741594877, -0.42730972024890057, 0.5378274791604805, -1.0610676568848123, 1.5419236540282981, 0.8025786326886725, -0.5998843281736796, 0.5881743466572328, -0.04764287832761427, 0.2743675479019204, -0.2555714125323206, 0.6015742978414451, 0.959379920060044, -1.0280270764615334, -0.24687565888405189, 0.41578079075524416, -0.9403436227526668, -0.7232215629677738, -1.4544950831523629, -0.07337021103468366, -1.7957986080880892, -1.0615903113568188, -0.9677261770565323, -0.9900926846614896, -1.9467941633472445, -0.38968317406040387, 1.18108807301559, 2.2933042007465785, -0.7707815457180169, -0.5453742461725655, -2.1452134837607604, 0.4673601618824798, 2.3374173897171233, -0.532213458145435, 0.3020731393119615, -1.0703121445039732, 0.5722946167284809, -1.12636070861931, -2.905239343653205, -1.2570275089835006, 0.5180962966398064, 0.1545759621733422, -1.1689272645244264, 0.8409561108467435, -0.2651722876271211, 2.361775722446801, -0.8082739582714793, -0.7465872641805845, -0.9966536644382684, 2.912971532256444, 1.3543455372968578, 0.30616915666047656, -0.9915720231492012, 0.9276192616927182, -0.4280058490330962, 1.4060304457970378, 2.1048133236115314, -0.055023043380736214, -0.14489618269988702, 0.10741416798829515, -1.4163208108087157, -1.1061911758530445, 2.153649561244002, 0.7136525008272323, -0.9008286327709033, 2.464534971036488, 0.13531159446449426, 0.40349749765912607, -5.283467693679954E-4, 0.10380815509947951, -0.3747210306190661, 0.1738684930251086, 1.533762114780188};

	public LinearBatchCEM() {
		for(int i=0; i<feaNum; i++){
			parMean[i] = coef0[i]; //2*random.nextDouble() - 1;
			parVar[i] = 0.25;
		}
		updateParWeight();
	}

	@Override
	public String getName() {
		return "CEM-Batch-Linear";
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
		if (context.getTurn() < 10)
			nowTurn = 0;
		else
			nowTurn = 1;

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

	private static int calculateThreatLevel(GameContext context, int playerId) {
		int damageOnBoard = 0;
		Player player = context.getPlayer(playerId);
		Player opponent = context.getOpponent(player);
		for (Minion minion : opponent.getMinions()) {
			damageOnBoard += minion.getAttack(); // * minion.getAttributeValue(Attribute.NUMBER_OF_ATTACKS);
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
		// 威胁等级标识特征
//		int threatLevelHigh= 0;
//		int threatLevelMiddle = 0;
//		int threatLevel = calculateThreatLevel(context, playerId);
//		if(threatLevel == 2){
//			threatLevelHigh = 1;
//		}else if(threatLevel == 1){
//			threatLevelMiddle = 1;
//		}
//		envState.add(threatLevelHigh);
//		envState.add(threatLevelMiddle);
//		logger.info("Feature Number: {}", envState.size());
//		logger.info("{} {} {} {}", feaSignle, feaNum, envState.size(), parWeight.length);
		double score = 0;
		for (int i = nowTurn * feaSignle; i < (nowTurn + 1) * feaSignle; i++){
			score += parWeight[i]*envState.get(i - nowTurn * feaSignle);
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
		gameCount++;
		if(playerId == winningPlayerId){
			batchWinCnt += 1;
		}

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
