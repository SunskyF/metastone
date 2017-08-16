package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.behaviour.heuristic.IGameStateHeuristic;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.logic.GameLogic;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

import org.dmg.pmml.PMML;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.model.PMMLUtil;

// 尝试使用Experience Replay方法训练一个Linear的Q-function
// 根据bellman最优方程，一般DQN类训练的loss函数： L(w) = [r_t + gamma*max_a{Q(s_t+1, a, w)} - Q(s_t, a_t, w)]^2
// 将执行action之后到达的局面s_t+1的描述向量作为 (s_t, a_t) 的描述, 训练目标变成 [r_t + gamma*max_a{Q(s_t+1, a, w)} - Q(s_t+1, w)]^2
// 暂时考虑将我方的每一次action算作一个time step，我方EndTurn动作和对方完整执行以及下一个turn己方发牌合在一起算作一个time step （将对方当做环境），需要记录每一次操作的数据
// 后面也可以尝试将我方一个turn内的完整action path当做一个action，以一个完整回合作为一个time step，似乎更合理，因为我们更希望评估的是一系列操作之后的状态优劣 （计算argmax_a{Q(s_t+1,a)}会比较复杂）
public class ReplayQLearningLinear extends Behaviour {

	private final static Logger logger = LoggerFactory.getLogger(ReplayQLearningLinear.class);
	private Random random = new Random();
	private double epsilon = 0.1;
	private int turn = 0;
	private final IGameStateHeuristic heuristic;
	private final static String pmmlFile = "E:\\workspace\\MetaStone\\app\\mlp.pmml";
	private static Evaluator modelEvaluator;
	private long[] playersWinCount = {0,0};
	private static double runningReward = 0;

	public ReplayQLearningLinear(IGameStateHeuristic heuristic) {
		this.heuristic = heuristic;
		File file = new File(pmmlFile);
		//如果PMML文件存在，直接加载已有模型
		if (file.exists()) {
			try {
				loadPMMLModel(pmmlFile);
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else{
			this.modelEvaluator = null;
		}
	}

	@Override
	public String getName() {
		return "Q-Linear";
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
		// Sjx, output info
		//logger.info("num of valid actions {}, they are {}.", validActions.size(), validActions);

//		if (validActions.size() == 1) {  //只剩一个action一般是 END_TURN
//			return validActions.get(0);
//		}

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
		// 记录需要的数据
		Player opponent = context.getOpponent(player);
		List<Integer> envState = player.getPlayerState();
		envState.addAll(opponent.getPlayerState());
		logger.info("{'Turn': {}, 'envState': {}, 'bestScore': {}}", this.turn, envState.toString(), bestScore);
		this.turn += 1;

		// get epsilon greedy action
		if (random.nextDouble() < epsilon) {  //返回随机action
			int randomIndex = random.nextInt(validActions.size());
			GameAction randomAction = validActions.get(randomIndex);
			return randomAction;
		} else {
//			logger.info("bestAction:{}", bestAction);
			return bestAction;  // 返回argmax_a{Q(s_t,a)
		}
	}

	private GameContext simulateAction(GameContext simulation, Player player, GameAction action) {
		GameLogic.logger.debug("");
		GameLogic.logger.debug("********SIMULATION starts********** " + simulation.getLogic().hashCode());
		simulation.getLogic().performGameAction(player.getId(), action);   // 在simulation GameContext中执行action，似乎是获取logic模块来执行action的
		GameLogic.logger.debug("********SIMULATION ends**********");
		GameLogic.logger.debug("");
		return simulation;
	}

	private void loadPMMLModel(String file) throws Exception {
		try(InputStream model = new FileInputStream(new File(file))){
			PMML pmml = PMMLUtil.unmarshal(model);
			ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
			this.modelEvaluator = modelEvaluatorFactory.newModelEvaluator(pmml);
//			logger.info("PMML evaluator updated!");
//			logger.info("PMML is null: {}", this.modelEvaluator==null);
//			logger.info("PMML targetFields {}", this.modelEvaluator.getTargetFields());
		} catch(Exception e) {
			logger.error(e.toString());
			throw e;
		}
	}

	private double evaluateContext(GameContext context, int playerId) {
		if (this.modelEvaluator==null){
			logger.info("PMML evaluator is null");
			return random.nextDouble();
		}
		Player player = context.getPlayer(playerId);
		Player opponent = context.getOpponent(player);
//		if (player.getHero().isDestroyed()) {   // 己方被干掉，得分 负无穷
//			return Float.NEGATIVE_INFINITY;  // 正负无穷会影响envState的解析，如果要加的话可以改成 +-100之类的
//		}
//		if (opponent.getHero().isDestroyed()) {  // 对方被干掉，得分 正无穷
//			return Float.POSITIVE_INFINITY;
//		}
		List<Integer> envState = player.getPlayerState();
		envState.addAll(opponent.getPlayerState());
//		logger.info("envState:{}", envState);
		// 准备模型输入数据
		Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
		List<InputField> inputFields = this.modelEvaluator.getInputFields();
		int i = 0;
		for(InputField inputField : inputFields){
			FieldName inputFieldName = inputField.getName();
			Object rawValue = envState.get(i);
			FieldValue inputFieldValue = inputField.prepare(rawValue);
			arguments.put(inputFieldName, inputFieldValue);
			i++;
		}
		//	logger.info("PMML inputs {}", arguments);
		Map<FieldName, ?> results = this.modelEvaluator.evaluate(arguments);
		List<TargetField> targetFields = this.modelEvaluator.getTargetFields();
		//	logger.info("PMML results {}", results);
		//	logger.info("PMML targetFields {}", targetFields);
		Object targetFieldValue = results.get(targetFields.get(0).getName());
//		logger.info("PMML targetFieldValue {}", targetFieldValue);
		return (double)targetFieldValue;
	}

	@Override
	public void onGameOver(GameContext context, int playerId, int winningPlayerId) {
		// GameOver的时候会跳入这个函数
		if(winningPlayerId <= 1)
			playersWinCount[winningPlayerId] += 1;

		Player player = context.getPlayer(playerId);
		Player opponent = context.getOpponent(player);
		List<Integer> envState = player.getPlayerState();
		envState.addAll(opponent.getPlayerState());
		int HpDiff =  player.getHero().getHp() - opponent.getHero().getHp();
		logger.info("{'Turn': {}, 'envState': {}, 'winner': {}, 'HpDiff': {}}", this.turn, envState.toString(), winningPlayerId, HpDiff);  // 可以根据HpDiff来判断胜负和设定reward
		this.turn = 0;
		if (this.runningReward == 0){
			this.runningReward = (double)HpDiff;
		}else{
			this.runningReward = this.runningReward*0.95 + HpDiff*0.05;
		}
		logger.info("##################Winning Player is : {} ##################", winningPlayerId);
		logger.info("playersWinCount: {}, player0 win ratio: {}, running reward: {}", playersWinCount, 1.0*playersWinCount[0]/(playersWinCount[0]+playersWinCount[1]), runningReward);

		try {
			// 记录胜率情况到文件
			FileWriter fileWriter = new FileWriter(new File("E:\\MetaStone\\app\\winStat.log"), true);
			String s = String.format("player0 wins:%d, player1 wins:%d, player0 win ratio:%f, running reward:%f\n",
					playersWinCount[0],playersWinCount[1],1.0*playersWinCount[0]/(playersWinCount[0]+playersWinCount[1]),runningReward);
			fileWriter.write(s);
			fileWriter.close();

			// 使用最新一局的数据更新模型，调用外部Python脚本，保存更新后的模型到PMML文件
			Process pr = Runtime.getRuntime().exec("D:\\Anaconda\\python.exe E:\\MetaStone\\game\\src\\main\\java\\net\\demilich\\metastone\\game\\behaviour\\model_RL.py");
			BufferedReader br = new BufferedReader(new InputStreamReader(pr.getInputStream()));
			String line;
			while ((line = br.readLine()) != null) {
				logger.info(line);
			}
			br.close();
			int exitValue = pr.waitFor();
			if (exitValue!=0)
				logger.info("exitValue: " + exitValue + "Attention: There must be something wrong in the Python script!!!!!!!!!!!!!!!!!!");  // exitValue = 0 表示python子进程正常退出，如果为1可能是Python代码报错了

			// 加载python训练保存的PMML文件
			loadPMMLModel(pmmlFile);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
