package net.demilich.metastone.game.behaviour.mcts;

import java.util.*;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.ActionType;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.behaviour.PlayRandomBehaviour;
import net.demilich.metastone.game.behaviour.heuristic.IGameStateHeuristic;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class Node {
	private final static Logger logger = LoggerFactory.getLogger(Node.class);
	private static int simulationCount = 0;
	private final IGameStateHeuristic heuristic;

	private GameContext state;
	public Map<GameAction, Node> nextNodes = new HashMap<>();
	private final List<Node> children = new LinkedList<>();
	public int visitsCount;
	public int totalReward;
	private final int player;
	private Random random = new Random();
	private final int depth = 2;
	private HashMap<List<Integer>, Double> storeValue;
	private HashMap<List<Integer>, Node> storeNode;
	Node parent = null;
	private int simCnt = 0;
	GameAction comingAct;
	double[] parWeight = {0.760326412267397, 1.0589083919017865, -0.9400597870303837, 1.8744045092345258, 0.40236541366111334, 0.7418310137683413, 0.484131665708331, -0.7552493510082674, -0.49584676055695726, -0.16698138431239073, 0.36754468394420575, 1.4864556790810401, 1.1277694490216241, -0.19385547308776607, -1.0543152353607872, -0.2839165182591483, -1.4812832356356582, 0.3152121620740715, 0.0576844177240588, 1.1548436693125355, 1.9800036654926079, -0.3901963327635122, -0.7253225619942502, 0.18436967723674624, -0.6734730405335783, 0.5670641396548124, 0.639198399863008, -2.323129891333293, -1.0595513754902268, -0.804768708781494, 0.049672910287548194, 0.45954962674589844, -1.2431053666129253, 1.453612258307805, 0.26218920504452126, -2.0395371675289904, 0.18812695989268322, -1.328556982125309, -0.09499834083064819, -1.3747489091708807, 0.4488316541321531, 1.557497817592364, 1.0913026067029423, -0.597424707768485, -0.8361789076762033, -0.30777376598675443, -0.12238371097680156, 1.5841665240946965, -1.6267115776236394, -0.8061619411443898, -0.6975022521655243, 0.8245686938375283, -0.1718025408001051, -1.965065224429499, -1.59218011905894, 0.5600873334680981, -0.1825971189270253, -0.584501980650917, 2.416352484932763, -0.994680624848772, 1.9001580351986664, 0.11200461958090441, -1.149550315376586, 0.08082039517078825, -0.16580058883653237, -1.571984435874137, -2.4570940338800336, -0.15228687237889565, -0.1436000925996575, -0.06952984168734147, 0.8184547686840771, 0.6615721068181535, -0.2850074919127527, 0.16685275047289128, -0.01777378273904684, -1.705839086129454, -0.22775080851185345, 1.740876345330731, 0.2765548960828773, -0.3556647399330023, 0.4369754630045146, 0.679533140719245, 0.7003181586567979, 0.3695026344221172, -0.45705065513923954, -0.5747965912920181, 1.4643969249437774, 3.3167090234141243};


	public Node(GameContext context, int player, Node parent, IGameStateHeuristic heuristic, HashMap<List<Integer>, Node> storeNode,
				HashMap<List<Integer>, Double> storeValue, GameAction act) {
		this.player = player;
		this.state = context.clone();
		this.state.getPlayer(1-player).setBehaviour(new PlayRandomBehaviour());
		this.parent = parent;
		this.visitsCount = 0;
		this.totalReward = 0;
		this.heuristic = heuristic;
		this.storeNode = storeNode;
		this.storeValue = storeValue;
		this.comingAct = act;
		initActionMap();
	}

	private boolean isFullyExpanded() {
		//logger.info("Next Nodes Size: {}", nextNodes.size());
		//logger.info("Next Valid Action: {}", state.getValidActions());
		for (GameAction act: nextNodes.keySet()){
			if (nextNodes.get(act) == null)
				return false;
		}
		return true;
	}

	private GameContext simulateAction(GameContext simulation, int playerId, GameAction action) {
		simulation.getLogic().performGameAction(playerId, action);   // 在simulation GameContext中执行action，似乎是获取logic模块来执行action的
		return simulation;
	}

	private Node expand() {
		double bestScore = Integer.MIN_VALUE;
		GameAction action = null;
		for (GameAction tmp: nextNodes.keySet()){
			action = tmp;
			break;
		}

		for (GameAction act: nextNodes.keySet()){//选择一个没有expand过的节点
			if (nextNodes.get(act) == null){
				GameContext simulationResult = simulateAction(state.clone(), state.getActivePlayerId(), act);  //假设执行gameAction，得到之后的game context
				double gameStateScore = evaluateContext(simulationResult, state.getActivePlayerId());	//heuristic.getScore(simulationResult, player.getId());     //heuristic评估执行gameAction之后的游戏局面的分数
				if(gameStateScore > bestScore){
					action = act;
				}
				simulationResult.dispose();  //GameContext环境每次仿真完销毁
			}
		}


//		List<GameAction> alterList = new ArrayList<>();
//		for (GameAction act: nextNodes.keySet()){//选择一个没有expand过的节点
//			if (nextNodes.get(act) == null){
//				alterList.add(act);
//			}
//		}
//		GameAction action = alterList.get(random.nextInt(alterList.size()));


		//logger.info("Player: {}, Choosed Action: {}", getState().getActivePlayerId(), action.toString());
		GameContext newState = getState().clone();

		try {
			newState.performAction(newState.getActivePlayer().getId(), action);
			if (action.getActionType() == ActionType.END_TURN) {
				//logger.info("After End Turn: {}", newState.getActivePlayerId());
				newState.startTurn(newState.getActivePlayerId());
			}
		} catch (Exception e) {
			System.err.println("Exception on action: " + action + " state decided: " + state.gameDecided());
			e.printStackTrace();
			throw e;
		}
		Node child = new Node(newState, getPlayer(), this, heuristic, this.storeNode, this.storeValue, action);
		//logger.info("Next Player: {}, Next Valid Actions: {}", newState.getActivePlayerId(), newState.getValidActions());
		//logger.info("Expand New Node Next Nodes Size: {}", child.nextNodes.size());

		nextNodes.put(action, child);
//		logger.info(""+nextNodes.get(action).hashCode());
		return child;
	}

	int defaultPolicy(){
		simulationCount++;
		if (getState().gameDecided()) {
			GameContext state = getState();
			return state.getWinningPlayerId() == getPlayer() ? 1 : 0;
		}

		GameContext simulation = getState().clone();//随机对战直到结束
		for (Player player : simulation.getPlayers()) {
			player.setBehaviour(new PlayRandomBehaviour());
		}
		//logger.info("Default Policy Winner: {}", simulation.getWinningPlayerId());
//		simulation.playFromState();
		simulation.playFromState(getState().getTurn() + 10);
//		return simulation.getPlayer(getPlayer()).getHero().getHp() > simulation.getPlayer(1-getPlayer()).getHero().getHp() ? 1:0;
//		logger.info("Default Policy Winner: {}", simulation.getWinningPlayerId());
		//logger.info("Got Reward: {}", simulation.getWinningPlayerId() == getPlayer() ? 1 : -1);
//		return simulation.getWinningPlayerId() == getPlayer() ? 1 : 0;
//		logger.info("Score: {}, {}", this.heuristic.getScore(simulation, getPlayer()), this.heuristic.getScore(simulation, 1-getPlayer()));
		return this.heuristic.getScore(simulation, getPlayer()) > this.heuristic.getScore(simulation, 1-getPlayer()) ? 1 : 0;
	}

	void backup(int reward){
		Node node = this;
		//logger.info("Player: {}, Active: {}", node.getPlayer(), getState().getActivePlayerId());
		if (node.getPlayer() == getState().getActivePlayerId()){
//			reward = -reward;
		}
		int prePlayer = getState().getActivePlayerId();
		while(node != null){
			node.visitsCount += 1;
//			logger.info("Reward: {}", reward);
			node.totalReward += reward;
			node = node.parent;
			if (node != null && node.getState().getActivePlayerId() != prePlayer){
//				reward = -reward;
				prePlayer = node.getState().getActivePlayerId();
			}
		}
	}

	public GameAction getBestAction() {
		Node best = bestChild(0);
		for(GameAction act: nextNodes.keySet()){
			if (nextNodes.get(act) == best){
				return act;
			}
		}
		return getState().getValidActions().get(0);
	}

	public List<Node> getChildren() {
		return children;
	}

	public int getPlayer() {
		return player;
	}

	public GameContext getState() {
		return state;
	}

	int getVisits() {
		return visitsCount;
	}

	public void initActionMap() {
		List<GameAction> validActions = state.getValidActions();
		//logger.info("Valid: {}", validActions);
		for(GameAction act: validActions){
			nextNodes.put(act, null);
		}
	}

	private boolean isTerminal() {
		return state.gameDecided();
	}

	private Node bestChild(double c){
		List<Node> childList = new ArrayList<>();

		for(GameAction act: nextNodes.keySet()){
			if (nextNodes.get(act) != null){
				childList.add(nextNodes.get(act));
				//logger.info("Valid Action: {}", act);
			}
		}
		assert(childList.size() != 0);
		Node bestChild = null;
		double maxQ = Integer.MIN_VALUE;

		for (Node child: childList){
			double winRate = child.totalReward * 1.0 / child.visitsCount;
//			if (getPlayer() == child.getState().getActivePlayerId())
//				winRate = 1 - winRate;
			double UCB = winRate +
					c * Math.sqrt(Math.log(visitsCount) / child.visitsCount);
			double heuristic = evaluateContext(child.getState(), getState().getActivePlayerId());
//			logger.info("UCB: {}, Heuristic: {}", UCB, heuristic);
			double Q = UCB * (simCnt / 10) + heuristic / 500;
//			double Q = UCB;
//			logger.info("Sim Count: {}", simulationCount);
//			logger.info("UCB: {}, Heuristic: {}, Q: {}", UCB, heuristic, Q);
			if (Q > maxQ){
				maxQ = Q;
				bestChild = child;
			}
		}
		if (bestChild == null){
			logger.info("Error");
		}
		return bestChild;
	}

	Node treePolicy() {
		Node current = this;
		int depth = 6;
		while (!current.isTerminal()) {
			if (current.comingAct != null && current.comingAct.getActionType() == ActionType.END_TURN)
				return current;

			if (!current.isFullyExpanded()) {
				return current.expand();
			} else {
				current = current.bestChild(0.5);
				depth--;
//				if (depth == 0 || current.getState().getActivePlayerId() != getPlayer()){
				if (depth == 0){
					break;
				}
			}
		}
		return current;
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

		double score = 0;
		assert(parWeight.length == envState.size());
		for (int i = 0; i < parWeight.length; i++){
			score += parWeight[i]*envState.get(i);
		}
		return score;
	}

	public static void resetSimCount(){
		simulationCount = 0;
	}
	public void setSimCnt(int simCnt){
		this.simCnt = simCnt;
	}
}
