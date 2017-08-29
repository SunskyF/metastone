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
	private final IGameStateHeuristic heuristic;

	private GameContext state;
	public Map<GameAction, Node> nextNodes = new HashMap<>();
	private final List<Node> children = new LinkedList<>();
	public int visitsCount;
	public int totalReward;
	private final int player;
	private Random random = new Random();
	private final int depth = 2;
	Node parent = null;

	public Node(GameContext context, int player, Node parent, IGameStateHeuristic heuristic) {
		this.player = player;
		this.state = context.clone();
		this.state.getPlayer(1-player).setBehaviour(new PlayRandomBehaviour());
		this.parent = parent;
		this.visitsCount = 0;
		this.totalReward = 0;
		this.heuristic = heuristic;
		initActionMap();
	}

	private boolean isFullyExpanded() {
		//logger.info("Next Nodes Size: {}", nextNodes.size());
		//logger.info("Next Valid Action: {}", state.getValidActions());
		for (Node value: nextNodes.values()){
			if (value == null)
				return false;
		}
		return true;
	}

	private Node expand() {
		List<GameAction> alterList = new ArrayList<>();
		for (GameAction act: nextNodes.keySet()){//选择一个没有expand过的节点
			if (nextNodes.get(act) == null)
				alterList.add(act);
		}

		GameAction action = alterList.get(random.nextInt(alterList.size()));
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
		Node child = new Node(newState, getPlayer(), this, heuristic);
		//logger.info("Next Player: {}, Next Valid Actions: {}", newState.getActivePlayerId(), newState.getValidActions());
		//logger.info("Expand New Node Next Nodes Size: {}", child.nextNodes.size());

		nextNodes.put(action, child);
//		logger.info(""+nextNodes.get(action).hashCode());
		return child;
	}

	int defaultPolicy(){
		if (getState().gameDecided()) {
			GameContext state = getState();
			return state.getWinningPlayerId() == getPlayer() ? 1 : 0;
		}

		GameContext simulation = getState().clone();//随机对战直到结束
		for (Player player : simulation.getPlayers()) {
			player.setBehaviour(new PlayRandomBehaviour());
		}
		//logger.info("Default Policy Winner: {}", simulation.getWinningPlayerId());
		simulation.playFromState();
//		logger.info("Default Policy Winner: {}", simulation.getWinningPlayerId());
		//logger.info("Got Reward: {}", simulation.getWinningPlayerId() == getPlayer() ? 1 : -1);
		return simulation.getWinningPlayerId() == getPlayer() ? 1 : 0;
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
				reward = -reward;
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
		double maxUCB = -200;
		for (Node child: childList){
			double UCB = child.totalReward * 1.0 / child.visitsCount +
					c * Math.sqrt(2.0 * Math.log(visitsCount) / child.visitsCount);
			//logger.info("UCB: {}", UCB);
			if (UCB > maxUCB){
				maxUCB = UCB;
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
		while (!current.isTerminal()) {
			//logger.info("Checkpoint 0");
			if (!current.isFullyExpanded()) {
				//logger.info("Checkpoint 1");
				return current.expand();
			} else {
				//logger.info("Checkpoint 2");
				current = current.bestChild(1);
				break;
			}
		}
		return current;
	}

}
