package net.demilich.metastone.game.behaviour.mcts;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import net.demilich.metastone.game.actions.ActionType;
import net.demilich.metastone.game.behaviour.heuristic.SupervisedLinearHeuristic;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.behaviour.Behaviour;
import net.demilich.metastone.game.cards.Card;

public class MonteCarloTreeSearch extends Behaviour {

	private final static Logger logger = LoggerFactory.getLogger(MonteCarloTreeSearch.class);

	private static final int ITERATIONS = 500;

	@Override
	public String getName() {  // 这个似乎还没有完整实现，没法跑， Node.process() 报NullPointer Exception
		return "MCTS";
	}

	@Override
	public List<Card> mulligan(GameContext context, Player player, List<Card> cards) {
		List<Card> discardedCards = new ArrayList<Card>();
		for (Card card : cards) {
			if (card.getBaseManaCost() >= 4) {
				discardedCards.add(card);
			}
		}
		return discardedCards;
	}

	@Override
	public GameAction requestAction(GameContext context, Player player, List<GameAction> validActions) {
		logger.info("HP: {}, {}", context.getPlayer(0).getHero().getHp(), context.getPlayer(1).getHero().getHp());
		Random random = new Random();
		if (validActions.size() == 1) {
			logger.info("MCTS selected best action (Only one choice): {}", validActions.get(0));
			return validActions.get(0);
		}

		if (validActions.get(0).getActionType() == ActionType.BATTLECRY) {
			return validActions.get(random.nextInt(validActions.size()));
		}
		if (validActions.get(0).getActionType() == ActionType.DISCOVER) {
			return validActions.get(random.nextInt(validActions.size()));

		}

		Node root = new Node(context, player.getId(), null, new SupervisedLinearHeuristic());
		//UctPolicy treePolicy = new UctPolicy();
		for (int i = 0; i < ITERATIONS; i++) {
			//logger.info("Iter: {}", i);
			Node node = root.treePolicy();
			int reward = node.defaultPolicy();
			node.backup(reward);
		}
		for(GameAction act: root.nextNodes.keySet()){
			if (root.nextNodes.get(act) != null){
				logger.info("Action: {}, score: {}, visit: {}", act, root.nextNodes.get(act).totalReward, root.nextNodes.get(act).visitsCount);
			}
		}
		GameAction bestAction = root.getBestAction();
		logger.info("MCTS selected best action {}", bestAction);
		return bestAction;
	}

}
