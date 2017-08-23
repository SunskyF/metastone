package net.demilich.metastone.game.behaviour.mcts;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

class UctPolicy implements ITreePolicy {
	private final static Logger logger = LoggerFactory.getLogger(UctPolicy.class);

	private static final double EPSILON = 1e-5;
	private static final Random random = new Random();

	private static final double C = 1 / Math.sqrt(2);

	@Override
	public Node select(Node parent) {
		Node selected = null;
		double bestValue = Double.NEGATIVE_INFINITY;
		for (Node child : parent.getChildren()) {
			double uctValue = child.getVisits() == 0 ? 1000000
					: child.totalReward / (double) child.getVisits() + C * Math.sqrt(Math.log(parent.getVisits()) / child.getVisits())
							+ random.nextDouble() * EPSILON;
			logger.info("UCT score: {}", uctValue);
			// small random number to break ties randomly in unexpanded nodes
			if (uctValue > bestValue) {
				selected = child;
				bestValue = uctValue;
			}
		}
		return selected;
	}

}
