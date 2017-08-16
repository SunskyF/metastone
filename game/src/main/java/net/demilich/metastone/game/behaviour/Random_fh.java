package net.demilich.metastone.game.behaviour;

import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Random_fh extends Behaviour{
    private final static Logger logger = LoggerFactory.getLogger(Random_fh.class);

    @Override
    public String getName() {
        return "Random fh";
    }

    @Override
    public List<Card> mulligan(GameContext context, Player player, List<Card> cards){
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
        Random random = new Random();
        int randomIndex = random.nextInt(validActions.size());
        logger.info("random index {}", randomIndex);
        return validActions.get(randomIndex);
    }


}
