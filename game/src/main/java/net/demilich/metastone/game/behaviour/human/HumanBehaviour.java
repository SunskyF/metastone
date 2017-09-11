package net.demilich.metastone.game.behaviour.human;

import java.util.ArrayList;
import java.util.List;

import net.demilich.metastone.game.Attribute;
import net.demilich.metastone.game.entities.heroes.Hero;
import net.demilich.metastone.game.entities.heroes.HeroClass;
import net.demilich.metastone.game.entities.minions.Minion;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.demilich.metastone.BuildConfig;
import net.demilich.metastone.GameNotification;
import net.demilich.metastone.NotificationProxy;
import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.actions.IActionSelectionListener;
import net.demilich.metastone.game.behaviour.Behaviour;
import net.demilich.metastone.game.cards.Card;

public class HumanBehaviour extends Behaviour implements IActionSelectionListener {

	private final static Logger logger = LoggerFactory.getLogger(HumanBehaviour.class);

	private GameAction selectedAction;
	private boolean waitingForInput;
	private List<Card> mulliganCards;

	@Override
	public String getName() {
		return "<Human controlled>";
	}

	@Override
	public List<Card> mulligan(GameContext context, Player player, List<Card> cards) {
		if (context.ignoreEvents()) {
			return new ArrayList<Card>();
		}
		waitingForInput = true;
		HumanMulliganOptions options = new HumanMulliganOptions(player, this, cards);
		NotificationProxy.sendNotification(GameNotification.HUMAN_PROMPT_FOR_MULLIGAN, options);
		while (waitingForInput) {
			try {
				Thread.sleep(BuildConfig.DEFAULT_SLEEP_DELAY);
			} catch (InterruptedException e) {
			}
		}
		return mulliganCards;
	}

	@Override
	public void onActionSelected(GameAction action) {
		this.selectedAction = action;
		waitingForInput = false;
	}

	private static int calculateThreatLevel(GameContext context, int playerId) {
		int damageOnBoard = 0;
		Player player = context.getPlayer(playerId);
		Player opponent = context.getOpponent(player);
		for (Minion minion : opponent.getMinions()) {
			damageOnBoard += minion.getAttack();  //(暂时没有考虑风怒、冻结等的影响，因为 之前 minion.getAttributeValue(Attribute.NUMBER_OF_ATTACKS)经常得到0)
		}
		damageOnBoard += getHeroDamage(opponent.getHero());  //对方随从 + 英雄的攻击力

		int remainingHp = player.getHero().getEffectiveHp() - damageOnBoard;  // 根据减去对方伤害后我方剩余血量来确定威胁等级
		return remainingHp;
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

	@Override
	public GameAction requestAction(GameContext context, Player player, List<GameAction> validActions) {
		// Sjx, output info

//		logger.info("num of valid actions {}, they are {}.", validActions.size(), validActions);
		/*
		logger.info("player state vec: {}.", player.getPlayerState());
		logger.info("opp state vec: {}.", context.getOpponent(player).getPlayerState());
		logger.info("Threat Level remainingHp:{}", calculateThreatLevel(context, player.getId()));
		logger.info("num of Summons:{}, Summons: {}.", player.getSummons().size(), player.getSummons());
		logger.info("num of Minions:{}, Minions: {}.", player.getMinions().size(), player.getMinions());
		logger.info("num in Hand: {}, Hand: {}.", player.getHand().getCount(), player.getHand().toList());
		*/
		waitingForInput = true;
		HumanActionOptions options = new HumanActionOptions(this, context, player, validActions);
		NotificationProxy.sendNotification(GameNotification.HUMAN_PROMPT_FOR_ACTION, options);
		while (waitingForInput) {
			try {
				Thread.sleep(BuildConfig.DEFAULT_SLEEP_DELAY);
				if (context.ignoreEvents()) {
					return null;
				}
			} catch (InterruptedException e) {
			}
		}
		return selectedAction;
	}

	public void setMulliganCards(List<Card> mulliganCards) {
		this.mulliganCards = mulliganCards;
		waitingForInput = false;
	}

}
