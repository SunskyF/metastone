package net.demilich.metastone.game;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

import net.demilich.metastone.game.behaviour.features.Feature_basic;
import net.demilich.metastone.game.cards.CardType;
import net.demilich.metastone.game.entities.heroes.HeroClass;
import net.demilich.metastone.game.entities.minions.Minion;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.demilich.metastone.game.actions.ActionType;
import net.demilich.metastone.game.actions.GameAction;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.cards.CardCatalogue;
import net.demilich.metastone.game.cards.CardCollection;
import net.demilich.metastone.game.cards.costmodifier.CardCostModifier;
import net.demilich.metastone.game.decks.DeckFormat;
import net.demilich.metastone.game.entities.Entity;
import net.demilich.metastone.game.entities.minions.Summon;
import net.demilich.metastone.game.events.GameEvent;
import net.demilich.metastone.game.logic.GameLogic;
import net.demilich.metastone.game.logic.MatchResult;
import net.demilich.metastone.game.logic.TargetLogic;
import net.demilich.metastone.game.spells.trigger.IGameEventListener;
import net.demilich.metastone.game.spells.trigger.TriggerManager;
import net.demilich.metastone.game.targeting.CardReference;
import net.demilich.metastone.game.targeting.EntityReference;
import net.demilich.metastone.utils.IDisposable;

public class GameContext implements Cloneable, IDisposable {
	public static final int PLAYER_1 = 0;
	public static final int PLAYER_2 = 1;

	private static final Logger logger = LoggerFactory.getLogger(GameContext.class);

	private final Player[] players = new Player[2];
	private final GameLogic logic;
	private final DeckFormat deckFormat;
	private final TargetLogic targetLogic = new TargetLogic();
	private TriggerManager triggerManager = new TriggerManager();
	private final HashMap<Environment, Object> environment = new HashMap<>();
	private final List<CardCostModifier> cardCostModifiers = new ArrayList<>();

	protected int activePlayer = -1;
	private Player winner;
	private MatchResult result;
	private TurnState turnState = TurnState.TURN_ENDED;

	private int turn;
	private int actionsThisTurn;

	private boolean ignoreEvents;

	private CardCollection tempCards = new CardCollection();

	public GameContext(Player player1, Player player2, GameLogic logic, DeckFormat deckFormat) {
		this.getPlayers()[PLAYER_1] = player1;
		player1.setId(PLAYER_1);
		this.getPlayers()[PLAYER_2] = player2;
		player2.setId(PLAYER_2);
		this.logic = logic;
		this.deckFormat = deckFormat;
		this.logic.setContext(this);
		tempCards.removeAll();
	}

	protected boolean acceptAction(GameAction nextAction) {
		return true;
	}

	public void addCardCostModifier(CardCostModifier cardCostModifier) {
		getCardCostModifiers().add(cardCostModifier);
	}

	public void addTempCard(Card card) {
		tempCards.add(card);
	}

	public void addTrigger(IGameEventListener trigger) {
		triggerManager.addTrigger(trigger);
	}

	@Override
	public GameContext clone() {
		GameLogic logicClone = getLogic().clone();
		Player player1Clone = getPlayer1().clone();
		// player1Clone.getDeck().shuffle();
		Player player2Clone = getPlayer2().clone();
		// player2Clone.getDeck().shuffle();
		GameContext clone = new GameContext(player1Clone, player2Clone, logicClone, deckFormat);
		clone.tempCards = tempCards.clone();
		clone.triggerManager = triggerManager.clone();
		clone.activePlayer = activePlayer;
		clone.turn = turn;
		clone.actionsThisTurn = actionsThisTurn;
		clone.result = result;
		clone.turnState = turnState;
		clone.winner = logicClone.getWinner(player1Clone, player2Clone);
		clone.cardCostModifiers.clear();
		for (CardCostModifier cardCostModifier : cardCostModifiers) {
			clone.cardCostModifiers.add(cardCostModifier.clone());
		}
		
		Stack<Integer> damageStack = new Stack<Integer>();
		damageStack.addAll(getDamageStack());
		clone.getEnvironment().put(Environment.DAMAGE_STACK, damageStack);
		Stack<EntityReference> summonReferenceStack = new Stack<EntityReference>();
		summonReferenceStack.addAll(getSummonReferenceStack());
		clone.getEnvironment().put(Environment.SUMMON_REFERENCE_STACK, summonReferenceStack);
		Stack<EntityReference> eventTargetReferenceStack = new Stack<EntityReference>();
		eventTargetReferenceStack.addAll(getEventTargetStack());
		clone.getEnvironment().put(Environment.EVENT_TARGET_REFERENCE_STACK, eventTargetReferenceStack);
		
		for (Environment key : getEnvironment().keySet()) {
			if (!key.customClone()) {
				clone.getEnvironment().put(key, getEnvironment().get(key));
			}
		}
		clone.getLogic().setLoggingEnabled(false);
		return clone;
	}

	@Override
	public void dispose() {
		for (int i = 0; i < players.length; i++) {
			players[i] = null;
		}
		getCardCostModifiers().clear();
		triggerManager.dispose();
		environment.clear();
	}

	private void endGame() {
		winner = logic.getWinner(getActivePlayer(), getOpponent(getActivePlayer()));
		for (Player player : getPlayers()) {
			player.getBehaviour().onGameOver(this, player.getId(), winner != null ? winner.getId() : -1);
		}

		if (winner != null) {
			logger.debug("Game finished after " + turn + " turns, the winner is: " + winner.getName());
			winner.getStatistics().gameWon();
			Player looser = getOpponent(winner);
			looser.getStatistics().gameLost();
		} else {
			logger.debug("Game finished after " + turn + " turns, DRAW");
			getPlayer1().getStatistics().gameLost();
			getPlayer2().getStatistics().gameLost();
		}
	}

	public void endTurn() {
		logic.endTurn(activePlayer);
		activePlayer = activePlayer == PLAYER_1 ? PLAYER_2 : PLAYER_1;
		onGameStateChanged();
		turnState = TurnState.TURN_ENDED;
	}

	private Card findCardinCollection(CardCollection cardCollection, int cardId) {
		for (Card card : cardCollection) {
			if (card.getId() == cardId) {
				return card;
			}
		}
		return null;
	}

	public void fireGameEvent(GameEvent gameEvent) {
		if (ignoreEvents()) {
			return;
		}
		try {
			triggerManager.fireGameEvent(gameEvent);	
		} catch(Exception e) {
			logger.error("Error while processing gameEvent {}", gameEvent);
			logic.panicDump();
			throw e;
		}
		
	}

	public boolean gameDecided() {
		result = logic.getMatchResult(getActivePlayer(), getOpponent(getActivePlayer()));
		winner = logic.getWinner(getActivePlayer(), getOpponent(getActivePlayer()));
		return result != MatchResult.RUNNING;
	}

	public Player getActivePlayer() {
		return getPlayer(activePlayer);
	}

	public int getActivePlayerId() {
		return activePlayer;
	}

	public List<Summon> getAdjacentSummons(Player player, EntityReference minionReference) {
		List<Summon> adjacentSummons = new ArrayList<>();
		Summon summon = (Summon) resolveSingleTarget(minionReference);
		List<Summon> summons = getPlayer(summon.getOwner()).getSummons();
		int index = summons.indexOf(summon);
		if (index == -1) {
			return adjacentSummons;
		}
		int left = index - 1;
		int right = index + 1;
		if (left > -1 && left < summons.size()) {
			adjacentSummons.add(summons.get(left));
		}
		if (right > -1 && right < summons.size()) {
			adjacentSummons.add(summons.get(right));
		}
		return adjacentSummons;
	}

	public GameAction getAutoHeroPowerAction() {
		return logic.getAutoHeroPowerAction(activePlayer);
	}

	public int getBoardPosition(Summon summon) {
		for (Player player : getPlayers()) {
			List<Summon> summons = player.getSummons();
			for (int i = 0; i < summons.size(); i++) {
				if (summons.get(i) == summon) {
					return i;
				}
			}
		}
		return -1;
	}

	public Card getCardById(String cardId) {
		Card card = CardCatalogue.getCardById(cardId);
		if (card == null) {
			for (Card tempCard : tempCards) {
				if (tempCard.getCardId().equalsIgnoreCase(cardId)) {
					return tempCard.clone();
				}
			}
		}
		return card;
	}

	public List<CardCostModifier> getCardCostModifiers() {
		return cardCostModifiers;
	}
	
	@SuppressWarnings("unchecked")
	public Stack<Integer> getDamageStack() {
		if (!environment.containsKey(Environment.DAMAGE_STACK)) {
			environment.put(Environment.DAMAGE_STACK, new Stack<Integer>());
		}
		return (Stack<Integer>) environment.get(Environment.DAMAGE_STACK);
	}

	public DeckFormat getDeckFormat() {
		return deckFormat;
	}

	public HashMap<Environment, Object> getEnvironment() {
		return environment;
	}
	
	public Card getEventCard() {
		return (Card) resolveSingleTarget((EntityReference) getEnvironment().get(Environment.EVENT_CARD));
	}

	@SuppressWarnings("unchecked")
	public Stack<EntityReference> getEventTargetStack() {
		if (!environment.containsKey(Environment.EVENT_TARGET_REFERENCE_STACK)) {
			environment.put(Environment.EVENT_TARGET_REFERENCE_STACK, new Stack<EntityReference>());
		}
		return (Stack<EntityReference>) environment.get(Environment.EVENT_TARGET_REFERENCE_STACK);
	}

	public List<Summon> getLeftSummons(Player player, EntityReference minionReference) {
		List<Summon> leftSummons = new ArrayList<>();
		Summon summon = (Summon) resolveSingleTarget(minionReference);
		List<Summon> summons = getPlayer(summon.getOwner()).getSummons();
		int index = summons.indexOf(summon);
		if (index == -1) {
			return leftSummons;
		}
		for (int i = 0; i < index; i++) {
			leftSummons.add(summons.get(i));
		}
		return leftSummons;
	}

	public GameLogic getLogic() {
		return logic;
	}

	public int getMinionCount(Player player) {
		return player.getMinions().size();
	}

	public int getSummonCount(Player player) {
		return player.getSummons().size();
	}

	public Player getOpponent(Player player) {
		return player.getId() == PLAYER_1 ? getPlayer2() : getPlayer1();
	}

	public List<Summon> getOppositeSummons(Player player, EntityReference minionReference) {
		List<Summon> oppositeSummons = new ArrayList<>();
		Summon summon = (Summon) resolveSingleTarget(minionReference);
		Player owner = getPlayer(summon.getOwner());
		Player opposingPlayer = getOpponent(owner);
		int index = owner.getSummons().indexOf(summon);
		if (opposingPlayer.getSummons().size() == 0 || owner.getSummons().size() == 0 || index == -1) {
			return oppositeSummons;
		}
		List<Summon> opposingSummons = opposingPlayer.getSummons();
		int delta = opposingPlayer.getSummons().size() - owner.getSummons().size();
		if (delta % 2 == 0) {
			delta /= 2;
			int epsilon = delta + index;
			if (epsilon > -1 && epsilon < opposingSummons.size()) {
				oppositeSummons.add(opposingSummons.get(epsilon));
			}
		} else {
			delta = (delta - 1) / 2;
			int epsilon = delta + index;
			if (epsilon > -1 && epsilon < opposingSummons.size()) {
				oppositeSummons.add(opposingSummons.get(epsilon));
			}
			if (epsilon + 1 > -1 && epsilon + 1 < opposingSummons.size()) {
				oppositeSummons.add(opposingSummons.get(epsilon + 1));
			}
		}
		return oppositeSummons;
	}
	
	public Card getPendingCard() {
		return (Card) resolveSingleTarget((EntityReference) getEnvironment().get(Environment.PENDING_CARD));
	}

	public Player getPlayer(int index) {
		return players[index];
	}

	public Player getPlayer1() {
		return getPlayers()[PLAYER_1];
	}

	public Player getPlayer2() {
		return getPlayers()[PLAYER_2];
	}

	public Player[] getPlayers() {
		return players;
	}

	public List<Summon> getRightSummons(Player player, EntityReference minionReference) {
		List<Summon> rightSummons = new ArrayList<>();
		Summon summon = (Summon) resolveSingleTarget(minionReference);
		List<Summon> summons = getPlayer(summon.getOwner()).getSummons();
		int index = summons.indexOf(summon);
		if (index == -1) {
			return rightSummons;
		}
		for (int i = index + 1; i < player.getSummons().size(); i++) {
			rightSummons.add(summons.get(i));
		}
		return rightSummons;
	}

	@SuppressWarnings("unchecked")
	public Stack<EntityReference> getSummonReferenceStack() {
		if (!environment.containsKey(Environment.SUMMON_REFERENCE_STACK)) {
			environment.put(Environment.SUMMON_REFERENCE_STACK, new Stack<EntityReference>());
		}
		return (Stack<EntityReference>) environment.get(Environment.SUMMON_REFERENCE_STACK);
	}

	public CardCollection getTempCards() {
		return tempCards;
	}

	public int getTotalMinionCount() {
		int totalMinionCount = 0;
		for (int i = 0; i < players.length; i++) {
			totalMinionCount += getMinionCount(players[i]);
		}
		return totalMinionCount;
	}

	public int getTotalSummonCount() {
		int totalSummonCount = 0;
		for (int i = 0; i < players.length; i++) {
			totalSummonCount += getSummonCount(players[i]);
		}
		return totalSummonCount;
	}

	public List<IGameEventListener> getTriggersAssociatedWith(EntityReference entityReference) {
		return triggerManager.getTriggersAssociatedWith(entityReference);
	}

	public int getTurn() {
		return turn;
	}

	public TurnState getTurnState() {
		return turnState;
	}

	public List<GameAction> getValidActions() {
		if (gameDecided()) {
			return new ArrayList<>();
		}
		return logic.getValidActions(activePlayer);
	}

	public int getWinningPlayerId() {
		return winner == null ? -1 : winner.getId();
	}

	public boolean hasAutoHeroPower() {
		if (gameDecided()) {
			return false;
		}
		return logic.hasAutoHeroPower(activePlayer);
	}

	public boolean ignoreEvents() {
		return ignoreEvents;
	}

	public void init() {
		int startingPlayerId = logic.determineBeginner(PLAYER_1, PLAYER_2);
		activePlayer = getPlayer(startingPlayerId).getId();
		logger.debug(getActivePlayer().getName() + " begins");
		logic.init(activePlayer, true);
		logic.init(getOpponent(getActivePlayer()).getId(), false);
	}

	protected void onGameStateChanged() {
	}

	public void performAction(int playerId, GameAction gameAction) {
		logic.performGameAction(playerId, gameAction);
		onGameStateChanged();
	}

	private void appendWrite(String fileName, String content) {
		FileWriter fw = null;
		try {
			//如果文件存在，则追加内容；如果文件不存在，则创建文件
			File f=new File(fileName);
			fw = new FileWriter(f, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		PrintWriter pw = new PrintWriter(fw);
		pw.println(content);
		pw.flush();
		try {
			fw.flush();
			pw.close();
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void play() {
		logger.debug("Game starts: " + getPlayer1().getName() + " VS. " + getPlayer2().getName());
		init();
		//StringBuilder sb = new StringBuilder();
		//int nowTurn = 0;
		int winner_id = 0;

		// feature
		/*
			1. Basic
			2. feature_fh_0
		 */
		String fea_name = "feature_fh_0";
		Feature_basic fea = new Feature_basic(hashCode(), fea_name);

		while (!gameDecided()) {  // 如果游戏胜负未分，开始切换后的activePlayer的turn
			startTurn(activePlayer);  // 开始当前activePlayer的Turn
			while (playTurn()) {}    // 循环play，直到执行END_TURN action，结束当前player的当前turn （主要是调用behaviour的requestAction）
			// add by sjx, 获取每一回合结束时的环境信息

			fea.append(players);
			winner_id = getWinningPlayerId();
			if (getTurn() > GameLogic.TURN_LIMIT) {
				break;
			}
		}

		logger.info(players[0].getDeckName());
		int num_sim = 50;
		fea.end(winner_id);
		String player_class_0 = "Warrior" + "_" + players[0].getDeckName();
		//players[1].getHero().getHeroPower().getHeroClass().toString()好像会报错
		String player_class_1 = "Warrior" + "_" + players[1].getDeckName();
		int fea_num = fea.feature_number();

		String filename = player_class_0 + "Vs" + player_class_1 +
				"_" + players[0].getBehaviour().getName() + "Vs" +
				players[1].getBehaviour().getName() + "_" +
				fea_name + "_" + fea_num + "_" + num_sim + ".log";

		//fea.appendWrite(filename);
		endGame();
		// add by sjx
		//logger.info("{'GameHash':" + hashCode() + ",'Turn':" + turn + ",'winner':" + winner.getId() + "}");
	}

	public void playFromState(){
		//Play the whole game starting from any turn
		while (!gameDecided()) {
			startTurn(getActivePlayer().getId());
			while (playTurn()) {}
			if (getTurn() > GameLogic.TURN_LIMIT) {
				break;
			}
		}

		endGame();

	}

	public boolean playTurn() {
		if (++actionsThisTurn > 99) {
			logger.warn("Turn has been forcefully ended after {} actions", actionsThisTurn);
			endTurn();
			return false;
		}
		if (logic.hasAutoHeroPower(activePlayer)) {
			performAction(activePlayer, getAutoHeroPowerAction());
			return true;
		}

		List<GameAction> validActions = getValidActions();
		if (validActions.size() == 0) {
			//endTurn();
			return false;
		}

		GameAction nextAction = getActivePlayer().getBehaviour().requestAction(this, getActivePlayer(), getValidActions());
		while (!acceptAction(nextAction)) {
			nextAction = getActivePlayer().getBehaviour().requestAction(this, getActivePlayer(), getValidActions());
		}
		if (nextAction == null) {
			throw new RuntimeException("Behaviour " + getActivePlayer().getBehaviour().getName() + " selected NULL action while "
					+ getValidActions().size() + " actions were available");
		}
		performAction(activePlayer, nextAction);

		return nextAction.getActionType() != ActionType.END_TURN;
	}

	public void printCurrentTriggers() {
		//logger.info("Active spelltriggers:");
		triggerManager.printCurrentTriggers();
	}
	
	public void removeTrigger(IGameEventListener trigger) {
		triggerManager.removeTrigger(trigger);
	}

	public void removeTriggersAssociatedWith(EntityReference entityReference, boolean removeAuras) {
		triggerManager.removeTriggersAssociatedWith(entityReference, removeAuras);
	}

	public Card resolveCardReference(CardReference cardReference) {
		Player player = getPlayer(cardReference.getPlayerId());
		if (getPendingCard() != null && getPendingCard().getCardReference().equals(cardReference)) {
			return getPendingCard();
		}
		switch (cardReference.getLocation()) {
		case DECK:
			return findCardinCollection(player.getDeck(), cardReference.getCardId());
		case HAND:
			return findCardinCollection(player.getHand(), cardReference.getCardId());
		case PENDING:
			return getPendingCard();
		case HERO_POWER:
			return player.getHero().getHeroPower();
		default:
			break;

		}
		logger.error("Could not resolve cardReference {}", cardReference);
		new RuntimeException().printStackTrace();
		return null;
	}

	public Entity resolveSingleTarget(EntityReference targetKey) {
		if (targetKey == null) {
			return null;
		}
		return targetLogic.findEntity(this, targetKey);
	}

	public List<Entity> resolveTarget(Player player, Entity source, EntityReference targetKey) {
		return targetLogic.resolveTargetKey(this, player, source, targetKey);
	}
	
	public void setEventCard(Card eventCard) {
		if (eventCard != null) {
			getEnvironment().put(Environment.EVENT_CARD, eventCard.getReference());
		} else {
			getEnvironment().put(Environment.EVENT_CARD, null);
		}
	}

	public void setIgnoreEvents(boolean ignoreEvents) {
		this.ignoreEvents = ignoreEvents;
	}
	
	public void setPendingCard(Card pendingCard) {
		if (pendingCard != null) {
			getEnvironment().put(Environment.PENDING_CARD, pendingCard.getReference());
		} else {
			getEnvironment().put(Environment.PENDING_CARD, null);
		}
	}

	public void startTurn(int playerId) {
		turn++;
		logic.startTurn(playerId);
		onGameStateChanged();
		actionsThisTurn = 0;
		turnState = TurnState.TURN_IN_PROGRESS;
	}

	public String feature_0(){//86 feature total, 43 each
		StringBuilder builder = new StringBuilder("{'GameHash':" + hashCode() + ",'Turn':" + getTurn());
		for (Player player : players){
			builder.append(",'player" + player.getId() + "':'");
			builder.append(player.getHero().getHp());  // 血量
			builder.append("|" + player.getHero().getArmor()); // 护甲
			builder.append("|" + player.getMana());  // 当前法力值
			int weaponDamage = 0;
			int weaponDurability = 0;
			if (player.getHero().getWeapon() != null) {
				weaponDamage = player.getHero().getWeapon().getWeaponDamage();  //武器伤害
				weaponDurability = player.getHero().getWeapon().getDurability(); //武器耐久
			}
			builder.append("|" + weaponDamage);
			builder.append("|" + weaponDurability);

			// 英雄技能, 暂时按照英雄类型将技能分为1、2、3三档，暂时不考虑一些非基础的英雄技能的影响
			int heroPower = 3; // 默认为3
			HeroClass heroPowerClass = player.getHero().getHeroPower().getHeroClass();
			if(heroPowerClass == HeroClass.HUNTER || heroPowerClass == HeroClass.MAGE || heroPowerClass == HeroClass.WARLOCK){
				heroPower = 3;
			}else if(heroPowerClass == HeroClass.DRUID || heroPowerClass == HeroClass.PALADIN || heroPowerClass == HeroClass.SHAMAN){
				heroPower = 2;
			}else if(heroPowerClass == HeroClass.PRIEST || heroPowerClass == HeroClass.ROGUE || heroPowerClass == HeroClass.WARRIOR){
				heroPower = 1;
			}
			builder.append("|" + heroPower);

			// 场上的随从相关数据
			int minionCount = 0;   // minions on board that can still attack (直观来说，一回合结束时，自己场上应该不会再有能攻击的随从还没用的情况)
			int minionAttack = 0;
			int minionHp = 0;
			int minionCountNot = 0; // minions on board that can not attack
			int minionAttackNot = 0;
			int minionHpNot = 0;
			int minionCountTaunt = 0;  // 带嘲讽的随从
			int minionAttackTaunt= 0;
			int minionHpTaunt = 0;
			int minionCountFrozen = 0;  // 冻结的随从
			int minionAttackFrozen= 0;
			int minionHpFrozen = 0;
			int minionCountStealth = 0;  // 潜行的随从
			int minionAttackStealth= 0;
			int minionHpStealth = 0;
			int minionCountShield = 0;  // 带圣盾的随从
			int minionAttackShield= 0;
			int minionHpShield = 0;
			int minionCountEnrage = 0;  // 带激怒的随从
			int minionAttackEnrage= 0;
			int minionHpEnrage = 0;
			int minionCountUntarget = 0;  // 不可被法术攻击的随从
			int minionAttackUntarget= 0;
			int minionHpUntarget = 0;
			int minionCountWindfury = 0; // 带风怒效果的随从
			int minionAttackWindfury = 0;
			int minionHpWindfury = 0;
			int minionCountSpell = 0;  // 带法术伤害的随从
			int minionSpellDamage = 0;

			for (Minion minion : player.getMinions()) {  // 场上的随从信息
				if (minion.canAttackThisTurn()) {
					minionCount += 1;
					minionAttack += minion.getAttack();
					minionHp += minion.getHp();
				} else {
					minionCountNot += 1;
					minionAttackNot += minion.getAttack();
					minionHpNot += minion.getHp();
				}
				if(minion.hasAttribute(Attribute.TAUNT)){
					minionCountTaunt += 1;
					minionAttackTaunt += minion.getAttack();
					minionHpTaunt += minion.getHp();
				}
				if (minion.hasAttribute(Attribute.FROZEN)) {  // 冻结的随从
					minionCountFrozen += 1;
					minionAttackFrozen += minion.getAttack();
					minionHpFrozen += minion.getHp();
				}
				if (minion.hasAttribute(Attribute.STEALTH)) {  // 潜行
					minionCountStealth += 1;
					minionAttackStealth += minion.getAttack();
					minionHpStealth += minion.getHp();
				}
				if (minion.hasAttribute(Attribute.DIVINE_SHIELD)) {  //圣盾
					minionCountShield += 1;
					minionAttackShield += minion.getAttack();
					minionHpShield += minion.getHp();
				}
				if (minion.hasAttribute(Attribute.ENRAGED)) {  // 激怒
					minionCountEnrage += 1;
					minionAttackEnrage += minion.getAttack();
					minionHpEnrage += minion.getHp();
				}
				if (minion.hasAttribute(Attribute.UNTARGETABLE_BY_SPELLS)) {  // 不能被法术指定
					minionCountUntarget += 1;
					minionAttackUntarget += minion.getAttack();
					minionHpUntarget += minion.getHp();
				}
				if (minion.hasAttribute(Attribute.WINDFURY) || minion.hasAttribute(Attribute.MEGA_WINDFURY)) {  // 风怒或超级风怒
					minionCountWindfury += 1;
					minionHpWindfury += minion.getHp();
					if (minion.hasAttribute(Attribute.MEGA_WINDFURY)){  // 风怒或超级风怒带来的额外的攻击力
						minionAttackWindfury += 3*minion.getAttack();
					}else{
						minionAttackWindfury += minion.getAttack();
					}
				}
				if (minion.hasAttribute(Attribute.SPELL_DAMAGE)) {  // 法术伤害
					minionCountSpell += 1;
					minionSpellDamage += minion.getAttributeValue(Attribute.SPELL_DAMAGE);
				}
			}

			builder.append("|" + minionCount + "|" + minionAttack + "|" + minionHp + "|" + minionCountNot + "|" + minionAttackNot + "|" + minionHpNot);
			builder.append("|" + minionCountTaunt + "|" + minionAttackTaunt + "|" + minionHpTaunt);
			builder.append("|" + minionCountFrozen + "|" + minionAttackFrozen + "|" + minionHpFrozen);
			builder.append("|" + minionCountStealth + "|" + minionAttackStealth + "|" + minionHpStealth);
			builder.append("|" + minionCountShield + "|" + minionAttackShield + "|" + minionHpShield);
			builder.append("|" + minionCountEnrage + "|" + minionAttackEnrage + "|" + minionHpEnrage);
			builder.append("|" + minionCountUntarget + "|" + minionAttackUntarget + "|" + minionHpUntarget);
			builder.append("|" + minionCountWindfury + "|" + minionAttackWindfury + "|" + minionHpWindfury);
			builder.append("|" + minionCountSpell + "|" + minionSpellDamage);


			// 手牌相关信息
			int cardMinionCount = 0;
			int cardMinionMana = 0;
			int cardMinionBattleCry = 0;
			int cardWeaponCount = 0;
			int cardWeaponMana = 0;
			int cardSpellCount = 0;
			int cardSpellMana = 0;
			int cardHardRemoval = 0;
			for (Card card : player.getHand()) {
				if (card.getCardType() == CardType.MINION) {  // 随从牌
					cardMinionCount += 1;
					cardMinionMana += card.getBaseManaCost();
					if (card.hasBattlecry()) {
						cardMinionBattleCry += 1;  // 这个似乎一直是0，可能没用
					}
				} else if(card.getCardType() == CardType.WEAPON){  // 武器牌
					cardWeaponCount += 1;
					cardWeaponMana += card.getBaseManaCost();
				} else{  // 剩下的应该就是Spell法术牌了，但貌似也有另外几个其他的, 不区分
					cardSpellCount += 1;
					cardSpellMana += card.getBaseManaCost();
				}

				if (Player.isHardRemoval(card)) {
					cardHardRemoval += 1;
				}
			}

			builder.append("|" + cardMinionCount + "|" + cardMinionMana + "|" + cardMinionBattleCry);
			builder.append("|" + cardWeaponCount + "|" + cardWeaponMana);
			builder.append("|" + cardSpellCount + "|" + cardSpellMana + "|" + cardHardRemoval);

			builder.append("'");
		}
		builder.append('}');
		return builder.toString();
	}

	public String feature_1(){
		StringBuilder builder = new StringBuilder("{'GameHash':" + hashCode() + ",'Turn':" + getTurn());

		for (Player player : players){
			builder.append(",'player" + player.getId() + "':'");
			builder.append(player.getHero().getHp());  // 血量
			builder.append("|" + player.getMana());  // 当前法力值
			builder.append("|" + player.getHero().getArmor()); // 护甲

			// 场上的随从相关数据
			int summonCount = 0;   // minions on board that can still attack (直观来说，一回合结束时，自己场上应该不会再有能攻击的随从还没用的情况)
			int summonAttack = 0;
			int summonHp = 0;
			int summonCountNot = 0; // minions on board that can not attack
			int summonAttackNot = 0;
			int summonHpNot = 0;
			for (Summon summon : player.getSummons()) {   // 场上的随从信息, 暂时只考虑攻击力和血量，跑通流程，各种特殊效果后面补充
				if (summon.canAttackThisTurn()) {
					summonCount += 1;
					summonAttack += summon.getAttack();
					summonHp += summon.getHp();
				} else {
					summonCountNot += 1;
					summonAttackNot += summon.getAttack();
					summonHpNot += summon.getHp();
				}
			}
			builder.append("|" + summonCount + "|" + summonAttack + "|" + summonHp);
			builder.append("|" + summonCountNot + "|" + summonAttackNot + "|" + summonHpNot);

			// 手牌相关信息
			int cardMinionCount = 0;
			int cardMinionBattleCry = 0;
			int cardSpellCount = 0;
			int cardSpellMana = 0;
			for (Card card : player.getHand()) {
				if (card.getCardType() == CardType.MINION) {
					cardMinionCount += 1;
					if (card.hasBattlecry()) {
						cardMinionBattleCry += 1;
					}
				} else {  // 除了Spell法术牌以外，其实还有 CHOOSE_ONE 等其他手牌类型，但目前暂时不考虑
					cardSpellCount += 1;
					cardSpellMana += card.getBaseManaCost();
				}
			}
			builder.append("|" + cardMinionCount + "|" + cardMinionBattleCry); // change by fh
			builder.append("|" + cardSpellCount + "|" + cardSpellMana);

			builder.append("'");
		}
		builder.append('}');
		return builder.toString();
	}

	public String contextInfoStr(){
		// 将我们希望提取的环境信息表示成格式化字符串
		StringBuilder builder = new StringBuilder("{'GameHash':" + hashCode() + ",'Turn':" + getTurn());

		for (Player player : players){
			builder.append(",'player" + player.getId() + "':'");
			builder.append(player.getHero().getHp());  // 血量
			builder.append("|" + player.getMana());  // 当前法力值
			builder.append("|" + player.getMaxMana()); // 当前最大法力值
			builder.append("|" + player.getHero().getArmor()); // 护甲

			// 场上的随从相关数据
			int summonCount = 0;   // minions on board that can still attack (直观来说，一回合结束时，自己场上应该不会再有能攻击的随从还没用的情况)
			int summonAttack = 0;
			int summonHp = 0;
			int summonCountNot = 0; // minions on board that can not attack
			int summonAttackNot = 0;
			int summonHpNot = 0;
			for (Summon summon : player.getSummons()) {   // 场上的随从信息, 暂时只考虑攻击力和血量，跑通流程，各种特殊效果后面补充
				if (summon.canAttackThisTurn()) {
					summonCount += 1;
					summonAttack += summon.getAttack();
					summonHp += summon.getHp();
				} else {
					summonCountNot += 1;
					summonAttackNot += summon.getAttack();
					summonHpNot += summon.getHp();
				}
			}
			builder.append("|" + summonCount + "|" + summonAttack + "|" + summonHp);
			builder.append("|" + summonCountNot + "|" + summonAttackNot + "|" + summonHpNot);

			// 手牌相关信息
			int cardMinionCount = 0;
			int cardMinionMana = 0;
			int cardMinionBattleCry = 0;
			int cardSpellCount = 0;
			int cardSpellMana = 0;
			for (Card card : player.getHand()) {
				if (card.getCardType() == CardType.MINION) {
					cardMinionCount += 1;
					cardMinionMana += card.getBaseManaCost();
					if (card.hasBattlecry()) {
						cardMinionBattleCry += 1;
					}
				} else {  // 除了Spell法术牌以外，其实还有 CHOOSE_ONE 等其他手牌类型，但目前暂时不考虑
					cardSpellCount += 1;
					cardSpellMana += card.getBaseManaCost();
				}
			}
			builder.append("|" + cardMinionCount + "|" + cardMinionMana + "|" + cardMinionBattleCry);
			//builder.append("|" + cardMinionCount + "|" + cardMinionBattleCry); // change by fh
			builder.append("|" + cardSpellCount + "|" + cardSpellMana);

			builder.append("'");
		}
		builder.append('}');
		return builder.toString();
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder("GameContext hashCode: " + hashCode() + "\nPlayer: ");
		for (Player player : players) {
			builder.append(player.getName());
			builder.append(" Mana: ");
			builder.append(player.getMana());
			builder.append('/');
			builder.append(player.getMaxMana());
			builder.append(" HP: ");
			builder.append(player.getHero().getHp() + "(" + player.getHero().getArmor() + ")");  // 血量和护甲
			builder.append('\n');
			builder.append("Behaviour: " + player.getBehaviour().getName() + "\n");
			builder.append("Minions:\n");
			for (Summon summon : player.getSummons()) {   // 场上的随从
				builder.append('\t');
				builder.append(summon);
				builder.append('\n');
			}
			builder.append("Cards (hand):\n");   // 手里的牌
			for (Card card : player.getHand()) {
				builder.append('\t');
				builder.append(card);
				builder.append('\n');
			}
			builder.append("Secrets:\n");   // 秘密
			for (String secretId : player.getSecrets()) {
				builder.append('\t');
				builder.append(secretId);
				builder.append('\n');
			}
		}
		builder.append("Turn: " + getTurn() + "\n");
		builder.append("Result: " + result + "\n");
		builder.append("Winner: " + (winner == null ? "tbd" : winner.getName()));

		return builder.toString();
	}

	public Entity tryFind(EntityReference targetKey) {
		try {
			return resolveSingleTarget(targetKey);
		} catch (Exception e) {
		}
		return null;
	}
}
