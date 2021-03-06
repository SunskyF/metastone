package net.demilich.metastone.game;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;

import net.demilich.metastone.game.behaviour.IBehaviour;
import net.demilich.metastone.game.behaviour.human.HumanBehaviour;
import net.demilich.metastone.game.cards.Card;
import net.demilich.metastone.game.cards.CardCollection;
import net.demilich.metastone.game.cards.CardType;
import net.demilich.metastone.game.cards.Rarity;
import net.demilich.metastone.game.decks.Deck;
import net.demilich.metastone.game.entities.Actor;
import net.demilich.metastone.game.entities.Entity;
import net.demilich.metastone.game.entities.EntityType;
import net.demilich.metastone.game.entities.heroes.Hero;
import net.demilich.metastone.game.entities.heroes.HeroClass;
import net.demilich.metastone.game.entities.minions.Minion;
import net.demilich.metastone.game.entities.minions.Summon;
import net.demilich.metastone.game.statistics.GameStatistics;
import net.demilich.metastone.game.gameconfig.PlayerConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Attr;

public class Player extends Entity {

	private Hero hero;
	private final String deckName;

	private final CardCollection deck;
	private final CardCollection hand = new CardCollection();
	private final List<Entity> setAsideZone = new ArrayList<>();
	private final List<Entity> graveyard = new ArrayList<>();
	private final List<Summon> summons = new ArrayList<>();
	private final HashSet<String> secrets = new HashSet<>();
	private final HashSet<String> quests = new HashSet<>();

	private final GameStatistics statistics = new GameStatistics();

	private int mana;
	private int maxMana;
	private int lockedMana;

	private boolean hideCards;

	private IBehaviour behaviour;
	private static final Logger logger = LoggerFactory.getLogger(GameContext.class);

	private static List<String> hardRemoval;
	static {
		hardRemoval = new ArrayList<String>();
		hardRemoval.add("spell_polymorph");
		hardRemoval.add("spell_execute");
		hardRemoval.add("spell_crush");
		hardRemoval.add("spell_assassinate");
		hardRemoval.add("spell_siphon_soul");
		hardRemoval.add("spell_shadow_word_death");
		hardRemoval.add("spell_naturalize");
		hardRemoval.add("spell_hex");
		hardRemoval.add("spell_humility");
		hardRemoval.add("spell_equality");
		hardRemoval.add("spell_deadly_shot");
		hardRemoval.add("spell_sap");
		hardRemoval.add("minion_doomsayer");
		hardRemoval.add("minion_big_game_hunter");
	}

	private Player(Player otherPlayer) {
		this.setName(otherPlayer.getName());
		this.deckName = otherPlayer.getDeckName();
		this.setHero(otherPlayer.getHero().clone());
		this.deck = otherPlayer.getDeck().clone();
		this.attributes.putAll(otherPlayer.getAttributes());
		this.hand.addAll(otherPlayer.getHand().clone());
		this.summons.addAll(otherPlayer.getSummons().stream().map(Summon::clone).collect(Collectors.toList()));
		this.graveyard.addAll(otherPlayer.getGraveyard().stream().map(Entity::clone).collect(Collectors.toList()));
		this.setAsideZone.addAll(otherPlayer.getSetAsideZone().stream().map(Entity::clone).collect(Collectors.toList()));
		this.secrets.addAll(otherPlayer.secrets);
		this.quests.addAll(otherPlayer.quests);
		this.setId(otherPlayer.getId());
		this.mana = otherPlayer.mana;
		this.maxMana = otherPlayer.maxMana;
		this.lockedMana = otherPlayer.lockedMana;
		this.behaviour = otherPlayer.behaviour;
		this.getStatistics().merge(otherPlayer.getStatistics());
	}

	public Player(PlayerConfig config) {
		config.build();
		Deck selectedDeck = config.getDeckForPlay();
		this.deck = selectedDeck.getCardsCopy();
		this.setHero(config.getHeroForPlay().createHero());
		this.setName(config.getName() + " - " + hero.getName());
		this.deckName = selectedDeck.getName();
		setBehaviour(config.getBehaviour().clone());
		setHideCards(config.hideCards());
	}

	@Override
	public Player clone() {
		return new Player(this);
	}

	public IBehaviour getBehaviour() {
		return behaviour;
	}

	public List<Actor> getCharacters() {
		List<Actor> characters = new ArrayList<Actor>();
		characters.add(getHero());
		characters.addAll(getMinions());
		return characters;
	}

	public CardCollection getDeck() {
		return deck;
	}

	public String getDeckName() {
		return deckName;
	}

	@Override
	public EntityType getEntityType() {
		return EntityType.PLAYER;
	}

	public List<Entity> getGraveyard() {
		return graveyard;
	}

	public CardCollection getHand() {
		return hand;
	}

	public Hero getHero() {
		return hero;
	}

	public int getLockedMana() {
		return lockedMana;
	}

	public int getMana() {
		return mana;
	}

	public int getMaxMana() {
		return maxMana;
	}

	public List<Minion> getMinions() {
		List<Minion> minions = new ArrayList<Minion>();
		for (Summon summon : getSummons()) {
			if (summon instanceof Minion) {
				minions.add((Minion) summon);
			}
		}
		return minions;
	}

	public HashSet<String> getQuests() {
		return quests;
	}

	public List<Summon> getSummons() {
		return summons;
	}

	public HashSet<String> getSecrets() {
		return secrets;
	}
	
	public List<Entity> getSetAsideZone() {
		return setAsideZone;
	}

	public GameStatistics getStatistics() {
		return statistics;
	}

	public static boolean isHardRemoval(Card card) {
		return hardRemoval.contains(card.getCardId());
	}

	public List<Integer> getPlayerStateBasic(){
		List<Integer> playerState = new ArrayList<Integer>();

		playerState.add(this.getHero().getHp() / 30);  // 血量
		playerState.add(this.getMana() / 10);  // 当前法力值
		playerState.add(this.getMaxMana() / 10);   // 当前最大法力值
		playerState.add(this.getHero().getArmor() / 10); // 护甲

		// 场上的随从相关数据
		int summonCount = 0;   // minions on board that can still attack (直观来说，一回合结束时，自己场上应该不会再有能攻击的随从还没用的情况)
		int summonAttack = 0;
		int summonHp = 0;
		int summonCountNot = 0; // minions on board that can not attack
		int summonAttackNot = 0;
		int summonHpNot = 0;
		for (Summon summon : this.getSummons()) {   // 场上的随从信息, 暂时只考虑攻击力和血量，跑通流程，各种特殊效果后面补充
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
		playerState.addAll(Arrays.asList(summonCount / 7, summonAttack / 70, summonHp / 70, summonCountNot / 7, summonAttackNot / 70, summonHpNot / 70));

		// 手牌相关信息
		int cardMinionCount = 0;
		int cardMinionMana = 0;
		int cardMinionBattleCry = 0;
		int cardSpellCount = 0;
		int cardSpellMana = 0;
		for (Card card : this.getHand()) {
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
		playerState.addAll(Arrays.asList(cardMinionCount / 8, cardMinionMana / 50, cardMinionBattleCry / 8, cardSpellCount / 8, cardSpellMana / 80));
		return playerState;
	}

	public List<Double> getPlayerStatefh0(boolean opposite){// 是不是对手，false不是
		List<Double> playerState = new ArrayList<>();

		playerState.add(this.getHero().getHp() / 1.0);  // 0. 血量
		playerState.add(this.getMana() / 1.0);  // 1. 当前法力值
		playerState.add(this.getMaxMana() / 1.0);   // 2. 当前最大法力值
		playerState.add(this.getHero().getArmor() / 1.0); // 3. 护甲

		int weaponDamage = 0;
		int weaponDurability = 0;
		if (this.getHero().getWeapon() != null) {
			weaponDamage = this.getHero().getWeapon().getWeaponDamage();
			weaponDurability = this.getHero().getWeapon().getDurability();
		}
		playerState.add(weaponDamage / 1.0); // 4. 武器伤害
		playerState.add(weaponDurability / 1.0); // 5. 武器耐久

		// 场上的随从相关数据
		int summonCount = 0;  // 6. // minions on board that can still attack (直观来说，一回合结束时，自己场上应该不会再有能攻击的随从还没用的情况)
		int summonAttack = 0; // 7.
		int summonHp = 0; // 8.
		int summonCountNot = 0; // 9. // minions on board that can not attack
		int summonAttackNot = 0; // 10.
		int summonHpNot = 0; // 11.

		for (Summon summon : this.getSummons()) {   // 场上的随从信息, 暂时只考虑攻击力和血量，跑通流程，各种特殊效果后面补充
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
		playerState.addAll(Arrays.asList(summonCount / 1.0, summonAttack / 1.0, summonHp / 1.0, summonCountNot / 1.0,
				summonAttackNot / 1.0, summonHpNot / 1.0));
		//playerState.addAll(Arrays.asList(summonManaLow, summonManaMid, summonManaHigh));

		// 手牌相关信息
		if (!opposite){// 如果不是对手
			int cardMinionCount = 0; // 12.
			int cardMinionMana = 0; // 13.
			int cardSpellCount = 0; // 14.
			int cardSpellMana = 0; // 15.

			for (Card card : this.getHand()) {

				if (card.getCardType() == CardType.MINION) {
					cardMinionCount += 1;
					cardMinionMana += card.getBaseManaCost();
				} else {  // 除了Spell法术牌以外，其实还有 CHOOSE_ONE 等其他手牌类型，但目前暂时不考虑
					cardSpellCount += 1;
					cardSpellMana += card.getBaseManaCost();
				}
			}

			playerState.addAll(Arrays.asList(cardMinionCount / 1.0, cardMinionMana / 1.0,
					cardSpellCount / 1.0, cardSpellMana / 1.0)); // 4.0 10.0 4.0  10.0
		}

		return playerState;
	}

	public List<Integer> getPlayerStatefh1(boolean opposite){
		List<Integer> playerState = new ArrayList<Integer>();

		// 英雄相关信息
		playerState.add(this.getHero().getHp());  // 血量
		playerState.add(this.getHero().getArmor()); // 护甲
		playerState.add(this.getMana());  // 当前法力值
		playerState.add(this.getDeck().getCount()); // 当前卡组剩余牌数
		playerState.add(this.getGraveyard().size()); // 墓地卡数目

		int weaponDamage = 0;
		int weaponDurability = 0;
		if (this.getHero().getWeapon() != null) {
			weaponDamage = this.getHero().getWeapon().getWeaponDamage();  //武器伤害
			weaponDurability = this.getHero().getWeapon().getDurability(); //武器耐久
		}
		playerState.add(weaponDamage);
		playerState.add(weaponDurability);
		// 英雄技能, 暂时按照英雄类型将技能分为1、2、3三档，暂时不考虑一些非基础的英雄技能的影响
		int heroPower = 3; // 默认为3
		HeroClass heroPowerClass = this.getHero().getHeroPower().getHeroClass();
		if(heroPowerClass == HeroClass.HUNTER || heroPowerClass == HeroClass.MAGE || heroPowerClass == HeroClass.WARLOCK){
			heroPower = 3;
		}else if(heroPowerClass == HeroClass.DRUID || heroPowerClass == HeroClass.PALADIN || heroPowerClass == HeroClass.SHAMAN){
			heroPower = 2;
		}else if(heroPowerClass == HeroClass.PRIEST || heroPowerClass == HeroClass.ROGUE || heroPowerClass == HeroClass.WARRIOR){
			heroPower = 1;
		}
		playerState.add(heroPower);

		// 场上的随从相关数据
		int minionCount = 0;   // minions on board that can still attack (直观来说，一回合结束时，自己场上应该不会再有能攻击的随从还没用的情况)
		int minionAttack = 0;
		int minionHp = 0;
		int minionCost = 0;
		int minionCountNot = 0; // minions on board that can not attack
		int minionAttackNot = 0;
		int minionHpNot = 0;
		int minionCostNot = 0;
		int minionCountTaunt = 0;  // 带嘲讽的随从
		int minionAttackTaunt= 0;
		int minionCostTaunt = 0;
		int minionHpTaunt = 0;
		int minionCountFrozen = 0;  // 冻结的随从
		int minionAttackFrozen= 0;
		int minionCostFrozen = 0;
		int minionHpFrozen = 0;
		int minionCountStealth = 0;  // 潜行的随从
		int minionAttackStealth= 0;
		int minionCostStealth = 0;
		int minionHpStealth = 0;
		int minionCountShield = 0;  // 带圣盾的随从
		int minionAttackShield= 0;
		int minionCostShield = 0;
		int minionHpShield = 0;
		int minionCountEnrage = 0;  // 带激怒的随从
		int minionAttackEnrage= 0;
		int minionCostEnrage = 0;
		int minionHpEnrage = 0;
		int minionCountUntarget = 0;  // 不可被法术攻击的随从
		int minionAttackUntarget= 0;
		int minionCostUntarget = 0;
		int minionHpUntarget = 0;
		int minionCountWindfury = 0; // 带风怒效果的随从
		int minionAttackWindfury = 0;
		int minionCostWindfury = 0;
		int minionHpWindfury = 0;
		int minionCountSpell = 0;  // 带法术伤害的随从
		int minionSpellDamage = 0;
		int minionCostSpell = 0;
		int minionManaMax = 0;
		int minionManaMin = 100;

		for (Minion minion : this.getMinions()) {  // 场上的随从信息
			int mana_cost = minion.getAttributeValue(Attribute.BASE_MANA_COST);
			if (mana_cost > minionManaMax)
				minionManaMax = mana_cost;
			if (mana_cost < minionManaMin)
				minionManaMin = mana_cost;

			if (minion.canAttackThisTurn()) {
				minionCount += 1;
				minionAttack += minion.getAttack();
				minionHp += minion.getHp();
				minionCost += minion.getAttributeValue(Attribute.BASE_MANA_COST);
			} else {
				minionCountNot += 1;
				minionAttackNot += minion.getAttack();
				minionHpNot += minion.getHp();
				minionCostNot += minion.getAttributeValue(Attribute.BASE_MANA_COST);
			}
			if(minion.hasAttribute(Attribute.TAUNT)){
				minionCountTaunt += 1;
				minionAttackTaunt += minion.getAttack();
				minionHpTaunt += minion.getHp();
				minionCostTaunt += minion.getAttributeValue(Attribute.BASE_MANA_COST);
			}
			if (minion.hasAttribute(Attribute.FROZEN)) {  // 冻结的随从
				minionCountFrozen += 1;
				minionAttackFrozen += minion.getAttack();
				minionHpFrozen += minion.getHp();
				minionCostFrozen += minion.getAttributeValue(Attribute.BASE_MANA_COST);
			}
			if (minion.hasAttribute(Attribute.STEALTH)) {  // 潜行
				minionCountStealth += 1;
				minionAttackStealth += minion.getAttack();
				minionHpStealth += minion.getHp();
				minionCostStealth += minion.getAttributeValue(Attribute.BASE_MANA_COST);
			}
			if (minion.hasAttribute(Attribute.DIVINE_SHIELD)) {  //圣盾
				minionCountShield += 1;
				minionAttackShield += minion.getAttack();
				minionHpShield += minion.getHp();
				minionCostShield += minion.getAttributeValue(Attribute.BASE_MANA_COST);
			}
			if (minion.hasAttribute(Attribute.ENRAGED)) {  // 激怒
				minionCountEnrage += 1;
				minionAttackEnrage += minion.getAttack();
				minionHpEnrage += minion.getHp();
				minionCostEnrage += minion.getAttributeValue(Attribute.BASE_MANA_COST);
			}
			if (minion.hasAttribute(Attribute.UNTARGETABLE_BY_SPELLS)) {  // 不能被法术指定
				minionCountUntarget += 1;
				minionAttackUntarget += minion.getAttack();
				minionHpUntarget += minion.getHp();
				minionCostUntarget += minion.getAttributeValue(Attribute.BASE_MANA_COST);
			}
			if (minion.hasAttribute(Attribute.WINDFURY) || minion.hasAttribute(Attribute.MEGA_WINDFURY)) {  // 风怒或超级风怒
				minionCountWindfury += 1;
				minionHpWindfury += minion.getHp();
				if (minion.hasAttribute(Attribute.MEGA_WINDFURY)){  // 风怒或超级风怒带来的额外的攻击力
					minionAttackWindfury += 3*minion.getAttack();
				}else{
					minionAttackWindfury += minion.getAttack();
				}
				minionCostWindfury += minion.getAttributeValue(Attribute.BASE_MANA_COST);
			}
			if (minion.hasAttribute(Attribute.SPELL_DAMAGE)) {  // 法术伤害
				minionCountSpell += 1;
				minionSpellDamage += minion.getAttributeValue(Attribute.SPELL_DAMAGE);
				minionCostSpell += minion.getAttributeValue(Attribute.BASE_MANA_COST);
			}
		}
		playerState.addAll(Arrays.asList(minionCount, minionAttack, minionHp, minionCountNot, minionAttackNot, minionHpNot,
				minionCountTaunt, minionAttackTaunt, minionHpTaunt,
				minionCountFrozen, minionAttackFrozen, minionHpFrozen,
				minionCountStealth, minionAttackStealth, minionHpStealth,
				minionCountShield, minionAttackShield, minionHpShield,
				minionCountEnrage, minionAttackEnrage, minionHpEnrage,
				minionCountUntarget, minionAttackUntarget, minionHpUntarget,
				minionCountWindfury, minionAttackWindfury, minionHpWindfury,
				minionCountSpell, minionSpellDamage,
				minionCost, minionCostNot, minionCostTaunt, minionCostFrozen, minionCostStealth,
				minionCostShield, minionCostEnrage, minionCostUntarget, minionCostWindfury, minionCostSpell));

		if (!opposite){
			// 手牌相关信息
			int cardMinionCount = 0;
			int cardWeaponCount = 0;
			int cardSpellCount = 0;
			int cardSpellManaLow = 0;
			int cardSpellManaMid = 0;
			int cardSpellManaHigh = 0;
			int cardMinionManaLow = 0;
			int cardMinionManaMid = 0;
			int cardMinionManaHigh = 0;
			int cardWeaponManaLow = 0;
			int cardWeaponManaMid = 0;
			int cardWeaponManaHigh = 0;
			int cardHardRemoval = 0;
			for (Card card : this.getHand()) {
				if (card.getCardType() == CardType.MINION) {  // 随从牌
					cardMinionCount += card.getBaseManaCost();
					if (card.getBaseManaCost() < 4)
						cardMinionManaLow += 1;
					else if (card.getBaseManaCost() < 7)
						cardMinionManaMid += 1;
					else
						cardMinionManaHigh += 1;
				} else if(card.getCardType() == CardType.WEAPON){  // 武器牌
					cardWeaponCount += card.getBaseManaCost();
					if (card.getBaseManaCost() < 4)
						cardWeaponManaLow += 1;
					else if (card.getBaseManaCost() < 7)
						cardWeaponManaMid += 1;
					else
						cardWeaponManaHigh += 1;
				} else{  // 剩下的应该就是Spell法术牌了，但貌似也有另外几个其他的, 不区分
					cardSpellCount += card.getBaseManaCost();
					if (card.getBaseManaCost() < 4)
						cardSpellManaLow += 1;
					else if (card.getBaseManaCost() < 7)
						cardSpellManaMid += 1;
					else
						cardSpellManaHigh += 1;
				}

				if (isHardRemoval(card)) {
					cardHardRemoval += 1;
				}
			}
			playerState.addAll(Arrays.asList(cardMinionCount, cardWeaponCount, cardSpellCount, cardHardRemoval));
			playerState.addAll(Arrays.asList(cardSpellManaLow, cardSpellManaMid, cardSpellManaHigh));
			playerState.addAll(Arrays.asList(cardMinionManaLow, cardMinionManaMid, cardMinionManaHigh));
			playerState.addAll(Arrays.asList(cardWeaponManaLow, cardWeaponManaMid, cardWeaponManaHigh));

		}
		return playerState;
	}

	public List<Double> getPlayerStateDouble(){
		List<Double> playerState = new ArrayList<>();

		// 英雄相关信息
		playerState.add(this.getHero().getHp() / 1.0);  // 0. 血量
		playerState.add(this.getHero().getArmor() / 1.0); // 1. 护甲
		playerState.add(this.getMana() / 1.0);  // 2. 当前法力值
		int weaponDamage = 0;
		int weaponDurability = 0;
		if (this.getHero().getWeapon() != null) {
			weaponDamage = this.getHero().getWeapon().getWeaponDamage();  //3. 武器伤害
			weaponDurability = this.getHero().getWeapon().getDurability(); //4. 武器耐久
		}
		playerState.add(weaponDamage / 1.0);
		playerState.add(weaponDurability / 1.0);
		// 英雄技能, 暂时按照英雄类型将技能分为1、2、3三档，暂时不考虑一些非基础的英雄技能的影响
		int heroPower = 3; // 默认为3
		HeroClass heroPowerClass = this.getHero().getHeroPower().getHeroClass();
		if(heroPowerClass == HeroClass.HUNTER || heroPowerClass == HeroClass.MAGE || heroPowerClass == HeroClass.WARLOCK){
			heroPower = 3;
		}else if(heroPowerClass == HeroClass.DRUID || heroPowerClass == HeroClass.PALADIN || heroPowerClass == HeroClass.SHAMAN){
			heroPower = 2;
		}else if(heroPowerClass == HeroClass.PRIEST || heroPowerClass == HeroClass.ROGUE || heroPowerClass == HeroClass.WARRIOR){
			heroPower = 1;
		}
		playerState.add(heroPower / 1.0); // 5. 英雄技能

		// 场上的随从相关数据
		int minionCount = 0;   // 6. 随从可攻击数目 (直观来说，一回合结束时，自己场上应该不会再有能攻击的随从还没用的情况)
		int minionAttack = 0;  // 7. 随从攻击力
		int minionHp = 0;  // 8. 随从血量
		int minionCountNot = 0; // 9. 随从不可攻击数目
		int minionAttackNot = 0; // 10. 不可攻击随从攻击力
		int minionHpNot = 0; // 11. 不可攻击随从血量
		int minionCountTaunt = 0;  // 12. 带嘲讽的随从数目
		int minionAttackTaunt= 0;  // 13.
		int minionHpTaunt = 0;  // 14
		int minionCountFrozen = 0;  // 15 冻结的随从
		int minionAttackFrozen= 0;  // 16
		int minionHpFrozen = 0; // 17
		int minionCountStealth = 0;  // 18 潜行的随从
		int minionAttackStealth= 0;  // 19
		int minionHpStealth = 0;  // 20
		int minionCountShield = 0;  // 21 带圣盾的随从
		int minionAttackShield= 0;  // 22
		int minionHpShield = 0;  // 23
		int minionCountEnrage = 0;  // 24 带激怒的随从
		int minionAttackEnrage= 0;  // 25
		int minionHpEnrage = 0;  // 26
		int minionCountUntarget = 0;  // 27 不可被法术攻击的随从
		int minionAttackUntarget= 0;  // 28
		int minionHpUntarget = 0;  // 29
		int minionCountWindfury = 0; // 30 带风怒效果的随从
		int minionAttackWindfury = 0; // 31
		int minionHpWindfury = 0;  // 32
		int minionCountSpell = 0;  // 33 带法术伤害的随从
		int minionSpellDamage = 0;  // 34

		for (Minion minion : this.getMinions()) {  // 场上的随从信息
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
		playerState.addAll(Arrays.asList(minionCount / 1.0, minionAttack / 1.0, minionHp / 1.0, minionCountNot / 1.0, minionAttackNot / 1.0, minionHpNot / 1.0,
				minionCountTaunt / 1.0, minionAttackTaunt / 1.0, minionHpTaunt / 1.0,
				minionCountFrozen / 1.0, minionAttackFrozen / 1.0, minionHpFrozen / 1.0,
				minionCountStealth / 1.0, minionAttackStealth / 1.0, minionHpStealth / 1.0,
				minionCountShield / 1.0, minionAttackShield / 1.0, minionHpShield / 1.0,
				minionCountEnrage / 1.0, minionAttackEnrage / 1.0, minionHpEnrage / 1.0,
				minionCountUntarget / 1.0, minionAttackUntarget / 1.0, minionHpUntarget / 1.0,
				minionCountWindfury / 1.0, minionAttackWindfury / 1.0, minionHpWindfury / 1.0,
				minionCountSpell / 1.0, minionSpellDamage / 1.0));

		// 手牌相关信息
		int cardMinionCount = 0; // 35
		int cardMinionMana = 0;  // 36
		int cardMinionBattleCry = 0;  // 37
		int cardWeaponCount = 0;  // 38
		int cardWeaponMana = 0;  // 39
		int cardSpellCount = 0;  // 40
		int cardSpellMana = 0;  // 41
		int cardHardRemoval = 0;  // 42
		for (Card card : this.getHand()) {
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

			if (isHardRemoval(card)) {
				cardHardRemoval += 1;
			}
		}
		playerState.addAll(Arrays.asList(cardMinionCount / 1.0, cardMinionMana / 1.0, cardMinionBattleCry / 1.0, cardWeaponCount / 1.0, cardWeaponMana / 1.0, cardSpellCount / 1.0, cardSpellMana / 1.0, cardHardRemoval / 1.0));
		return playerState;
	}

	public List<Integer> getPlayerState(){
		List<Integer> playerState = new ArrayList<Integer>();

		// 英雄相关信息
		playerState.add(this.getHero().getHp());  // 血量
		playerState.add(this.getHero().getArmor()); // 护甲
		playerState.add(this.getMana());  // 当前法力值
		int weaponDamage = 0;
		int weaponDurability = 0;
		if (this.getHero().getWeapon() != null) {
			weaponDamage = this.getHero().getWeapon().getWeaponDamage();  //武器伤害
			weaponDurability = this.getHero().getWeapon().getDurability(); //武器耐久
		}
		playerState.add(weaponDamage);
		playerState.add(weaponDurability);
		// 英雄技能, 暂时按照英雄类型将技能分为1、2、3三档，暂时不考虑一些非基础的英雄技能的影响
		int heroPower = 3; // 默认为3
		HeroClass heroPowerClass = this.getHero().getHeroPower().getHeroClass();
		if(heroPowerClass == HeroClass.HUNTER || heroPowerClass == HeroClass.MAGE || heroPowerClass == HeroClass.WARLOCK){
			heroPower = 3;
		}else if(heroPowerClass == HeroClass.DRUID || heroPowerClass == HeroClass.PALADIN || heroPowerClass == HeroClass.SHAMAN){
			heroPower = 2;
		}else if(heroPowerClass == HeroClass.PRIEST || heroPowerClass == HeroClass.ROGUE || heroPowerClass == HeroClass.WARRIOR){
			heroPower = 1;
		}
		playerState.add(heroPower);

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

		for (Minion minion : this.getMinions()) {  // 场上的随从信息
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
		playerState.addAll(Arrays.asList(minionCount, minionAttack, minionHp, minionCountNot, minionAttackNot, minionHpNot,
				minionCountTaunt, minionAttackTaunt, minionHpTaunt,
				minionCountFrozen, minionAttackFrozen, minionHpFrozen,
				minionCountStealth, minionAttackStealth, minionHpStealth,
				minionCountShield, minionAttackShield, minionHpShield,
				minionCountEnrage, minionAttackEnrage, minionHpEnrage,
				minionCountUntarget, minionAttackUntarget, minionHpUntarget,
				minionCountWindfury, minionAttackWindfury, minionHpWindfury,
				minionCountSpell, minionSpellDamage));

		// 手牌相关信息
		int cardMinionCount = 0;
		int cardMinionMana = 0;
		int cardMinionBattleCry = 0;
		int cardWeaponCount = 0;
		int cardWeaponMana = 0;
		int cardSpellCount = 0;
		int cardSpellMana = 0;
		int cardHardRemoval = 0;
		for (Card card : this.getHand()) {
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

			if (isHardRemoval(card)) {
				cardHardRemoval += 1;
			}
		}
		playerState.addAll(Arrays.asList(cardMinionCount, cardMinionMana, cardMinionBattleCry, cardWeaponCount, cardWeaponMana, cardSpellCount, cardSpellMana, cardHardRemoval));
		return playerState;
	}

	public boolean hideCards() {
		return hideCards && !(behaviour instanceof HumanBehaviour);
	}

	public void setBehaviour(IBehaviour behaviour) {
		this.behaviour = behaviour;
	}

	public void setHero(Hero hero) {
		this.hero = hero;
	}

	public void setHideCards(boolean hideCards) {
		this.hideCards = hideCards;
	}

	public void setLockedMana(int lockedMana) {
		this.lockedMana = lockedMana;
	}

	public void setMana(int mana) {
		this.mana = mana;
	}

	public void setMaxMana(int maxMana) {
		this.maxMana = maxMana;
	}

	@Override
	public String toString() {
		return "[PLAYER " + "id: " + getId() + ", name: " + getName() + ", hero: " + getHero() + "]";
	}

}
