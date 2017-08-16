package net.demilich.metastone.game.behaviour.heuristic;

import net.demilich.metastone.game.Attribute;
import net.demilich.metastone.game.GameContext;
import net.demilich.metastone.game.Player;
import net.demilich.metastone.game.entities.minions.Minion;

public class WeightedHeuristic implements IGameStateHeuristic {

	private float calculateMinionScore(Minion minion) {  //计算小兵的分数（Heuristic）
		float minionScore = minion.getAttack() + minion.getHp();  // 攻击 + 血量
		float baseScore = minionScore;
		if (minion.hasAttribute(Attribute.FROZEN)) {  // 如果冻结中，返回血量作为得分
			return minion.getHp();
		}
		if (minion.hasAttribute(Attribute.TAUNT)) {   // 嘲讽技能，得分 +2
			minionScore += 2;
		}
		if (minion.hasAttribute(Attribute.WINDFURY)) { //风怒 攻击两次
			minionScore += minion.getAttack() * 0.5f;
		}
		if (minion.hasAttribute(Attribute.DIVINE_SHIELD)) {  //圣盾
			minionScore += 1.5f * baseScore;
		}
		if (minion.hasAttribute(Attribute.SPELL_DAMAGE)) {  // 法术伤害
			minionScore += minion.getAttributeValue(Attribute.SPELL_DAMAGE);
		}
		if (minion.hasAttribute(Attribute.ENRAGED)) { //激怒，不是满血
			minionScore += 1;
		}
		if (minion.hasAttribute(Attribute.STEALTH)) { // 潜行
			minionScore += 1;
		}
		if (minion.hasAttribute(Attribute.UNTARGETABLE_BY_SPELLS)) {  // 不能被法术指定
			minionScore += 1.5f * baseScore;
		}

		return minionScore;
	}

	@Override
	public double getScore(GameContext context, int playerId) {
		float score = 0;
		Player player = context.getPlayer(playerId);
		Player opponent = context.getOpponent(player);
		if (player.getHero().isDestroyed()) {   // 己方被干掉，得分 负无穷
			return Float.NEGATIVE_INFINITY;
		}
		if (opponent.getHero().isDestroyed()) {  // 对方被干掉，得分 正无穷
			return Float.POSITIVE_INFINITY;
		}
		int ownHp = player.getHero().getHp() + player.getHero().getArmor();  // 自身血量 + 装甲
		int opponentHp = opponent.getHero().getHp() + opponent.getHero().getArmor();  // 对方血量 + 装甲
		score += ownHp - opponentHp;  // 己方Hp - 对方Hp 作为基础得分

		score += player.getHand().getCount() * 3;  // 得分 + 自身手牌数*3
		score -= opponent.getHand().getCount() * 3; // 得分 - 对方手牌数*3
		score += player.getMinions().size() * 2;  // 得分 + 自身小兵数*2
		score -= opponent.getMinions().size() * 2; // 得分 - 对方小兵数*2
		for (Minion minion : player.getMinions()) {  // 得分 + 己方每个小兵的评分
			score += calculateMinionScore(minion);
		}
		for (Minion minion : opponent.getMinions()) {  // 得分 - 对方每个小兵的评分
			score -= calculateMinionScore(minion);
		}

		return score;
	}

	@Override
	public void onActionSelected(GameContext context, int playerId) {
	}

}
