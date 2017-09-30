import json
from collections import defaultdict

cardsNum = defaultdict(lambda: 0)

with open("wild_pirate_warrior.json", "r") as f:
    deck = json.load(f)
    cards = deck['cards']
    

    for card in cards:
        cardsNum[card] += 1

print(len(cardsNum.keys()))