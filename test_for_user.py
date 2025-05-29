import HoldemHelper

decision, probs = HoldemHelper.recommender.recommend("93o", "UTG", "", 6)
print("Decision:", decision)
print("Probabilities:", probs)