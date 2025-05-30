import HoldemHelper

decision, probs = HoldemHelper.recommender.recommend("Th9h", "CO", "UTG/call/HJ/2.5bb", 6)
print("Decision:", decision)
print("Probabilities:", probs)