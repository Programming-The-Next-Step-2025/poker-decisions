from .recommend import PokerRecommender
from .import_dataset import PokerModelTrainer

__all__ = ["PokerRecommender", "PokerModelTrainer", "recommender"]

# Lazy-safe loading with error capture
try:
    recommender = PokerRecommender()
except Exception as e:
    print("⚠️ Failed to initialize recommender:", e)
    recommender = None