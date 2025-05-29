from .recommend import PokerRecommender
from .import_dataset import PokerModelTrainer

# Optional convenience instance
recommender = PokerRecommender()

__all__ = ["PokerRecommender", "PokerModelTrainer", "recommender"]