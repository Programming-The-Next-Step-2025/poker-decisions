import pandas as pd
import joblib
from .hand_strengths import hand_strength
from .utils import canonical_hand, parse_prev_line
import streamlit as st
import importlib.resources
from . import model  # model/ must have __init__.py

# Loads pretrained model and encoders for real-time inference.
# Designed for stateless prediction using a fully serialized pipeline.
class PokerRecommender:
    def __init__(self):
        """
        Loads model and preprocessing artifacts (XGBoost model, label encoder, feature columns).
        - Uses importlib to safely access files inside the packaged `model/` directory.
        - Assumes all artifacts were saved via joblib and are scikit-learn compatible.
        """
        with importlib.resources.files(model).joinpath("poker_model.pkl").open("rb") as f:
            self.model = joblib.load(f)
        with importlib.resources.files(model).joinpath("label_encoder.pkl").open("rb") as f:
            self.label_encoder = joblib.load(f)
        with importlib.resources.files(model).joinpath("feature_columns.pkl").open("rb") as f:
            self.feature_columns = joblib.load(f)

    def recommend(self, hero_holding, hero_pos, prev_line, num_players):
        """
        Constructs feature vector and predicts the recommended action.
        - Canonicalizes input hand to match training format.
        - Parses previous actions to derive contextual numeric features.
        - Applies one-hot encoding to match feature layout from training.
        
        Returns:
            Tuple[str, dict]: Predicted action label, and class probabilities.
        """
        # Normalize hand
        hero_holding = canonical_hand(hero_holding)

        # Parse betting history
        parsed = parse_prev_line(prev_line, hero_pos)

        # Assemble features
        data = {
            'hero_holding': hero_holding,
            'hero_pos': hero_pos,
            'num_players': num_players,
            'facing_raise': parsed['facing_raise'],
            'num_raises': parsed['num_raises'],
            'estimated_pot': parsed['estimated_pot'],
            'last_raise_size': parsed['last_raise_size'],
            'num_players_still_in': parsed['num_players_still_in'],
            'to_call': parsed['to_call'],
            'pot_odds': parsed['pot_odds'],
            'is_3bet_plus': parsed['is_3bet_plus'],
            'hero_acted_before': parsed['hero_acted_before'],
            'hand_strength': hand_strength.get(hero_holding, 0.5)
        }

        # Create DataFrame and one-hot encode
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        # Predict
        prediction = self.model.predict(df)
        probabilities = self.model.predict_proba(df)[0]
        decoded_prediction = self.label_encoder.inverse_transform(prediction)[0]
        class_probs = dict(zip(self.label_encoder.classes_, probabilities))

        return decoded_prediction, class_probs