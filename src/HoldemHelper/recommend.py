import pandas as pd
import joblib
from .hand_strengths import hand_strength
from .utils import canonical_hand, parse_prev_line

class PokerRecommender:
    def __init__(self, model_path="model/poker_model.pkl", encoder_path="model/label_encoder.pkl", feature_path="model/feature_columns.pkl"):
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.feature_columns = joblib.load(feature_path)

    def recommend(self, hero_holding, hero_pos, prev_line, num_players):
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

        print(f"🧪 Inference hand: {hero_holding}, position: {hero_pos}, features: {df.columns[df.any()].tolist()}")
        print(f"🧠 Model predicts: {decoded_prediction}, Probabilities: {class_probs}")

        return decoded_prediction, class_probs