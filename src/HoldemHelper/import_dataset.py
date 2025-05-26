from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from HoldemHelper.hand_strengths import hand_strength
import joblib
import os
import random

class PokerModelTrainer:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        self.df_raw = None
        self.df = None
        self.X = None
        self.y = None
        self.features = None
        self.target = None

    def canonical_hand(self, hand):
        """Convert hand like 'AsKs' or 'KcKh' to canonical form like 'AKs' or 'KK'."""
        if not isinstance(hand, str) or len(hand) != 4:
            return hand

        rank_order = '23456789TJQKA'
        ranks = [hand[0], hand[2]]
        suits = [hand[1], hand[3]]

        # Sort cards by rank
        if rank_order.index(ranks[0]) < rank_order.index(ranks[1]):
            ranks = [ranks[1], ranks[0]]
            suits = [suits[1], suits[0]]

        suited = suits[0] == suits[1]
        if ranks[0] == ranks[1]:
            return f"{ranks[0]}{ranks[1]}"
        return f"{ranks[0]}{ranks[1]}{'s' if suited else 'o'}"

    def normalize_action(self, action):
        action = str(action).lower()
        if 'fold' in action:
            return 'fold'
        elif 'call' in action:
            return 'call'
        elif 'check' in action:
            return 'fold'
        elif 'allin' in action or 'bb' in action:
            return 'raise'
        else:
            return 'unknown'

    def parse_prev_line(self, prev_line, hero_pos):
        """
        Parses PokerBench-style action lines up to the hero's first action.
        Returns: facing_raise, num_raises, last_raiser_pos, estimated_pot, last_raise_size, num_players_still_in
        """
        facing_raise = False
        num_raises = 0
        last_raiser_pos = None
        estimated_pot = 1.5  # start with SB + BB
        last_raise_size = 0.0
        positions_in = set()

        if not isinstance(prev_line, str):
            return facing_raise, num_raises, last_raiser_pos, estimated_pot, last_raise_size, 1  # only hero still in

        position_order = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        tokens = prev_line.split('/')
        i = 0
        # Find the last action index for hero_pos to stop there
        hero_indices = [j for j in range(0, len(tokens), 2) if tokens[j] == hero_pos]
        stop_index = hero_indices[-1] if hero_indices else len(tokens)

        while i < len(tokens) - 1 and i < stop_index:
            pos = tokens[i]
            action = tokens[i + 1]

            if 'bb' in action.lower():  # raise or bet
                try:
                    amount = float(action.lower().replace('bb', ''))
                    estimated_pot += amount
                    facing_raise = True
                    num_raises += 1
                    last_raiser_pos = pos
                    last_raise_size = amount
                    positions_in.add(pos)
                except:
                    pass
            elif action.lower() == 'call':
                estimated_pot += last_raise_size
                positions_in.add(pos)
            elif action.lower() == 'allin':
                estimated_pot += last_raise_size
                positions_in.add(pos)
            elif action.lower() == 'fold':
                pass  # explicitly folded
            else:
                pass  # ignore unknowns

            i += 2

        # Infer folded positions
        hero_index = position_order.index(hero_pos)
        positions_before_hero = position_order[:hero_index]
        for pos in positions_before_hero:
            if pos not in positions_in:
                continue  # assumed to have folded

        num_players_still_in = len(positions_in) + 1  # include hero

        return facing_raise, num_raises, last_raiser_pos, estimated_pot, last_raise_size, num_players_still_in

    def load_and_preprocess(self):
        # Step 1: Load dataset
        ds = load_dataset(
            "RZ412/PokerBench",
            data_files="preflop_60k_train_set_game_scenario_information.csv"
        )
        df = ds['train'].to_pandas()

        # Normalize hero_holding to canonical form
        df['hero_holding'] = df['hero_holding'].apply(self.canonical_hand)
        df['hand_strength'] = df['hero_holding'].map(hand_strength).fillna(0.5)
        self.df_raw = df.copy()  # Save raw data for later inspection
        # Expand each row into multiple decision points (one per hero action)
        expanded_rows = []

        df['normalized_decision'] = df['correct_decision'].apply(self.normalize_action)

        for idx, row in self.df_raw.iterrows():
            if not isinstance(row['prev_line'], str):
                continue

            tokens = row['prev_line'].split('/')
            hero_pos = row['hero_pos']

            found = False
            for i in range(0, len(tokens) - 1, 2):
                pos, action = tokens[i], tokens[i + 1]
                if pos == hero_pos:
                    partial_prev_line = '/'.join(tokens[:i])
                    expanded_rows.append({
                        'prev_line': partial_prev_line,
                        'hero_pos': hero_pos,
                        'hero_holding': row['hero_holding'],
                        'correct_decision': self.normalize_action(action),
                        'num_players': row.get('num_players', None),
                    })
                    found = True

            # ✅ Only add this once per row if hero never acted in the prev_line
            if not found:
                expanded_rows.append({
                    'prev_line': row['prev_line'],
                    'hero_pos': hero_pos,
                    'hero_holding': row['hero_holding'],
                    'correct_decision': self.normalize_action(row['correct_decision']),
                    'num_players': row.get('num_players', None),
                })

        # Infer folds from positions missing in prev_line before hero acts
        position_order = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        for idx, row in self.df_raw.iterrows():
            if not isinstance(row['prev_line'], str):
                continue

            tokens = row['prev_line'].split('/')
            hero_pos = row['hero_pos']
            hero_index = position_order.index(hero_pos)
            acted_positions = set(tokens[i] for i in range(0, len(tokens), 2))
            expected_before_hero = position_order[:hero_index]

            for pos in expected_before_hero:
                # Skip low-information inferred folds (no prev_line and no hand)
                if row.get('hero_holding') is None and row.get('prev_line') == '':
                    continue

                if pos not in acted_positions:
                    expanded_rows.append({
                        'prev_line': '',  # no prior action since they folded pre-emptively
                        'hero_pos': pos,
                        'hero_holding': None,
                        'correct_decision': 'fold',
                        'num_players': None,
                    })

        # Convert to DataFrame
        df = pd.DataFrame(expanded_rows)
        # Add marker for synthetic rows (all False here, will add True on synthetic appends)
        df['is_synthetic'] = False
        df['hero_holding'] = df['hero_holding'].apply(self.canonical_hand)
        df['hand_strength'] = df['hero_holding'].map(hand_strength).fillna(0.5)
        # Fill missing hero_holding and num_players in inferred folds
        df['hero_holding'].fillna('unknown', inplace=True)
        df['num_players'].fillna(6, inplace=True)  # assume 6 players if unknown
        self.df_raw = df.copy()  # Keep a copy for later reference

        # Flag: has the hero already acted earlier in this hand?
        df['hero_acted_before'] = df.apply(
            lambda row: row['prev_line'].count(row['hero_pos']) > 1 if isinstance(row['prev_line'], str) else False,
            axis=1
        )

        # Step 2: Select features and target
        # Apply feature extraction to each row
        parsed = df.apply(lambda row: self.parse_prev_line(row['prev_line'], row['hero_pos']), axis=1)
        df['facing_raise'] = parsed.apply(lambda x: x[0])
        df['num_raises'] = parsed.apply(lambda x: x[1])
        df['last_raiser_pos'] = parsed.apply(lambda x: x[2])
        df['estimated_pot'] = parsed.apply(lambda x: x[3])
        df['last_raise_size'] = parsed.apply(lambda x: x[4])
        df['num_players_still_in'] = parsed.apply(lambda x: x[5])
        df['to_call'] = df['facing_raise'].apply(lambda x: 1.0 if x else 0.0) * df['last_raise_size']
        df['pot_odds'] = df.apply(lambda row: row['to_call'] / (row['estimated_pot'] + row['to_call']) if row['to_call'] > 0 else 0, axis=1)
        df['is_3bet_plus'] = df['num_raises'].apply(lambda x: x >= 2)
        # Remove hands with no meaningful info (likely auto-folds)
        df = df[~((df['hero_holding'] == 'unknown') & (df['prev_line'] == ''))].copy()
        features = [
            'hero_holding', 'hero_pos',
            'facing_raise', 'num_raises',
            'estimated_pot', 'last_raise_size', 'num_players_still_in',
            'to_call', 'pot_odds', 'is_3bet_plus', 'hand_strength', 'hero_acted_before',
            'is_synthetic'
        ]
        target = 'correct_decision'

        df = df[df['correct_decision'] != 'check']

        df_balanced = pd.concat([
            df[df[target] == 'raise'].sample(n=8000, replace=True, random_state=42),
            df[df[target] == 'call'].sample(n=8000, replace=True, random_state=42),
            df[df[target] == 'fold'].sample(n=8000, replace=True, random_state=42),
        ])

        # Shuffle the balanced DataFrame
        df = df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

        df = df[features + [target]]

        # Add synthetic data to balance and enrich dataset
        # Add synthetic UTG folds with realistic weak hands (use real card strings)
        weak_offsuit_hands = [
            '9s3h', '7d2c', 'Th2c', '8c4d', 'Jc3s', '3d2h'
        ]
        hero_positions_weak = ['UTG', 'HJ', 'CO']
        players_still_in_options = [6, 5, 4]
        position_pool = list(zip(hero_positions_weak, players_still_in_options))
        for _ in range(1000):  # Increase quantity for stronger signal
            raw_hand = random.choice(weak_offsuit_hands)
            canon_hand = self.canonical_hand(raw_hand)
            heropos, num_in = random.choice(position_pool)
            df = pd.concat([df, pd.DataFrame([{
                'hero_holding': canon_hand,
                'hero_pos': heropos,
                'facing_raise': False,
                'num_raises': 0,
                'estimated_pot': 1.5,
                'last_raise_size': 0.0,
                'num_players_still_in': num_in,
                'to_call': 0.0,
                'pot_odds': 0.0,
                'is_3bet_plus': False,
                'hand_strength': hand_strength.get(canon_hand, 0.2),
                'hero_acted_before': False,
                'correct_decision': 'fold',
                'is_synthetic': True
            }])], ignore_index=True)

        # Add synthetic premium 4-bet/5-bet hands with realistic strong hands (use real card strings)
        premium_hands = [
            'AsAd', 'KsKh', 'QhQc', 'AcKc', 'AhAs', 'AdKc'
        ]
        last_raise_sizes = [50.0, 30.0, 25.0, 15.0, 60.0, 77.0, 34.0]
        players_still_in = [2, 3, 4, 5, 6]
        hero_poss = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        raises = [3, 4]
        for _ in range(1000):  # Increase for better learning
            raw_hand = random.choice(premium_hands)
            canon_hand = self.canonical_hand(raw_hand)
            pl_still_in = random.choice(players_still_in)
            numraise = random.choice(raises)
            heroposs = random.choice(hero_poss)
            last_raise_s = random.choice(last_raise_sizes)
            estimated_pot_val = random.uniform(25, 80)
            to_call_val = last_raise_s
            df = pd.concat([df, pd.DataFrame([{
                'hero_holding': canon_hand,
                'hero_pos': heroposs,
                'facing_raise': True,
                'num_raises': numraise,
                'estimated_pot': estimated_pot_val,
                'to_call': to_call_val,
                'pot_odds': to_call_val / (estimated_pot_val + to_call_val),
                'last_raise_size': last_raise_s,
                'num_players_still_in': pl_still_in,
                'is_3bet_plus': True,
                'hand_strength': hand_strength.get(canon_hand, 1.0),
                'hero_acted_before': True,
                'correct_decision': 'raise',
                'is_synthetic': True
            }])], ignore_index=True)

        print("✅ Total samples after augmentation:", len(df))
        print("🧾 Label distribution after augmentation:")
        print(df['correct_decision'].value_counts())
        self.df = df
        self.features = features
        self.target = target

    def train_model(self):
        # Step 3: Encode the label (target) AFTER resampling
        self.label_encoder = LabelEncoder()
        self.df['label'] = self.label_encoder.fit_transform(self.df[self.target])

        # Step 4: One-hot encode categorical features
        df_encoded = pd.get_dummies(self.df[['hero_holding', 'hero_pos']])
        print("🧩 Encoded columns:", df_encoded.columns.tolist())
        print("🔍 Sample encoded rows from the end (should include synthetic):")
        print(df_encoded.tail())

        # Step 5: Combine encoded features with numeric ones
        self.X = pd.concat([
            df_encoded,
            self.df[self.features[2:]]  # skip 'hero_holding', 'hero_pos' because they're already in df_encoded
        ], axis=1)
        self.y = self.df['label']

        # Step 6: Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        # Step 7: Train XGBoost model
        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )
        self.model.fit(X_train, y_train, sample_weight=sample_weights)

        # Save test sets for evaluation
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_model(self):
        # Step 8: Evaluate model
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\n✅ Model Accuracy: {accuracy:.2%}")

        # 🔬 Testing model on synthetic UTG folds:
        print("\n🔬 Testing model on synthetic UTG folds:")
        synthetic_utg_folds = self.df[
            (self.df['is_synthetic']) & 
            (self.df['hero_holding'].isin(['93o', '92s'])) & 
            (self.df['hero_pos'] == 'UTG')
        ]
        if len(synthetic_utg_folds) > 0:
            synthetic_utg_folds = synthetic_utg_folds.sample(n=min(5, len(synthetic_utg_folds)), random_state=1)
            for idx, row in synthetic_utg_folds.iterrows():
                input_vector = self.X.loc[idx:idx]
                pred_label = self.model.predict(input_vector)[0]
                decoded = self.label_encoder.inverse_transform([pred_label])[0]
                probs = self.model.predict_proba(input_vector)[0]
                print(f"\n🧪 Synthetic Sample {idx}")
                print(f"✋ Hero Holding: {row['hero_holding']} | Position: {row['hero_pos']} | Label: {row['correct_decision']}")
                print(f"🔍 Predicted: {decoded} | Probabilities: {dict(zip(self.label_encoder.classes_, probs))}")

    def save_artifacts(self):
        # Ensure model directory exists
        os.makedirs("model", exist_ok=True)

        # Save trained model, label encoder, and feature columns
        joblib.dump(self.model, "model/poker_model.pkl")
        joblib.dump(self.label_encoder, "model/label_encoder.pkl")
        joblib.dump(self.X.columns.tolist(), "model/feature_columns.pkl")

    def run(self):
        self.load_and_preprocess()
        self.train_model()
        self.evaluate_model()
        self.save_artifacts()

if __name__ == "__main__":
    trainer = PokerModelTrainer()
    trainer.run()