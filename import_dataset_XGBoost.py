from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Step 1: Load dataset
ds = load_dataset(
    "RZ412/PokerBench",
    data_files="preflop_60k_train_set_game_scenario_information.csv"
)
df = ds['train'].to_pandas()
df_raw = df.copy()  # Save raw data for later inspection

def normalize_action(action):
    action = str(action).lower()
    if 'fold' in action:
        return 'fold'
    elif 'call' in action:
        return 'call'
    elif 'allin' in action:
        return 'raise'
    elif 'bb' in action:
        return 'raise'
    else:
        return action  # fallback, or log unexpected values

df['normalized_decision'] = df['correct_decision'].apply(normalize_action)

def parse_prev_line(prev_line, hero_pos):
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

    while i < len(tokens) - 1:
        pos = tokens[i]
        action = tokens[i + 1]

        if pos == hero_pos:
            break  # stop at hero's first appearance

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

# Step 2: Select features and target
# Apply feature extraction to each row
parsed = df.apply(lambda row: parse_prev_line(row['prev_line'], row['hero_pos']), axis=1)
df['facing_raise'] = parsed.apply(lambda x: x[0])
df['num_raises'] = parsed.apply(lambda x: x[1])
df['last_raiser_pos'] = parsed.apply(lambda x: x[2])
df['estimated_pot'] = parsed.apply(lambda x: x[3])
df['last_raise_size'] = parsed.apply(lambda x: x[4])
df['num_players_still_in'] = parsed.apply(lambda x: x[5])
df['to_call'] = df['facing_raise'].apply(lambda x: 1.0 if x else 0.0) * df['last_raise_size']
df['pot_odds'] = df.apply(lambda row: row['to_call'] / (row['estimated_pot'] + row['to_call']) if row['to_call'] > 0 else 0, axis=1)
df['is_3bet_plus'] = df['num_raises'].apply(lambda x: x >= 2)
features = [
    'hero_holding', 'hero_pos',
    'facing_raise', 'num_raises', 'last_raiser_pos',
    'estimated_pot', 'last_raise_size', 'num_players_still_in', 
    'to_call', 'pot_odds', 'is_3bet_plus'
]
target = 'normalized_decision'

# One-hot encode available_moves if present
if 'available_moves' in df.columns:
    df['available_moves'] = df['available_moves'].apply(lambda moves: ','.join(sorted(eval(moves))) if isinstance(moves, str) else '')
    df_available_moves = df['available_moves'].str.get_dummies(sep=',')
    df = pd.concat([df, df_available_moves], axis=1)
    features.extend(df_available_moves.columns.tolist())

# Update the features list if available_moves were included
features = list(set(features))  # remove duplicates if any

df = df[features + [target]]

# Step 3: Encode the label (target) AFTER resampling
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df[target])

# Step 4: One-hot encode categorical features
df_encoded = pd.get_dummies(df[['hero_holding', 'hero_pos', 'last_raiser_pos']])

# Step 5: Combine encoded features with numeric ones
X = pd.concat([
    df_encoded,
    df[['facing_raise', 'num_raises', 'estimated_pot', 'last_raise_size',
        'num_players_still_in', 'to_call', 'pot_odds', 'is_3bet_plus'] + [col for col in df.columns if col not in features + [target, 'label'] and col in df.columns]]
], axis=1)
y = df['label']

# Step 6: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Step 7: Train XGBoost model
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, y_train)

# Step 8: Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2%}")

# Step 9: Predict a sample
sample = X_test.sample(1, random_state=None)
original_index = sample.index[0]
predicted = model.predict(sample)
decoded = label_encoder.inverse_transform(predicted)
print("\n📋 Sample input the model saw:\n", sample.iloc[0])
print("\n🎯 Original decision and input:\n", df_raw.loc[original_index])
print(f"🔍 Predicted action for sample hand: {decoded[0]}")
probs = model.predict_proba(sample)
class_probs = dict(zip(label_encoder.classes_, probs[0]))
print("\n🔎 Prediction probabilities:", class_probs)

import joblib
import os

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Save trained model, label encoder, and feature columns
joblib.dump(model, "model/poker_model.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")
joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")