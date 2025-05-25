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

# Approximate preflop hand strengths (0.0 to 1.0)
hand_strength = {
    'AA': 1.00, 'KK': 0.995, 'QQ': 0.985, 'JJ': 0.97, 'TT': 0.95,
    'AKs': 0.96, 'AQs': 0.94, 'AJs': 0.92, 'KQs': 0.91,
    'AKo': 0.93, 'AQo': 0.91, 'AJo': 0.89, 'KQo': 0.88,
    '99': 0.90, '88': 0.87, '77': 0.84,
    '66': 0.80, '55': 0.76, '44': 0.72, '33': 0.68, '22': 0.64,
    'ATs': 0.90, 'KJs': 0.88, 'QJs': 0.86, 'JTs': 0.85, 'T9s': 0.82,
    # Add more if you'd like, or default others
}

# Normalize hero_holding to canonical form
def canonical_hand(hand):
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

def categorize_hand(hand):
    if not isinstance(hand, str) or len(hand) != 4:
        return 'unknown'
    rank_order = '23456789TJQKA'
    r1, s1, r2, s2 = hand[0], hand[1], hand[2], hand[3]
    suited = s1 == s2
    if r1 == r2:
        return f"{r1}{r2}"  # e.g., 'KK'
    ranks = sorted([r1, r2], key=lambda x: rank_order.index(x), reverse=True)
    high, low = ranks
    if suited:
        return f"{high}{low}s"  # e.g., 'AKs'
    return f"{high}{low}o"  # e.g., 'AKo'

## The following three lines are now applied after DataFrame expansion below.
# df['hero_holding'] = df['hero_holding'].apply(canonical_hand)
# df['hand_category'] = df['hero_holding'].apply(categorize_hand)
# df['hand_strength'] = df['hero_holding'].map(hand_strength).fillna(0.5)
df_raw = df.copy()  # Save raw data for later inspection
# Expand each row into multiple decision points (one per hero action)
expanded_rows = []

def normalize_action(action):
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
    
df['normalized_decision'] = df['correct_decision'].apply(normalize_action)

for idx, row in df_raw.iterrows():
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
                'correct_decision': normalize_action(action),
                'num_players': row.get('num_players', None),
            })
            found = True

    # ✅ Only add this once per row if hero never acted in the prev_line
    if not found:
        expanded_rows.append({
            'prev_line': row['prev_line'],
            'hero_pos': hero_pos,
            'hero_holding': row['hero_holding'],
            'correct_decision': normalize_action(row['correct_decision']),
            'num_players': row.get('num_players', None),
        })

# Infer folds from positions missing in prev_line before hero acts
position_order = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']
for idx, row in df_raw.iterrows():
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
# Now apply canonical_hand, categorize_hand, and hand_strength to the expanded df
df['hero_holding'] = df['hero_holding'].apply(canonical_hand)
df['hand_category'] = df['hero_holding'].apply(categorize_hand)
df['hand_strength'] = df['hand_category'].map(hand_strength).fillna(0.5)

# Fill missing hero_holding and num_players in inferred folds
df['hero_holding'].fillna('unknown', inplace=True)
df['num_players'].fillna(6, inplace=True)  # assume 6 players if unknown
df_raw = df.copy()  # Keep a copy for later reference

# Flag: has the hero already acted earlier in this hand?
df['hero_acted_before'] = df.apply(
    lambda row: row['prev_line'].count(row['hero_pos']) > 1 if isinstance(row['prev_line'], str) else False,
    axis=1
)

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
 # Remove hands with no meaningful info (likely auto-folds)
df = df[~((df['hero_holding'] == 'unknown') & (df['prev_line'] == ''))].copy()
features = [
    'hand_category', 'hero_pos',
    'facing_raise', 'num_raises', 'last_raiser_pos',
    'estimated_pot', 'last_raise_size', 'num_players_still_in', 
    'to_call', 'pot_odds', 'is_3bet_plus', 'hand_strength', 'hero_acted_before'
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

# Step 3: Encode the label (target) AFTER resampling
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df[target])

# Step 4: One-hot encode categorical features
df_encoded = pd.get_dummies(df[['hand_category', 'hero_pos', 'last_raiser_pos']])

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
model.fit(X_train, y_train, sample_weight=sample_weights)

# Step 8: Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2%}")

# Step 9: Predict multiple samples for inspection
num_samples = 10
samples = X_test.sample(n=num_samples, random_state=42)
predicted = model.predict(samples)
decoded = label_encoder.inverse_transform(predicted)
probs = model.predict_proba(samples)

for i, idx in enumerate(samples.index):
    print(f"\n--- Sample {i+1} ---")
    print("🎯 Original decision and input:\n", df_raw.loc[idx])
    print(f"🔍 Predicted action for sample hand: {decoded[i]}")
    class_probs = dict(zip(label_encoder.classes_, probs[i]))
    print("🔎 Prediction probabilities:", class_probs)
    print(df[['hero_holding', 'hand_category', 'hand_strength']].sample(10))

print(df['correct_decision'].value_counts())

import joblib
import os

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Save trained model, label encoder, and feature columns
joblib.dump(model, "model/poker_model.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")
joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")