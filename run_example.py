import joblib
from recommend import recommend_action

# Load model and metadata
model = joblib.load("model/poker_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# Use the function
action = recommend_action(
    hero_holding='AhKh',
    hero_pos='BTN',
    last_raiser_pos='CO',      # 'None' if no one raised
    facing_raise=True,         # True if a raise occurred before
    num_raises=1,              # Number of raises before hero
    num_players=6,
    pot_size=90.0,
    model=model,
    label_encoder=label_encoder,
    feature_columns=feature_columns
)

print("🤖 Recommended action:", action)