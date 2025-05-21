import pandas as pd

def recommend_action(hero_holding, hero_pos, last_raiser_pos, facing_raise, num_raises, num_players, pot_size, model, label_encoder, feature_columns):    # 1. Create a one-row input DataFrame
    input_df = pd.DataFrame([{
        'hero_holding': hero_holding,
        'hero_pos': hero_pos,
        'num_players': num_players,
        'pot_size': pot_size
    }])

    # 2. One-hot encode categorical features
    encoded_input = pd.get_dummies(input_df[['hero_holding', 'hero_pos']])

    # 3. Add numeric features
    encoded_input['num_players'] = num_players
    encoded_input['pot_size'] = pot_size

    # 4. Align with training columns (fill missing with 0)
    encoded_input = encoded_input.reindex(columns=feature_columns, fill_value=0)

    # 5. Predict and decode
    prediction = model.predict(encoded_input)
    return label_encoder.inverse_transform(prediction)[0]