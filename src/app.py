import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st
import matplotlib.pyplot as plt
from HoldemHelper.recommend import PokerRecommender

# Set up the title
st.title("Poker Decision Assistant")

st.subheader("Select Your Cards")
col1, col2 = st.columns(2)
ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
suits = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}

with col1:
    rank1 = st.selectbox("Card 1 Rank", ranks, key="rank1")
    suit1_symbol = st.radio("Card 1 Suit", list(suits.keys()), horizontal=True, key="suit1")
    suit1 = suits[suit1_symbol]

with col2:
    rank2 = st.selectbox("Card 2 Rank", ranks, key="rank2")
    suit2_symbol = st.radio("Card 2 Suit", list(suits.keys()), horizontal=True, key="suit2")
    suit2 = suits[suit2_symbol]

hero_holding = f"{rank1}{suit1}{rank2}{suit2}"
hero_pos = st.selectbox("Hero Position", ["UTG", "HJ", "CO", "BTN", "SB", "BB"])

# Disable player count and update label
num_players = st.number_input("Number of Players (fixed at 6 for solver accuracy)", min_value=6, max_value=6, value=6, disabled=True)

# Multi-input form to build the betting line
st.subheader("Opponent Actions (in order before Hero acts)")
positions = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]

num_actions = st.number_input("Number of Opponent Actions", min_value=0, max_value=5, value=0, step=1)
betting_line_parts = []

for i in range(num_actions):
    col1, col2 = st.columns(2)
    available_positions = [p for p in positions if p not in [bp.split("/")[0] for bp in betting_line_parts]]
    pos = col1.selectbox(f"Position for Action {i+1}", available_positions, key=f"pos_{i}")
    action = col2.selectbox(f"Action for {pos}", ["fold", "call", "raise"], key=f"act_type_{i}")
    if action == "raise":
        raise_size = st.number_input(f"Raise size for {pos} (in bb)", min_value=0.1, step=0.1, key=f"raise_{i}")
        betting_line_parts.append(f"{pos}/{raise_size}bb")
    else:
        betting_line_parts.append(f"{pos}/{action}")

prev_line = "/".join(betting_line_parts)

# Load recommender
recommender = PokerRecommender()

# When the user clicks the Predict button
if st.button("Get Recommendation"):
    if not hero_holding or not hero_pos:
        st.warning("Please fill in all required fields.")
    else:
        prediction, probabilities = recommender.recommend(hero_holding, hero_pos, prev_line, num_players)
        st.success(f"Recommended Action: **{prediction.upper()}**")
        st.subheader("Prediction Probabilities:")

        st.write("")

        # Create a simple bar chart for probabilities
        fig, ax = plt.subplots()
        actions = list(probabilities.keys())
        probs = [float(probabilities[a]) for a in actions]
        ax.bar(actions, probs)
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        st.pyplot(fig)