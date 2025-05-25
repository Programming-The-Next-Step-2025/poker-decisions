import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st
from HoldemHelper.recommend import PokerRecommender

# Set up the title
st.title("Poker Decision Assistant")

# Input fields
hero_holding = st.text_input("Hero Holding (e.g. AsKs, QdQc):")
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
    act = col2.text_input(f"Action for {pos} (e.g., fold, call, 6.5bb)", key=f"act_{i}")
    betting_line_parts.append(f"{pos}/{act}")

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
        st.write(probabilities)