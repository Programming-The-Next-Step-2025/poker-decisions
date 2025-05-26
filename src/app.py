import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st
import plotly.express as px
from HoldemHelper.recommend import PokerRecommender

# Set up the title
st.title("Poker Assistant")

st.subheader("Select Your Cards")
col1, col2 = st.columns(2)
ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
suits = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}

with col1:
    rank1 = st.selectbox(
        "Card 1 Rank",
        ranks,
        key="rank1",
        help="Select the rank (e.g. A, K, Q, etc.) of your first card."
    )
    suit1_symbol = st.radio(
        "Card 1 Suit",
        list(suits.keys()),
        horizontal=True,
        key="suit1",
        help="Select the suit for your first card."
    )
    suit1 = suits[suit1_symbol]

with col2:
    rank2 = st.selectbox(
        "Card 2 Rank",
        ranks,
        key="rank2",
        help="Select the rank (e.g. A, K, Q, etc.) of your second card."
    )
    suit2_symbol = st.radio(
        "Card 2 Suit",
        list(suits.keys()),
        horizontal=True,
        key="suit2",
        help="Select the suit for your second card."
    )
    suit2 = suits[suit2_symbol]

hero_holding = f"{rank1}{suit1}{rank2}{suit2}"
hero_pos = st.selectbox(
    "Hero Position",
    ["UTG", "HJ", "CO", "BTN", "SB", "BB"],
    help="Your position at the table. UTG is first to act, BB is last."
)
with st.expander("Show Table Positions Help"):
    st.image("src/images/poker_positions.jpg", caption="Poker table positions (1 = UTG, ..., 6 = BB)", use_container_width=True)

# Disable player count and update label
num_players = st.number_input(
    "Number of Players (fixed at 6 for solver accuracy)",
    min_value=6,
    max_value=6,
    value=6,
    disabled=True
)

# Multi-input form to build the betting line
st.subheader("Opponent Actions (in order before Hero acts)")
positions = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]

num_actions = st.number_input(
    "Number of Opponent Actions",
    min_value=0,
    max_value=15,
    value=0,
    step=1,
    help="How many opponents have acted before you?"
)
betting_line_parts = []

for i in range(num_actions):
    col1, col2 = st.columns(2)
    available_positions = positions
    pos = col1.selectbox(
        f"Position for Action {i+1}",
        available_positions,
        key=f"pos_{i}",
        help="The position of the opponent who acted."
    )
    action = col2.selectbox(
        f"Action for {pos}",
        ["fold", "call", "raise"],
        key=f"act_type_{i}",
        help="What action did this opponent take?"
    )
    if action == "raise":
        raise_size = st.number_input(
            f"Raise size for {pos} (in bb)",
            min_value=0.1,
            step=0.1,
            key=f"raise_{i}",
            help="Amount of big blinds raised."
        )
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
        # Create a horizontal bar chart for probabilities using Plotly with blue gradients, no grid, and no title
        import pandas as pd
        actions = list(probabilities.keys())
        probs = [float(probabilities[a]) for a in actions]
        df = pd.DataFrame({
            "Action": actions,
            "Probability": probs
        })
        fig = px.bar(
            df,
            x="Probability",
            y="Action",
            orientation='h',
            color="Action",
            color_discrete_sequence=["#003f5c", "#2f4b7c", "#665191"],
            opacity=0.50,
            hover_data={"Probability": ":.2f"}
        )
        fig.update_traces(hovertemplate='Probability: %{x:.2f}<extra></extra>')
        fig.update_layout(
            title="",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            showlegend=False
        )
        st.plotly_chart(fig)