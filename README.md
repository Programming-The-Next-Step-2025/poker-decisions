# HoldemHelper Package

HoldemHelper is a Python package for making statistically grounded poker decisions in 6-player Texas Hold'em. It includes functions for generating training data, evaluating scenarios, and inferring preflop actions using a trained machine learning model.

While a Streamlit app is provided for demonstration and interactive use, the core functionality is built for integration as a standalone Python package.

# HoldemHelper

HoldemHelper is a Python package and Streamlit web application designed to assist poker players in making **data-driven preflop decisions** for 6-player Texas Hold'em games.

## Overview
This project combines expert-sourced strategy charts, synthetic data augmentation, and machine learning to recommend optimal plays (fold, call, raise) based on:
- Your hand (hole cards)
- Your position at the table
- Opponent betting actions before your turn

The package can be used programmatically or via an optional Streamlit interface.

## Machine Learning
- Trains an XGBoost classifier using both real and synthetic data.
- Encodes hands, positions, and game context into predictive features.
- Evaluates accuracy, tests decision-making against known edge cases.

## Features
- Input card ranks and suits via an intuitive UI.
- Track opponent action history.
- Visualize recommendation probabilities.
- Continuous improvements using synthetic scenarios like UTG folds and 4-bet/5-bet premiums.

## Installation
Clone the repository and install dependencies using pip:

```bash
git clone https://github.com/Programming-The-Next-Step-2025/poker-decisions.git
cd poker-decisions
pip install -r requirements.txt
```

Note: The `requirements.txt` file is automatically generated from `pyproject.toml` and includes all necessary dependencies.

### Running the Optional Streamlit App
To launch the Streamlit web app locally:

```bash
streamlit run src/app.py
```

## Training the Model
Regenerate training data and retrain the model:

```bash
python src/HoldemHelper/import_dataset.py
```

This will save the model to `model/poker_model.pkl`, along with the encoder and feature columns.

## Using HoldemHelper as a Python Package

After installing, you can use the core functionality in your own Python scripts:

```python
from HoldemHelper.recommend import PokerRecommender

recommender = PokerRecommender()
prediction, probabilities = recommender.recommend(
    hero_holding="QJs",
    hero_pos="BTN",
    prev_line="UTG/fold/HJ/raise/CO/call",
    num_players=6
)
print("Recommended Action:", prediction)
print("Probabilities:", probabilities)
```

This allows you to integrate the recommendation system into notebooks, simulations, or larger tools without relying on the web UI.

## Project Structure
```
poker-decisions/
├── src/
│   ├── app.py                  # Streamlit app UI
│   ├── HoldemHelper/
│   │   ├── import_dataset.py   # Data creation and training
│   │   ├── recommend.py        # Model inference logic
│   │   ├── hand_strengths.py   # Static strength values for all hands
│   │   └── ...                 # Other helpers
├── model/                      # Trained model artifacts (.pkl files)
└── README.md
```

## Background
Poker decisions are modeled based on preflop game theory, augmented using realistic betting lines and strategic assumptions. Weak and premium hands are synthetically reinforced to encourage better generalization on edge cases.

## Contributing
Contributions and suggestions are welcome! Feel free to fork the repo, submit issues, or make pull requests.

## License
MIT License