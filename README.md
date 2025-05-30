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

## User Scenario

**Purpose:** Demonstrate how a typical user interacts with the HoldemHelper software to receive a recommendation for a preflop poker decision.

**Individual:** Student A, a Research Master's student in Psychology at the University of Amsterdam. Interested in decision-making models and poker strategies.

**Assumptions:**
- The user has Python 3.9+ installed.
- The HoldemHelper package has been installed locally.
- The user has access to a web browser and a terminal.

**Scenario:**

1. Student A opens the terminal on their laptop and navigates to the HoldemHelper project directory.
2. They run the command `streamlit run src/app.py` to launch the Streamlit app.
3. The app opens in their default browser, showing a card and position input interface.
4. Student A selects `"Q"` and `"J"` as ranks, and hearts for both suits.
5. They choose `"BTN"` (Button) as their position.
6. They enter prior actions: `"UTG/fold", "HJ/raise", "CO/call"`.
7. Student A clicks the **Get Recommendation** button.
8. The app shows: **Recommended Action: call** and displays a probability chart.
9. Student A uses this to explore how hand strength and position affect optimal plays.

This scenario exemplifies how HoldemHelper provides an intuitive and educational experience for learning optimal poker strategies based on machine learning predictions.

## Installation
Clone the repository and install dependencies using pip:

```bash
git clone https://github.com/Programming-The-Next-Step-2025/poker-decisions.git
cd poker-decisions
pip install -e .
```

Example usage of the package as a script or in a package

```python
import HoldemHelper
decision, probs = HoldemHelper.recommender.recommend(
    hero_holding = "Ts9h", 
    hero_pos = "SB", 
    prev_line = "UTG/fold/HJ/call/CO/2.0bb/BTN/call", 
    5)
print(decision, probs)
```

### Running the Optional Streamlit App

To launch the Streamlit web app locally, ensure you're in the correct project directory, using the correct Python version (e.g., Python 3.12), and that your Streamlit version is up-to-date (v1.25.0 or later is recommended) to avoid compatibility issues such as unsupported keyword arguments like `use_container_width`. You can update Streamlit using:

```bash
pip install --upgrade streamlit
# or for conda users:
conda update streamlit
```

Booting up the Streamlit App:

```bash
cd path/to/poker-decisions
python3.12 -m streamlit run src/app.py
```

This will open a browser window displaying the HoldemHelper interface. From there, you can:

1. Select two hole cards (rank and suit).
2. Choose your position at the table (UTG, HJ, CO, BTN, SB, BB).
3. Input prior actions from opponents (e.g., `UTG/fold`, `HJ/raise`).
4. Click **Get Recommendation** to receive a model-driven action.
5. View a horizontal bar chart showing the probability distribution for `fold`, `call`, or `raise`.

The app uses a trained XGBoost model to simulate optimal preflop decisions in real-time.

## Training the Model
Regenerate training data and retrain the model:

```bash
python src/HoldemHelper/import_dataset.py
```

This will save the model to `model/poker_model.pkl`, along with the encoder and feature columns.

## Using HoldemHelper as a Python Package

After installing the package, you can use the core functionality in your own Python scripts, provided you're running the script with the correct Python interpreter (e.g., Python 3.12 if that's where HoldemHelper was installed):

```python
import HoldemHelper

decision, probabilities = HoldemHelper.recommender.recommend(
    hero_holding="QJs",
    hero_pos="BTN",
    prev_line="UTG/fold/HJ/raise/CO/call",
    num_players=6
)

print("Recommended Action:", decision)
print("Probabilities:", probabilities)
```

Make sure to run the script using the Python version where HoldemHelper was installed, for example:

```bash
python3.12 my_script.py
```

This lets you use HoldemHelper from scripts or interactive terminals without relying on the Streamlit web app.

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

## Flowchart of User Interaction with HoldemHelper

![Flowchart showing user flow](src/images/flowchart_image.png)

## Example Usage

### Launching the Streamlit App
![Launch Streamlit](src/images/streamlit_launch.png)

### Interface for Input
![User Input](src/images/interface_input.png)

### Output Recommendation & Probabilities
![Model Output](src/images/interface_output.png)

## Background
Poker decisions are modeled based on preflop game theory, augmented using realistic betting lines and strategic assumptions. Weak and premium hands are synthetically reinforced to encourage better generalization on edge cases.

## Contributing
Contributions and suggestions are welcome! Feel free to fork the repo, submit issues, or make pull requests.

## License
MIT License