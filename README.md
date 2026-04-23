# Intelligent Travel Decision Agent

**Author:** Teague Schnittger  
**Course:** Intro to Artificial Intelligence — IFSA Study Abroad Program  
**Python Version:** 3.13.5

🌍 **Live App:** [aiproject-guvtowzs2ukdcpuk7fkcqh.streamlit.app](https://aiproject-guvtowzs2ukdcpuk7fkcqh.streamlit.app/)

---

![App Screenshot](data/IFSA_AI_PROJECT.png)

## Overview

A Streamlit web application that acts as an intelligent travel agent, recommending European cities based on user-defined preferences for distance, temperature, and population. The system combines three AI techniques: a neural network scorer that learns city match scores from data, a rule-based utility agent that serves as the training signal, and a Genetic Algorithm that produces an optimized 3-city itinerary.

---

## AI Techniques Used

### Neural Network Scorer
The core decision-making component is a feedforward neural network (`ScorerNet` in `src/nn_scorer.py`) that learns to predict city match scores from preference data rather than relying on hardcoded rules. This is the key property of a learned intelligent agent: behavior that emerges from training, not explicit programming.

**Architecture:** 4-layer MLP — `6 → 64 → 64 → 32 → 1` with ReLU activations and Sigmoid output.

**Input features:** `[dist_norm, temp_norm, pop_norm, w_dist, w_temp, w_pop]`

**Training:** 12,000 synthetic samples (80/20 train/test split), Adam optimizer, 400 epochs.

**Results:** Test R² = 0.993, MSE = 0.00069.

To retrain the model from scratch:
```bash
python -m src.train_scorer
```

### Rule-Based Utility Agent
The `TravelAgent` scores each city using a multiplicative utility function that acts as the training signal for the neural network:

```
score = (1 - |dist_norm - w_dist|) × (1 - |temp_norm - w_temp|) × (1 - |pop_norm - w_pop|)
```

Where `w_*` are user preference weights on a 0–1 scale and `*_norm` are min-max normalized city attributes. The multiplicative combination penalizes cities that miss on any single dimension. The UI displays both the utility score and NN score side by side for comparison.

### Genetic Algorithm Planner
The `GeneticPlanner` evolves a population of 3-city routes over 50 generations using selection, elitism, and crossover. Fitness balances total city match score against round-trip travel distance, favoring geographically compact, high-scoring itineraries.

---

## Dependencies

| Package      | Version |
|--------------|---------|
| streamlit    | 1.55.0  |
| pandas       | 2.3.3   |
| pydeck       | 0.9.1   |
| geopy        | 2.4.1   |
| scikit-learn | 1.8.0   |
| matplotlib   | 3.10.8  |
| numpy        | 2.4.2   |

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run main.py
```

> **Note:** Do not run with `python3 main.py` — Streamlit requires its own server to function correctly.

---

## Project Structure

```
AI_Project/
├── main.py                  # Streamlit UI and orchestration
├── requirements.txt         # Python dependencies
├── data/
│   ├── cities.json          # Static dataset of 20 European cities
│   ├── scorer_model.pkl     # Trained neural network model
│   ├── nn_loss_curve.png    # Training/validation loss curve
│   └── nn_pred_vs_actual.png # Predicted vs actual scores plot
└── src/
    ├── agent.py             # TravelAgent — utility + NN scoring
    ├── planner.py           # GeneticPlanner — 3-city itinerary optimization
    ├── utils.py             # Distance calculation and min-max normalization
    ├── nn_scorer.py         # ScorerNet — MLP definition, training, inference
    ├── train_scorer.py      # Standalone training script
    └── visualize_nn.py      # Generates training visualizations
```

---

## How to Use

1. Enter your current **latitude and longitude** in the sidebar.
2. Adjust the three preference sliders:
   - **Distance Importance** — 0 = prefer nearby cities, 1 = prefer farther cities
   - **Temperature Preference** — 0 = prefer cold climates, 1 = prefer hot climates
   - **Population Preference** — 0 = prefer small cities, 1 = prefer large cities
3. The agent instantly ranks all cities by match strength, showing both the utility score and NN score side by side.
4. Click **Generate Genetic 3-City Itinerary** to run the Genetic Algorithm and get an optimized route with minimal travel between stops.

---

## Figures

**Figure 1.** Ranked recommendations table and interactive map showing top 5 cities. The gold dot marks the top recommended city; blue dots mark remaining top matches; red dot marks the user's location.

![Figure 1](data/IFSA_AI_PROJECT2.png)

**Figure 2.** Genetic Algorithm itinerary result showing the optimized 3-city route. White lines trace the full round-trip path from home through each city and back.

![Figure 2](data/IFSA_AI_PROJECT.png)

---

## AI Tools Declared

| Tool | Version | Purpose |
|------|---------|---------|
| Claude Code (Anthropic) | claude-sonnet-4-6 | AI coding assistant — used for code generation, debugging, docstrings, and README authoring |

---

## References

[1] Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

[2] Mitchell, M. (1998). *An Introduction to Genetic Algorithms*. MIT Press.

[3] Kingma, D. P. & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980.

[4] Streamlit Inc. (2024). *Streamlit Documentation*. https://docs.streamlit.io

[5] Uber Technologies. (2024). *deck.gl — WebGL2 Powered Geospatial Visualization Layers*. https://deck.gl

[6] Kostya Lopuhin et al. (2024). *GeoPy Documentation*. https://geopy.readthedocs.io/en/stable/

[7] pandas Development Team. (2024). *pandas: Powerful Python Data Analysis Toolkit*. https://pandas.pydata.org/docs/
