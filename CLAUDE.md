# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
streamlit run main.py
```

Install dependencies (no requirements.txt exists yet; install manually):

```bash
pip install streamlit pandas pydeck geopy
```

Note: `requirments.py` is an empty placeholder file, not a real requirements file.

## Architecture

This is a Streamlit web app that recommends European cities based on user-defined preferences.

**Data flow:**
1. `main.py` collects user inputs (lat/lon coordinates + three preference sliders) via the Streamlit sidebar
2. `src/utils.py:prepare_data()` calculates geodesic distance from the user to each city (via `geopy`), then min-max normalizes distance, temperature, and population across all cities into `[0, 1]` ranges
3. `src/agent.py:TravelAgent.evaluate()` scores each city using a multiplicative utility function: `score = (1 - |dist_norm - w_dist|) × (1 - |temp_norm - w_temp|) × (1 - |pop_norm - w_pop|)` where `w_*` are the user's slider values
4. Results are displayed as a ranked table (top 10) and a `pydeck` map (top 5)

**Key files:**
- `main.py` — Streamlit UI and orchestration
- `src/agent.py` — `TravelAgent` class with `evaluate(w_dist, w_temp, w_pop)` scoring logic
- `src/utils.py` — distance calculation and normalization pipeline
- `src/planner.py` — empty stub, likely intended for multi-city route planning
- `data/cities.json` — static dataset of ~20+ European cities with `name`, `pop`, `temp`, `lat`, `lon`

**Scoring model:** Sliders represent user preferences on a 0–1 scale (0 = small/cold/close, 1 = big/hot/far). A city scores highest when its normalized attributes closely match the slider values. The multiplicative combination makes the agent "pickier" — a city that misses on any single dimension scores poorly overall.
