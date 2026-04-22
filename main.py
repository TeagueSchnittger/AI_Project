import streamlit as st
import json
import pandas as pd
from src.utils import prepare_data
from src.agent import TravelAgent
import pydeck as pdk
from src.planner import GeneticPlanner

# Page Config
st.set_page_config(page_title="AI Travel Agent", layout="wide")
st.title("🌍 Intelligent Travel Decision Agent")
st.markdown("Adjust the sliders to find your perfect European destination.")

# 1. Sidebar - Get Inputs
st.sidebar.header("🗺️ Your Location")
u_lat = st.sidebar.number_input("Your Latitude", value=51.5074, format="%.4f")
u_lon = st.sidebar.number_input("Your Longitude", value=-0.1278, format="%.4f")
user_coords = (u_lat, u_lon)

st.sidebar.header("Your Preferences")

w_dist = st.sidebar.slider("Distance Importance", 0.0, 1.0, 0.5)
_d1, _d2 = st.sidebar.columns(2)
_d1.caption("← Close")
_d2.markdown("<p style='text-align:right;font-size:0.8em;color:gray'>Far →</p>", unsafe_allow_html=True)

w_temp = st.sidebar.slider("Temperature Preference", 0.0, 1.0, 0.5)
_t1, _t2 = st.sidebar.columns(2)
_t1.caption("← Cold")
_t2.markdown("<p style='text-align:right;font-size:0.8em;color:gray'>Hot →</p>", unsafe_allow_html=True)

w_pop = st.sidebar.slider("Population Preference", 0.0, 1.0, 0.5)
_p1, _p2 = st.sidebar.columns(2)
_p1.caption("← Low Pop")
_p2.markdown("<p style='text-align:right;font-size:0.8em;color:gray'>High Pop →</p>", unsafe_allow_html=True)

# 2. Process Data
def load_and_process(coords):
    with open('data/cities.json', 'r') as f:
        data = json.load(f)
    return prepare_data(data, coords)

processed_data = load_and_process(user_coords)
agent = TravelAgent(processed_data)

# 3. Agent Decision Logic (Single City)
recommendations = agent.evaluate(w_dist, w_temp, w_pop)
best_city_name = recommendations[0]['name']

# --- 3.5 ITINERARY LOGIC (Initialization) ---
# We define these here so the code below ALWAYS sees them
st.sidebar.markdown("---")
itinerary_route = []
line_data = []
current_home = {"lat": u_lat, "lon": u_lon} 

if st.sidebar.button("🧬 Generate Genetic 3-City Itinerary", help="Finds the optimal 3-city trip so you don't have to travel too far between each stop."):
    planner = GeneticPlanner(processed_data, user_coords, agent)
    
    # Identify the city objects for the top 10 matches, excluding cities too close to home
    top_names = [res['name'] for res in recommendations[:10]]
    top_city_objects = [c for c in processed_data if c['name'] in top_names and c['dist'] > 150]
    
    # Run the Evolution
    itinerary_route = planner.evolve(top_city_objects, w_dist, w_temp, w_pop)
    
    # Build the Line Data for the map
    temp_pos = current_home 
    for city in itinerary_route:
        line_data.append({
            "start": [float(temp_pos['lon']), float(temp_pos['lat'])],
            "end": [float(city['lon']), float(city['lat'])],
        })
        temp_pos = city
        
    # Add final return leg
    line_data.append({
        "start": [float(temp_pos['lon']), float(temp_pos['lat'])],
        "end": [float(u_lon), float(u_lat)],
    })

# 4. Display Results
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"The Agent Recommends: {best_city_name}")

    if agent.nn_model is None:
        st.warning("NN model not trained. Run `python -m src.train_scorer` then restart.", icon="⚠️")

    df_results = pd.DataFrame(recommendations)
    display_df = df_results[['name', 'score', 'nn_score']].copy()
    display_df['score'] = (display_df['score'] * 100).round(1).astype(str) + '%'
    if agent.nn_model is not None:
        display_df['nn_score'] = (display_df['nn_score'] * 100).round(1).astype(str) + '%'
    else:
        display_df['nn_score'] = 'N/A'
    display_df.columns = ['City', 'Utility Score', 'NN Score']
    st.table(display_df.head(10))
    
    if itinerary_route:
        st.markdown("### 🧬 Genetic Algorithm Itinerary")
        st.caption("Optimizes for closeness + match score across 50 generations")
        names = [c['name'] for c in itinerary_route]
        st.success(f"**Route:** Home ➔ {names[0]} ➔ {names[1]} ➔ {names[2]} ➔ Home")

with col2:
    st.subheader("Top Recommendations & Route")

    # Map Visuals
    top_5_names = [res['name'] for res in recommendations[:5]]
    plot_data = []
    for city in processed_data:
        if city['name'] in top_5_names:
            is_winner = (city['name'] == best_city_name)
            color = [255, 200, 0, 200] if is_winner else [0, 150, 255, 160]
            plot_data.append({
                "name": city['name'], "lat": float(city['lat']), "lon": float(city['lon']),
                "color": color, "radius": 50000 if is_winner else 30000
            })
            
    # Add User Dot
    plot_data.append({"name": "YOU", "lat": u_lat, "lon": u_lon, "color": [255, 0, 0, 255], "radius": 40000})

    layers = [
        pdk.Layer(
            'ScatterplotLayer',
            data=plot_data,
            get_position='[lon, lat]',
            get_color='color',
            get_radius='radius',
            pickable=True,
        )
    ]
    
    if line_data:
        layers.append(
            pdk.Layer(
                'LineLayer',
                data=line_data,
                get_source_position='start',
                get_target_position='end',
                get_color=[255, 255, 255, 200],
                get_width=5,
            )
        )

    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=2.5),
        layers=layers,
        tooltip={"text": "{name}"}
    ))