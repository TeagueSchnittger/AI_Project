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
st.sidebar.info("0 = Small/Cold/Close | 1 = Big/Hot/Far")
w_dist = st.sidebar.slider("Distance Importance", 0.0, 1.0, 0.5)
w_temp = st.sidebar.slider("Temperature preference", 0.0, 1.0, 0.5)
w_pop  = st.sidebar.slider("Population preference", 0.0, 1.0, 0.5)

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

if st.sidebar.button("🧬 Generate 3-City Itinerary"):
    planner = GeneticPlanner(processed_data, user_coords, agent)
    
    # Identify the city objects for the top 5 matches
    top_names = [res['name'] for res in recommendations[:10]]
    top_city_objects = [c for c in processed_data if c['name'] in top_names]
    
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
    df_results = pd.DataFrame(recommendations)
    
    # Formatting table for better UX
    display_df = df_results[['name', 'score']].copy()
    display_df['score'] = (display_df['score'] * 100).round(1).astype(str) + '%'
    display_df.columns = ['City', 'Match Strength']
    st.table(display_df.head(10))
    
    if itinerary_route:
        st.markdown("### 🧬 AI-Optimized Itinerary")
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