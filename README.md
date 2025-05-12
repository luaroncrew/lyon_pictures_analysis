# lyon_pictures_analysis
a project meant to analyze a large dataset of picture metadata to understand ?something? about Lyon (the best city in the world)

# Run the project

Mac / Linux
```bash
python -m venv venv && source venv/bin/activate
```

Windows
```bash
python -m venv venv && source venv/Scripts/activate
```

```bash 
pip install -r requirements.txt
```

please run the pre-processing before running the app

then from the root
```bash
streamlit run app.py
```

# Project Structure

- `initial_data.csv` / data provided by the "client"
- `app.py` / entrypoint of the streamlit app
- `data_exploration_tab.py` / a tab allowing to show the map, clustering parameters


