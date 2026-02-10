import pickle
import os
from datetime import datetime

def save_simulation_data(results, params):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"results_{timestamp}.pkl"
    data = {"results": results, "params": params}
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Data saved to {filename}")
    return filename

def load_simulation_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data["results"], data["params"]