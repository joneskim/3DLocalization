import numpy as np
import pandas as pd
import pytest
from core import ThreeD_Navigator
import pickle
import json

knn_tdoa = pickle.load(open('knn_tdoa_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def convert_to_array(str_list):
    return np.array(json.loads(str_list)).tolist()

def load_test_data(file_path):
    data = pd.read_csv(file_path)
    test_data = []

    for column in ['arrival_times', 'stations', 'receiver1', 'receiver2', 'receiver3', 'receiver4', 'source']:
        if column in data.columns:
            data[column] = data[column].apply(convert_to_array)

    for _, row in data.iterrows():
        test_case = {
            'arrival_times': row['arrival_times'],
            'stations': row['stations'],
            'receiver1': row['receiver1'],
            'receiver2': row['receiver2'],
            'receiver3': row['receiver3'],
            'receiver4': row['receiver4'],
            'source': row['source'],
        }
        test_data.append(test_case)

    return test_data



test_data = load_test_data('testing.csv')

class TestHyperbolicNavigator:
    @pytest.fixture
    def nav(self):
        return ThreeD_Navigator(knn_tdoa, scaler)

    @pytest.mark.parametrize("event", test_data)
    def test_solve(self, nav, event):
        arrival_times = list(event['arrival_times']) 
        stations = list(event['stations'])
        station_coords = [event['receiver1'], event['receiver2'], 
                          event['receiver3'], event['receiver4']]
        solution = nav.solve(arrival_times, stations, station_coords)
        distance = np.linalg.norm(solution - event['source'])
        assert distance < 100

    
