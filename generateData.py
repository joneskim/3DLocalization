import random
# import numpy as np
import cupy as np
import json
from coordinate_system.coordinate_system import CoordinateSystem as cs # from rno-g tools
import pandas as pd

class NavSimulator: 
    def __init__(self, stations = [11, 12, 13]):
        
        self.stations_dict = {
        11: [72.589227, -38.502299],
        12: [72.600087, -38.496227],
        13: [72.610947, -38.490147],
        21: [72.587406, -38.466030],
        22: [72.598265, -38.459936],
        23: [72.609124, -38.453833],
        24: [72.619983, -38.447723],
        # 'Disc Borehole': [72.589227, -38.502299],
        }
        self.local = cs()
        coords, actual_nums = self.generateReceivers()
        self.coord1, self.coord2, self.coord3, self.coord4 = coords
        self.actual_nums = actual_nums

    def generateReceivers(self):
        ix1, ix2, ix3, ix4 = random.sample(range(len(self.stations_dict)), 4)
        station1 = list(self.stations_dict.keys())[ix1]
        station2 = list(self.stations_dict.keys())[ix2]
        station3 = list(self.stations_dict.keys())[ix3]
        station4 = list(self.stations_dict.keys())[ix4]

        actual_nums = [station1, station2, station3, station4]
        
        coord1 = self.local.geodetic_to_enu(*self.stations_dict[station1])[:3]
        coord2 = self.local.geodetic_to_enu(*self.stations_dict[station2])[:3]
        coord3 = self.local.geodetic_to_enu(*self.stations_dict[station3])[:3]
        coord4 = self.local.geodetic_to_enu(*self.stations_dict[station4])[:3]
        
        coord1 = np.array(coord1)
        coord2 = np.array(coord2)
        coord3 = np.array(coord3)
        coord4 = np.array(coord4)

        return (coord1, coord2, coord3, coord4), actual_nums

    def generateTestCoordsForSource(self):
        
        x = random.uniform(-10000, 10000)
        y = random.uniform(-10000, 10000)
        z = random.uniform(-200, 10000)

        return np.array([x, y, z])

    def generateArrivalTimes(self, source):
        receiver_coords = [self.coord1, self.coord2, self.coord3, self.coord4]
        arrival_times = []
        
        for coord in receiver_coords:
            distance = np.linalg.norm(coord - source)
            arrival_time = distance / 299792458.0
            arrival_times.append(arrival_time)

        return arrival_times
    
    def generateTestData(self, start, end):
        events = []
        for i in range(start, end):
            source = self.generateTestCoordsForSource()
            arrival_times = self.generateArrivalTimes(source)
            event = {
                "stations": self.actual_nums,
                "receiver1": self.coord1.get().tolist(),
                "receiver2": self.coord2.get().tolist(),
                "receiver3": self.coord3.get().tolist(),
                "receiver4": self.coord4.get().tolist(),
                "source": source.get().tolist(),
                "arrival_times": [time.item() for time in arrival_times]
            }
            events.append(event)
            coords, actual_nums = self.generateReceivers()
            self.coord1, self.coord2, self.coord3, self.coord4 = coords
            self.actual_nums = actual_nums
        return events

    def saveTestData(self, filename, n=100, batch_size=10):
        first_batch = True  # Checking if its the first batch to write to file

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            data_batch = self.generateTestData(start, end)
            dataframe = pd.DataFrame(data_batch)

            if first_batch:
                dataframe.to_csv(filename, mode='w', header=True, index=False)
                first_batch = False
            else:
                dataframe.to_csv(filename, mode='a', header=False, index=False)
    
    def savedf(self, filename, n=100, batch_size=10):
        df = self.saveTestData(filename, n, batch_size)
        df.to_csv(filename[:-5] + '.csv')



s = NavSimulator()
s.saveTestData("testing.csv", 100)

