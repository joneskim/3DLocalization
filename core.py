"""
Written by Joneskim Kimo for RNO-G vertex reconstruction
"""

import numpy as np
from scipy.optimize import minimize
from coordinate_system.coordinate_system import CoordinateSystem as cs

class ThreeD_Navigator:
    """
    Class to solve for the source location given the arrival times and station coordinates
    """
    arrival_times: list[float]
    station_coords: list[float]
    v: float
    knn_tdoa: "KNNRegressor"
    scaler: "Scaler"

    def __init__(self, model, scaler) -> None:
        self.v: float = 299792458.0
        self.knn_tdoa: "KNNRegressor" = model
        self.scaler: "Scaler" = scaler

    def solve(self, arrival_times: list[float], stations: list[int], station_coords: list[float]) -> list[float]:
        """
        Function to solve for the source location given the arrival times and station coordinates
        The method used is minization of the objective function which is a sum of squared errors
        between the measured TDOA and the calculated TDOA. The calculated TDOA is calculated using
        the distance between the candidate location and the station coordinates.

        The objective function is minimized using the Nelder-Mead method.

        The initial guess is calculated using the KNN model trained on simulated data. The KNN model
        is trained on the TDOA and Station labels. 

        To get even better results, the initial guess is perturbed by adding a random number to each
        of the coordinates. This is done 20 times and the best result is returned.

        The best result is the candidate location that minimizes the objective function.

        Parameters
        ----------
        arrival_times : list[float]
            List of arrival times in seconds

        stations : list[str]
            List of station names

        station_coords : list[list[float]]
            List of station coordinates

        Returns
        -------
        list[float]
            List of coordinates of the candidate location
        """
        def objective_function(candidate_location: list[float], stations_coords: list[float], 
                               arrival_times: list[float], speed_of_signal: float, 
                               max_possible_distance = 15000) -> float:
            """
            Objective function to minimize. The objective function is a sum of squared errors
            between the measured TDOA and the calculated TDOA. The calculated TDOA is calculated using
            the distance between the candidate location and the station coordinates.
            """
            total_error: float = 0
            for i in range(len(stations_coords)):
                for j in range(i + 1, len(stations_coords)):
                    dist_to_station_i = np.linalg.norm(candidate_location - stations_coords[i])
                    dist_to_station_j = np.linalg.norm(candidate_location - stations_coords[j])
                    tdoa = (dist_to_station_i - dist_to_station_j) / speed_of_signal
                    measured_tdoa = arrival_times[i] - arrival_times[j]
                    error = (tdoa - measured_tdoa) ** 2
                    total_error += error
            penalize: float = 0
            distance_from_origin: float = np.linalg.norm(candidate_location)
            if distance_from_origin > max_possible_distance:
                penalize = 1000 * (distance_from_origin - max_possible_distance) ** 2
            total_error += penalize
            return total_error

        initial_guess: float = self.calculate_initial_guess(stations, arrival_times)
        results: list[list[float]] = []
        result_: list[float] = minimize(objective_function, initial_guess, args=(station_coords, arrival_times, self.v), 
                           method='Nelder-Mead', tol=1e-6)
        
        results.append(result_.x)
        for _ in range(20):
            initial_guess = self.perturb_initial_guess(initial_guess)
            result = minimize(objective_function, initial_guess, args=(station_coords, arrival_times, self.v), 
                              method='Nelder-Mead', tol=1e-6)
            
            results.append(result.x)

        best_result: list[float] = min(results, key=lambda x: objective_function(x, station_coords, arrival_times, self.v))
        return best_result
    
    def perturb_initial_guess(self, initial_guess: list[float]) -> list[float]:
        """
        Perturbs the initial guess by adding a random number to each of the coordinates
        
        Parameters
        ----------
        initial_guess : list[float]
            List of coordinates of the initial guess
        
        Returns
        -------
        list[float]
            List of coordinates of the perturbed initial guess
        """
        x: float = initial_guess[0] + np.random.normal(-20, 20)
        y: float = initial_guess[1] + np.random.normal(-20, 20)
        z: float = initial_guess[2] + np.random.normal(-20, 20)
        return [x,y,z]

    def calculate_initial_guess(self, stations: list[int], arrival_times: list[float]) -> list[float]: 
        """
        Predicts the initial guess using the KNN model trained on simulated data.
        This is specific to the RNO-G. Stations 11, 12, 13, 21, 22, 23, 24.

        Parameters
        ----------
        stations : list[str]
            List of station names

        arrival_times : list[float]
            List of arrival times in seconds

        Returns
        -------
        float
            Initial guess for the candidate location
        """


        def compute_tdoas(arrival_times: list[float]) -> list[float]:
            """
            Helper function to compute the TDOAs from the arrival times in prep
            for the KNN model
            Parameters
            ----------
            arrival_times : list[float]
                List of arrival times in seconds

            Returns
            -------
            list[float]
                List of TDOAs
            """
            tdoas: list[float] = []
            tdoas.append(arrival_times[0] - arrival_times[1])
            tdoas.append(arrival_times[0] - arrival_times[2])
            tdoas.append(arrival_times[0] - arrival_times[3])
            return tdoas
       
        tdoas: list[float] = compute_tdoas(arrival_times)
        features_array: np.array = np.concatenate([stations, tdoas])
        scaled_features: "Scaler" = self.scaler.transform([features_array])
        initial_guess: list[float] = self.knn_tdoa.predict(scaled_features)
        return initial_guess[0] 
