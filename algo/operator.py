"""
   Copyright 2022 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

__all__ = ("Operator", )

import util
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import kneed
import os
from itertools import chain
import pickle
from collections import defaultdict

class Operator(util.OperatorBase):
    def __init__(self, device_id, data_path, device_name='das Gerät'):
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        self.device_id = device_id
        self.device_name = device_name

        self.hourly_consumption_list_dict = defaultdict(list)

        self.consumption_same_hour = []

        self.current_hour = None

        self.hourly_consumption_clustering = {}

        self.clustering_file_path = f'{data_path}/{self.device_id}_clustering.pickle'
        self.epsilon_file_path = f'{data_path}/{self.device_id}_epsilon.pickle'
        self.hourly_consumption_list_file_path = f'{data_path}/{self.device_id}_hourly_consumption_list.pickle'

    def todatetime(self, timestamp):
        if str(timestamp).isdigit():
            if len(str(timestamp))==13:
                return pd.to_datetime(int(timestamp), unit='ms')
            elif len(str(timestamp))==19:
                return pd.to_datetime(int(timestamp), unit='ns')
        else:
            return pd.to_datetime(timestamp)

    def update_hourly_consumption_list_dict(self):
        min_index = np.argmin([float(datapoint['Consumption']) for datapoint in self.consumption_same_hour])
        max_index = np.argmax([float(datapoint['Consumption']) for datapoint in self.consumption_same_hour])
        hour_consumption_max = float(self.consumption_same_hour[max_index]['Consumption'])
        hour_consumption_min = float(self.consumption_same_hour[min_index]['Consumption'])
        #day_consumption_max_time = self.todatetime(self.consumption_same_day[max_index]['energy_time']).tz_localize(None)
        #day_consumption_min_time = self.todatetime(self.consumption_same_day[min_index]['energy_time']).tz_localize(None)
        overall_hourly_consumption = hour_consumption_max-hour_consumption_min
        self.hourly_consumption_list_dict[self.current_hour.hour-1].append((self.current_hour-pd.Timedelta(1,'hour'), overall_hourly_consumption))
        with open(self.hourly_consumption_list_file_path, 'wb') as f:
            pickle.dump(self.hourly_consumption_list_dict, f)
        return

    def determine_epsilon(self):
        neighbors = NearestNeighbors(n_neighbors=10)
        neighbors_fit = neighbors.fit(np.array([hourly_consumption for _, hourly_consumption in self.hourly_consumption_list_dict[self.current_hour.hour-1]]).reshape(-1,1))
        distances, _ = neighbors_fit.kneighbors(np.array([hourly_consumption for _, hourly_consumption in self.hourly_consumption_list_dict[self.current_hour.hour-1]]).reshape(-1,1))
        distances = np.sort(distances, axis=0)
        distances_x = distances[:,1]
        kneedle = kneed.KneeLocator(np.linspace(0,1,len(distances_x)), distances_x, S=0.9, curve="convex", direction="increasing")
        epsilon = kneedle.knee_y
        with open(self.epsilon_file_path, 'wb') as f:
            pickle.dump(epsilon, f)
        return epsilon

    def create_clustering(self, epsilon):
        self.hourly_consumption_clustering[self.current_hour.hour-1] = DBSCAN(eps=epsilon, min_samples=10).fit(np.array([hourly_consumption 
                                                                     for _, hourly_consumption in self.hourly_consumption_list_dict[self.current_hour.hour-1]]).reshape(-1,1))
        with open(self.clustering_file_path, 'wb') as f:
            pickle.dump(self.hourly_consumption_clustering, f)
        return self.hourly_consumption_clustering[self.current_hour.hour-1].labels_
    
    def test_hourly_consumption(self, clustering_labels):
        anomalous_indices = np.where(clustering_labels==clustering_labels.min())[0]
        quartile_3 = np.quantile([hourly_consumption for _, hourly_consumption in self.hourly_consumption_list_dict[self.current_hour.hour-1]],0.75)
        anomalous_indices_high = [i for i in anomalous_indices if self.hourly_consumption_list_dict[self.current_hour.hour-1][i][1] > quartile_3]
        if len(self.hourly_consumption_list_dict[self.current_hour.hour-1])-1 in anomalous_indices:
            print(f'In der letzten Stunde wurde durch {self.device_name} ungewöhnlich viel Strom verbraucht.')
        return [self.hourly_consumption_list_dict[self.current_hour.hour-1][i] for i in anomalous_indices_high]
    
    def run(self, data, selector='energy_func'):
        timestamp = self.todatetime(data['Time']).tz_localize(None)
        print('energy: '+str(data['Consumption'])+'  '+'time: '+str(timestamp))
        self.current_hour = timestamp.floor('h')
        if self.consumption_same_hour == []:
            self.consumption_same_hour.append(data)
            return
        elif self.consumption_same_hour != []:
            if self.current_hour==self.todatetime(self.consumption_same_hour[-1]['Time']).tz_localize(None).floor('h'):
                self.consumption_same_hour.append(data)
                return
            else:
                self.update_hourly_consumption_list_dict()
                if len(self.hourly_consumption_list_dict[self.current_hour.hour-1]) >= 24:
                    epsilon = self.determine_epsilon()
                    clustering_labels = self.create_clustering(epsilon)
                    days_with_excessive_consumption_during_this_hour_of_day = self.test_hourly_consumption(clustering_labels)
                    self.consumption_same_hour = [data]                 
                    if self.current_hour-pd.Timedelta(1,'hour') in list(chain.from_iterable(days_with_excessive_consumption_during_this_hour_of_day)):
                        return {'value': f'Nachricht vom {str(timestamp.date())} um {str(timestamp.hour)}:{str(timestamp.minute)} Uhr: In der letzten Stunde wurde übermäßig viel Energie durch das Gerät verbraucht.'} # Excessive hourly consumption detected.
                    else:
                        return  # No excessive hourly consumtion yesterday.
                else:
                    self.consumption_same_hour = [data]
                    return
