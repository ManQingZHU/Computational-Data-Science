import sys
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file1 = sys.argv[1]
file2 = sys.argv[2]
output = sys.argv[3]

station_fh = gzip.open(file1, 'rt', encoding='utf-8')
stations = pd.read_json(station_fh, lines=True)
stations['avg_tmax'] = np.divide(stations['avg_tmax'],10)
#print(stations)

cities = pd.read_csv(file2).dropna(subset=["population", 'area'])
cities["area"] = np.divide(cities["area"], 1000000)
cities = cities[cities["area"] <= 10000]
cities["density"] = np.divide(cities["population"], cities["area"])
#print(cities)

def distance(city, stations):
    R2 = 12742000
    city_lat = np.deg2rad(city["latitude"])
    city_lon = np.deg2rad(city["longitude"])
    sta_lat = np.deg2rad(stations["latitude"])
    sta_lon = np.deg2rad(stations["longitude"])
    deltalat = np.deg2rad(sta_lat-city_lat)
    deltalon = np.deg2rad(sta_lon-city_lon)
    
    a = np.square(np.sin(deltalat/2)) + np.cos(np.deg2rad(city_lat))*np.cos(np.deg2rad(sta_lat))*np.square(np.sin(deltalon/2))
    return np.multiply(R2,np.arcsin(np.sqrt(a)))

def best_tmax(city, stations):
    dis = distance(city, stations)
    min_dis = np.argmin(dis)
    city["avg_tmax"] = stations.iloc[min_dis]["avg_tmax"]
    return city

cities = cities.apply(best_tmax, stations=stations, axis=1)

plt.plot(cities["avg_tmax"], cities["density"], 'b.', alpha=0.5)
plt.title("Temperature vs Population Density")
plt.xlabel("Avg Max Temperature (\u00b0C)")
plt.ylabel("Population Density (people/km\u00b2)")
plt.savefig(output)