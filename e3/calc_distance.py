import sys
from xml.dom import minidom
import pandas as pd
import numpy as np
from pykalman import KalmanFilter

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')

def element_to_data(element):
    lat = float(element.getAttribute('lat'))
    lon = float(element.getAttribute('lon'))
    return lat, lon

def get_data(filename):
    doc = minidom.parse(filename)
    elements = doc.getElementsByTagName('trkpt')
    data = pd.DataFrame(list(map(element_to_data,elements)),columns=['lat', 'lon'])
    return data

# reference of this function: 
# wikipedia page: https://en.wikipedia.org/wiki/Haversine_formula
# stackoverflow: https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def compute_dis(lat, lon, lat2, lon2):
    R2 = 12742
    deltalatD = np.deg2rad(lat2-lat)
    deltalonD = np.deg2rad(lon2-lon)
    a = (np.sin(deltalatD/2)*np.sin(deltalatD/2)) + np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(lat2))*np.sin(deltalonD/2)*np.sin(deltalonD/2)
    return R2*np.arcsin(np.sqrt(a))

def distance(data):
    data['lat2'] = data['lat'].shift(-1) 
    data['lon2'] = data['lon'].shift(-1)
    data['distance'] = compute_dis(data['lat'], data['lon'], data['lat2'], data['lon2'])
    return np.sum(data['distance'])*1000

def smooth(data):
    kalman_data = data[['lat', 'lon']]
    initial_state = kalman_data.iloc[0]
    ob_stddev = 4  
    trans_stddev = 2 
    dim = 2
    observation_covariance = ob_stddev**2 * np.identity(dim) # TODO: shouldn't be zero
    transition_covariance = trans_stddev**2 * np.identity(dim) # TODO: shouldn't be zero
    transition = np.identity(dim)
    kf = KalmanFilter(
    initial_state_mean=initial_state,
    initial_state_covariance=observation_covariance,
    observation_covariance=observation_covariance,
    transition_covariance=transition_covariance,
    transition_matrices=transition
    )
    kalman_smoothed, _ = kf.smooth(kalman_data)
    smoothedData = pd.DataFrame(kalman_smoothed, columns=['lat', 'lon'])
    return smoothedData

def main():
    filename = sys.argv[1]
    points = get_data(filename)
    print('Unfiltered distance: %0.2f' % (distance(points),))
    
    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')

if __name__ == '__main__':
    main()