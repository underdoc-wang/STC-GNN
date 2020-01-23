import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_spatial_graph(data, graph_channel):
    rest_data = np.delete(data, obj=graph_channel, axis=-1)
    graph_map = data[:,:,graph_channel]

    graph_feature = graph_map.reshape((-1, 1))     # reshape for kmeans input
    # spatial clustering
    km_model = KMeans(n_clusters=k)
    km = km_model.fit_predict(graph_feature)

    km_map = km.reshape(graph_map.shape)   # reshape back to grid map
    plt.imshow(km_map)
    plt.show()

    # onehot coding
    km = km.astype('str')
    km = pd.get_dummies(km).to_numpy()
    spatial_graph = km.reshape((graph_map.shape[0], graph_map.shape[1], k))
    print('Spatial graph feature: ', spatial_graph.shape, '\n',
          'Rest data shape: ', rest_data.shape)

    return spatial_graph, rest_data


def get_time_invariants(timestamps, feat_lst):
    static_feat = np.concatenate(feat_lst, axis=-1)
    print('All time-invariant features: ', static_feat.shape)
    H, W, C = static_feat.shape

    static_feat_seq = np.broadcast_to(static_feat, (timestamps, H, W, C))
    print('Time-invariant feature sequence: ', static_feat_seq.shape)

    return static_feat_seq


# global
delta_t = 4     # time interval
graph_channel = -1     # stand for "population density" in demo data
k = 3       # number of clusters

emerg_dir = f'../../../data/{delta_t}h/EmergNYC_bi_20x10.npy'
demo_dir = '../../../preprocessing/from_raw/raw/demo.npy'
poi_dir = '../../../preprocessing/from_raw/raw/POI.npy'
out_dir = f'../../../data/{delta_t}h/Hetero_invar_feat.npy'


if __name__ == '__main__':
    emerg_data = np.load(emerg_dir)
    # timestamps
    timestamps = emerg_data.shape[0]

    demo_data = np.load(demo_dir)
    poi_data = np.load(poi_dir)

    spatial_graph, rest_demo = get_spatial_graph(demo_data, graph_channel)

    # get all time variant features
    time_invariant_features = get_time_invariants(timestamps, [spatial_graph, rest_demo, poi_data])

    # save feature
    np.save(out_dir, time_invariant_features)