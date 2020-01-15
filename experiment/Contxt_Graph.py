import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def channel_wise_normlization(tensor_3d):
    assert len(tensor_3d.shape) == 3

    tensor_norm = []
    for i in range(tensor_3d.shape[-1]):
        channel_norm = (tensor_3d[:,:,i] - tensor_3d[:,:,i].min()) / (tensor_3d[:,:,i].max() - tensor_3d[:,:,i].min())
        tensor_norm.append(channel_norm)

    tensor_norm = np.array(tensor_norm)
    tensor_norm = tensor_norm.transpose((1,2,0))   # channel_last

    return tensor_norm


def load_data(in_dir):
    # load data
    demo = np.load(os.path.join(in_dir, 'demo.npy'))
    poi = np.load(os.path.join(in_dir, 'POI.npy'))
    print('demo shape:', demo.shape, '\n', 'POI shape:', poi.shape, '\n')

    # normalize to 0~1 (channel-wise)
    demo_norm = channel_wise_normlization(demo)
    poi_norm = channel_wise_normlization(poi)

    print(f'demo_norm shape: {demo_norm.shape}, demo_norm range ({np.amin(demo_norm)}, {np.amax(demo_norm)})')
    print(f'POI_norm shape: {poi_norm.shape}, POI_norm range ({np.amin(poi_norm)}, {np.amax(poi_norm)})')
    print(f'POI_count shape: {poi.shape}, POI_count range ({np.amin(poi)}, {np.amax(poi)}) \n')

    return demo_norm, poi_norm, poi


def get_context_feature(out_path, graph, feature, sim_strat):
    graph_vec = graph.reshape((-1, graph.shape[-1]))

    # context graph on similarity
    n = graph_vec.shape[0]     # nodes
    A = np.zeros((n, n))     # adjacency matrix

    if sim_strat == 'recip_d':
        for i in range(n):
            for j in range(n):
                A[i, j] = 1 / (epsilon + np.linalg.norm(graph_vec[i] - graph_vec[j]))     # Reciprocal of Euclidean distance
    elif sim_strat == 'cos':
        for i in range(n):
            for j in range(n):
                A[i, j] = np.dot(graph_vec[i], graph_vec[j] / (
                            epsilon + np.linalg.norm(graph_vec[i]) * np.linalg.norm(graph_vec[j])))     # cosine similarity

    print(f'Adjacency matrix shape: {A.shape}, range ({np.amin(A)}, {np.amax(A)})')
    A = np.clip(A, 1e-08, 1)
    print(f'Clipped adjacency matrix shape: {A.shape}, range ({np.amin(A)}, {np.amax(A)})')

    # graph signal
    signal = feature.reshape((-1, feature.shape[-1]))
    assert A.shape[-1] == signal.shape[0]

    # check if A symmetric
    print('If matrix A symmetric:', check_symmetric(A))
    # check how many 0s in A
    print('#0:', np.count_nonzero(A == 0))

    # check distribution
    plt.hist(A.flatten())
    #plt.show()
    plt.savefig(os.path.join(out_path, 'A_similarity_distribution.png'))

    # save data
    np.save(os.path.join(out_path, 'A_sim.npy'), A)
    np.save(os.path.join(out_path, 'X_sig.npy'), signal)

    return None

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


epsilon = 0.001     # to avoid zero division


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Context graph feature generation')
    parser.add_argument('-in', '--in_dir', type=str, help='Input directory', default='../preprocessing/from_raw/raw')
    parser.add_argument('-out', '--out_dir', type=str, help='Output directory', default='../data/context')
    # graph generation strategies
    parser.add_argument('--POI_count', type=bool, default=False, help='Use normalized POI (False) or POI count (True)')
    parser.add_argument('-g', '--graph', type=str, choices=['POI', 'demo'], default='POI',
                        help='Identify feature for graph construction')
    parser.add_argument('-sim', '--sim_strat', type=str, choices=['recip_d', 'cos'], default='recip_d',
                        help='Identify strategy for similarity calculation')

    args = parser.parse_args()

    # load data
    demo_norm, poi_norm, poi_count = load_data(args.in_dir)

    print(
        f'Strategy: if POI normalized {not args.POI_count}; context graph on {args.graph}; similarity calculation {args.sim_strat}')
    # generate context features
    if not args.POI_count:     # normlized POI
        if args.graph == 'POI':
            if args.sim_strat == 'recip_d':  # strategy 1
                out_path = os.path.join(args.out_dir, 'strat1')     # try *pearson's r?
            elif args.sim_strat == 'cos':  # strategy 2
                out_path = os.path.join(args.out_dir, 'strat2')
            get_context_feature(out_path, poi_norm, demo_norm, args.sim_strat)

        elif args.graph == 'demo':
            if args.sim_strat == 'recip_d':  # strategy 3
                out_path = os.path.join(args.out_dir, 'strat3')
            elif args.sim_strat == 'cos':  # strategy 4
                out_path = os.path.join(args.out_dir, 'strat4')
            get_context_feature(out_path, demo_norm, poi_norm, args.sim_strat)

    else:     # POI count
        if args.graph == 'POI':
            if args.sim_strat == 'recip_d':  # strategy 5
                out_path = os.path.join(args.out_dir, 'strat5')
            elif args.sim_strat == 'cos':  # strategy 6
                out_path = os.path.join(args.out_dir, 'strat6')
            get_context_feature(out_path, poi_count, demo_norm, args.sim_strat)

        elif args.graph == 'demo':
            if args.sim_strat == 'recip_d':  # strategy 7
                out_path = os.path.join(args.out_dir, 'strat7')
            elif args.sim_strat == 'cos':  # strategy 8
                out_path = os.path.join(args.out_dir, 'strat8')
            get_context_feature(out_path, demo_norm, poi_count, args.sim_strat)

