import os
import argparse
import scipy.stats as stats
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from math import floor
import holidays
from preprocessing.Emerg2Grid import pgps_to_xy
from preprocessing.utils import heatmap


H = 20
W = 10
epsilon = 0.001     # to avoid zero division


def process_poi(reg_dir, poi_dir, args):
    grid = gpd.read_file(reg_dir)
    poi = gpd.read_file(poi_dir)

    print(poi.columns)
    #print(poi.iloc[0])
    print('POI categories:', poi['category'].unique())

    print(poi.shape[0])
    poi.dropna(inplace=True)
    poi.drop_duplicates(inplace=True)
    print('Drop invalids:', poi.shape[0])

    poi_lst = ['arts_entertainment', 'education', 'event', 'food', 'nightlife', 'parks_outdoors', 'professional', 'residence', 'shops', 'travel']
    maps = []
    #for item in poi['category'].unique():
    for item in poi_lst:
        poi_cat = poi[poi['category']==item]
        print(item, poi_cat.shape[0])
        category_map = np.zeros((H, W), dtype=int)

        for i, row in poi_cat.iterrows():
            lon, lat = row['location_1'], row['location.l']
            x, y = pgps_to_xy(lon, lat)  # calculate using origin array & inverse bias
            inside = (0 <= x <= 1) and (0 <= y <= 1)  # T/F whether point within
            if inside:
                grid_x = floor(x * H)
                grid_y = floor(y * W)
                category_map[grid_x, grid_y] += 1
        print(item, 'in range', np.sum(category_map), f'{round(np.sum(category_map)/poi_cat.shape[0], 4)*100}%')
        maps.append(category_map)

    maps_array = np.array(maps)
    print('Total in range', maps_array.shape)
    print(np.sum(maps_array), f'{round(np.sum(maps_array)/poi.shape[0], 4)*100}%')

    '''
    # print POI maps
    for i in range(maps_array.shape[0]):
        plt.imshow(maps_array[i])
        #plt.show()
        plt.savefig(os.path.join(args.aim_dir, 'images', 'poi', f'{poi_lst[i]}.png'))
    '''
    maps_array = maps_array.transpose((1,2,0))     # category_last
    print('Transposed shape:', maps_array.shape)

    # save POI data
    #np.save(os.path.join(args.out_dir, 'POI.npy'), maps_array)

    return None


def process_demo(reg_dir, demo_dir, ct_dir, args):
    grid = gpd.read_file(reg_dir)
    demo = pd.read_csv(demo_dir)
    ct = gpd.read_file(ct_dir)

    # demo on census tract
    print(demo.iloc[0], ct.iloc[0])
    print(demo.dtypes, ct.dtypes)
    ct['GEO_num'] = ct['GEOID'].apply(lambda x: int(x))
    print(ct.dtypes)

    demo_join = demo.join(ct.set_index('GEO_num'), on='GEO.id2')
    print(demo_join.iloc[0])
    print(demo_join.shape[0])
    print(demo_join[demo_join['geometry'].isnull()].shape[0])

    # calculate population density
    demo_join['pop_density'] = demo_join.apply(
        lambda x: x['population']/x['ALAND']if x['ALAND']!=0 else x['population']/epsilon, axis=1)

    # convert to geodataframe
    demo_ct_geo = gpd.GeoDataFrame(demo_join, crs={'init': 'epsg:4326'}, geometry=demo_join['geometry'])
    print(demo_join.iloc[0])

    # grid
    #grid['wkt'] = grid['geometry'].apply(lambda x: x.wkt).values
    #print(grid.dtypes)
    print(grid.iloc[0])
    print(type(demo_ct_geo), type(grid))
    print(grid['id'])

    df_grid = pd.DataFrame(columns=['population', 'sex_ratio', 'median_age', 'senior_ratio', '%white', '%black', '%asian',
                                    '%hispanic', 'fract', '%poverty', '%unemployment', 'median_income', 'gini',
                                    '%no_high_school', '%bachelor', '%same_house', 'mean_bldg_age', '%occupancy',
                                    '%ownership', 'pop_density'])
    for i, grid_unit in grid.iterrows():
        assert i == grid_unit['id'] - (8 + (8-1)*(i//H)), f"{i}, {grid_unit['id']}"     # ensure ordering of the grids
        #grid_index = pd.Series([i])
        grid_overlay = pd.DataFrame(columns=['population', 'sex_ratio', 'median_age', 'senior_ratio', '%white', '%black',
                                    '%asian', '%hispanic', 'fract', '%poverty', '%unemployment', 'median_income',
                                    'gini', '%no_high_school', '%bachelor', '%same_house', 'mean_bldg_age', '%occupancy',
                                    '%ownership', 'pop_density', 'geometry'])

        for j, ct_unit in demo_ct_geo.iterrows():
            if grid_unit['geometry'].intersects(ct_unit['geometry']):
                #print(ct_unit)
                #print(i, j)
                grid_overlay = grid_overlay.append(ct_unit, ignore_index=True)
        grid_overlay = grid_overlay.astype({'population': 'int32'})     # col population recognized as object
        print(f'Finish grid {i}, #overlayed census tracts {grid_overlay.shape[0]}')
        print(grid_overlay.mean(numeric_only = True))
        grid_mean = grid_overlay.mean(numeric_only = True)
        #grid_index = grid_index.append(grid_overlay, ignore_index=True)
        df_grid = df_grid.append(grid_mean, ignore_index=True)

    df_grid = df_grid.drop(['ALAND', 'AWATER', 'GEO.id2', 'Unnamed: 0'], axis=1)
    print(df_grid.head())
    print(df_grid.shape)
    # save grid-aggregated demo data
    df_grid.to_csv(os.path.join(args.aim_dir, f'demo{str(args.year)[-2:]}_grid.csv'))

    return None


def demo_pd2np(demo_grid_dir):
    demo = pd.read_csv(demo_grid_dir)

    print(demo.columns)
    # find vacant pixel
    print(demo[demo.isna().any(axis=1)])
    demo.fillna(0, inplace=True)
    print('After fill NA', demo[demo.isna().any(axis=1)])

    C = demo.shape[1] - 1
    demo_grid = np.zeros((H, W, C))
    for i, row in demo.iterrows():
        h = i % H
        w = i // H
        for c in range(C):
            demo_grid[h, w, c] = row.iloc[c+1]

    print(demo_grid.shape)
    demo_lst = ['population', 'sex_ratio', 'median_age', 'senior_ratio', '%white', '%black', '%asian', '%hispanic',
                'fract', '%poverty', '%unemployment', 'median_income', 'gini_coeff', '%no_high_school', '%bachelor',
                '%same_house', 'mean_bldg_age', '%occupancy', '%ownership', 'pop_density']
    '''
    # print demo maps
    for i in range(demo_grid.shape[-1]):
        plt.imshow(demo_grid[:, :, i])
        #plt.show()
        plt.savefig(os.path.join(args.aim_dir, 'images', 'demo', f'{demo_lst[i]}.png'))
    '''
    np.save(os.path.join(args.out_dir, 'demo.npy'), demo_grid)

    return None


def get_pearson(args):
    emerg_dir = f'../data/{args.t_interval}h/EmergNYC_bi_{H}x{W}.npy'
    emerg_data = np.load(emerg_dir)
    print(emerg_data.shape)

    # all emergency
    emerg_lst = ['violation', 'misdemeanor', 'felony', 'EMS', 'rescue', 'fire']
    emerg_maps = np.zeros((H, W, emerg_data.shape[-1]))
    for h in range(H):
        for w in range(W):
            for c in range(emerg_data.shape[-1]):
                emerg_maps[h, w, c] = np.sum(emerg_data[:, h, w, c])
    '''
    for i in range(emerg_maps.shape[-1]):
        plt.imshow(emerg_maps[:, :, i])
        #plt.show()
        plt.savefig(os.path.join(args.aim_dir, 'figures', 'emerg', f'{emerg_lst[i]}.png'))
    '''
    emerg = emerg_maps.reshape((-1, emerg_maps.shape[-1]))

    # demo
    demo_dir = os.path.join(args.out_dir, 'demo.npy')
    demo_maps = np.load(demo_dir)
    demo = demo_maps.reshape((-1, demo_maps.shape[-1]))

    print(demo.shape, emerg.shape)
    demo_lst = ['population', 'sex_ratio', 'median_age', 'senior_ratio', '%white', '%black', '%asian', '%hispanic',
                'fract', '%poverty', '%unemployment', 'median_income', 'gini_coeff', '%no_high_school', '%bachelor',
                '%same_house', 'mean_bldg_age', '%occupancy', '%ownership', 'pop_density']
    pearson_demo = []
    for i in range(len(demo_lst)):
        row =[]
        for e in range(len(emerg_lst)):
            r = stats.pearsonr(demo[:, i], emerg[:, e])[0]
            print(f'{demo_lst[i]}~{emerg_lst[e]}: {r}')
            row.append(r)
        pearson_demo.append(row)
    pearson_demo_np = np.array(pearson_demo)
    print(pearson_demo_np.shape)

    # print avg. pearson's r
    print("Pearson's r - row mean")
    print(np.mean(pearson_demo_np, axis=1))

    fig, ax = plt.subplots()
    im, cbar = heatmap.heatmap(pearson_demo_np, demo_lst, emerg_lst, ax=ax, vmin=-0.3, vmax=0.5,
                               cmap='seismic', cbarlabel='Correlation coeff.')
    # texts = heatmap.annotate_heatmap(im, valfmt="{x:.1f} t")
    plt.title("Pearson's Correlation Coefficients - Demo/Socio & Emergency")
    fig.tight_layout()
    plt.show()
    #plt.savefig(os.path.join(args.aim_dir, 'images', 'demo', 'demo_pearson_r.png'))


    # poi
    poi_dir = os.path.join(args.out_dir, 'POI.npy')
    poi_maps = np.load(poi_dir)
    poi = poi_maps.reshape((-1, poi_maps.shape[-1]))
    # normalize
    poi_norm = np.zeros((H*W, poi_maps.shape[-1]))
    for r in range(H*W):
        for c in range(poi_maps.shape[-1]):
            poi_norm[r, c] = poi[r, c]/np.sum(poi[r, :]) if np.sum(poi[r, :])!=0 else poi[r, c]/epsilon

    print(poi_norm.shape, emerg.shape)
    poi_lst = ['%arts_entertainment', '%education', '%event', '%food', '%nightlife',
               '%parks_outdoors', '%professional', '%residence', '%shops', '%travel']

    pearson_poi = []
    for i in range(len(poi_lst)):
        row =[]
        for e in range(len(emerg_lst)):
            r = stats.pearsonr(poi_norm[:, i], emerg[:, e])[0]
            print(f'{poi_lst[i]}~{emerg_lst[e]}: {r}')
            row.append(r)
        pearson_poi.append(row)

    pearson_poi_np = np.array(pearson_poi)
    print(pearson_poi_np.shape)

    fig, ax = plt.subplots()
    im, cbar = heatmap.heatmap(pearson_poi_np, poi_lst, emerg_lst, ax=ax, vmin=-0.3, vmax=0.5,
                               cmap='seismic', cbarlabel= 'Correlation coeff.')
    #texts = heatmap.annotate_heatmap(im, valfmt="{x:.1f} t")
    plt.title("Pearson's Correlation Coefficients - POI & Emergency")
    fig.tight_layout()
    plt.show()
    #plt.savefig(os.path.join(args.aim_dir, 'images', 'poi', 'poi_pearson_r.png'))

    return None


def get_context_feature(args):
    demo = np.load(os.path.join(args.out_dir, 'demo.npy'))     # normalized
    poi = np.load(os.path.join(args.out_dir, 'POI.npy'))       # normalized POI vs. pure POI counts?
    print('demo shape:', demo.shape, f'demo range ({np.amin(demo)}, {np.amax(demo)}) \n',
          'POI shape:', poi.shape, f'POI range ({np.amin(poi)}, {np.amax(poi)}) \n')

    # concate
    #attributes = np.concatenate((demo, poi), axis=-1)
    #print('attributes shape:', attributes.shape)
    #np.save(os.path.join(args.aim_dir, 'Attribute', 'attributes_POInorm.npy'), attributes)

    # normalize to 0~1
    demo_norm = demo / demo.max(axis=0)
    print(f'demo_norm shape: {demo_norm.shape}, demo_norm range ({np.amin(demo_norm)}, {np.amax(demo_norm)})')
    poi_norm = poi / poi.max(axis=0)
    print(f'POI_norm shape: {poi_norm.shape}, POI_norm range ({np.amin(poi_norm)}, {np.amax(poi_norm)})')

    demo_norm_vec = demo_norm.reshape((-1, demo_norm.shape[-1]))

    # calculate node cosine similarity - on demo/socio factors
    n_nodes = demo_norm_vec.shape[0]
    demo_sim = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            demo_sim[i, j] = np.dot(demo_norm_vec[i], demo_norm_vec[j]/(epsilon + np.linalg.norm(demo_norm_vec[i])*np.linalg.norm(demo_norm_vec[j])))
    print(f'Cosine similarity shape: {demo_sim.shape}, cos_sim range ({np.amin(demo_sim)}, {np.amax(demo_sim)})')

    poi_feat = poi_norm.reshape((-1, poi_norm.shape[-1]))
    assert demo_sim.shape[-1] == poi_feat.shape[0]

    # check if A (demo_sim) symmetric
    print('If matrix A symmetric:', check_symmetric(demo_sim))
    # check how many 0s in A
    #print(demo_sim)
    print('#0:', np.count_nonzero(demo_sim==0))
    # check distribution
    #plt.hist(demo_sim.flatten())
    #plt.show()
    # clip min to 1e-08
    demo_sim = np.clip(demo_sim, 1e-08, 1)
    print(f'Clipped cos_sim range ({np.amin(demo_sim)}, {np.amax(demo_sim)})')

    #np.save('../Data/A_demo_sim.npy', demo_sim)
    #np.save('../Data/X_poi_feat.npy', poi_feat)

    return None

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NYC attribute data preprocessing')
    parser.add_argument('-in', '--aim_dir', type=str, help='Aim directory', default='./from_raw/static')
    parser.add_argument('-out', '--out_dir', type=str, help='Output directory', default='./from_raw/raw')
    parser.add_argument('-type', '--contxt_type', type=str, choices=['demo', 'POI', 'pass'],
                        default='pass', help='Context type')
    parser.add_argument('-y', '--year', type=int, default=2015, help='Aim year')
    parser.add_argument('-m', '--month_span', type=int, default=12, help='How many months')
    parser.add_argument('-t', '--t_interval', type=int, default=4, help='Time interval in hour(s)')

    args = parser.parse_args()

    # dir grid data
    reg_dir = os.path.join(args.aim_dir, 'Grid_900res_20x10_WGS84.geojson')

    if args.contxt_type == 'POI':
        poi_dir = os.path.join(args.aim_dir, 'POI.geojson')
        process_poi(reg_dir, poi_dir, args)
    elif args.contxt_type == 'demo':
        demo_dir = os.path.join(args.aim_dir, f'demo{str(args.year)[-2:]}.csv')
        ct_dir = os.path.join(args.aim_dir, 'CensusTractNYC.geojson')
        #process_demo(reg_dir, demo_dir, ct_dir, args)
        demo_grid_dir = os.path.join(args.aim_dir, f'demo{str(args.year)[-2:]}_grid.csv')
        demo_pd2np(demo_grid_dir)
    else:
        pass

    # Pearson's R
    #get_pearson(args)

    # generate context features
    get_context_feature(args)
