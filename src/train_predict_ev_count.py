import argparse
import random
import ast

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely import wkt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import accuracy_score, classification_report

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.output = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

    
def main():
    parser = argparse.ArgumentParser(description='EV Station Assignment')

    parser.add_argument('--input_nc_emb_path', type=str, default='./data/output/NC_EV_Registrations_ZIP_Count_embedding_autoenc_64.csv', help='Path to EV demand file')
    parser.add_argument('--input_ga_emb_path', type=str, default='./data/output/georgia_place_infrastructure_emb_zip_code_avg_enc64.csv', help='Path to POI file')
    parser.add_argument('--input_census_block_shp_path', type=str, default='./data/GA_cb_2018_13_bg_500k', help='Path to POI file')
    
    parser.add_argument('--output_csv', type=str, default='./data/output/georgia_place_infrastructure_emb_zip_code_avg_enc64_predicted_ev.csv', help='Path to output CSV')
    
    args = parser.parse_args()

    nc_df = pd.read_csv(args.input_nc_emb_path)
    nc_df.spabert_emb_autoenc_64 = nc_df.spabert_emb_autoenc_64.apply(lambda x: ast.literal_eval(x))

    georgia_df = pd.read_csv(args.input_ga_emb_path)
    georgia_df.spabert_emb_autoenc_64 = georgia_df.spabert_emb_autoenc_64.apply(lambda x: ast.literal_eval(x))
    
    X = nc_df.spabert_emb_autoenc_64.values
    y = nc_df.ev_count.values

    n_bins = 4
    kbin_discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_train = kbin_discretizer.fit_transform(y.reshape(-1, 1)).astype(int).ravel()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.tolist())
    X_test = georgia_df.spabert_emb_autoenc_64.values
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test.tolist())

    input_size = 64
    hidden_sizes = [32, 16]
    output_size = n_bins
    num_epochs = 6000
    batch_size = 64

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    model = MLPClassifier(input_size, hidden_sizes, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)

    georgia_df['predicted_ev_count_bin'] = predicted.tolist()

    local_boundary_gdf = gpd.read_file(args.input_census_block_shp_path, dtype={'GEOID': str})
    local_boundary_gdf = local_boundary_gdf.rename(columns={'GEOID': 'geohash'})
    local_boundary_gdf = local_boundary_gdf[['geohash', 'geometry']]
    local_boundary_gdf = local_boundary_gdf.to_crs("EPSG:4326")

    georgia_df.geohash = georgia_df.geohash.astype(str)
    georgia_df.merge(local_boundary_gdf, how='left', on='geohash').to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()