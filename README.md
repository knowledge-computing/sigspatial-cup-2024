# ACM SIGSPATIAL Cup 2024

The code for the 13th SIGSPATIAL Cup competition (GISCUP 2024) [https://sigspatial2024.sigspatial.org/giscup/index.html](https://sigspatial2024.sigspatial.org/giscup/index.html)

The goal is to optimize the placement of EV charging stations. Our UMN-UL team proposes an approach that analyzes traffic demands and leverages point-of-interest contextual embeddings using a language model to predict EV registration rates per census block. 


## 0. Preparing publicly available and provided datasets
- Datasets
  - Census block shapefile for Georgia (GA)
  `./data/GA_cb_2018_13_bg_500k`
  - Census block shapefile for North Carolina (NC)
  `./data/NC_cb_2018_37_bg_500k`
  - Justice 40 shapefile for GA
  `./data/Georgia-Justice40-map`
  - Existing charging stations for GA
  `./data/georgia_ev_stations.csv`


## 1. Learning region embeddings to predict EV registration rate for each census block
### Description

We leverage a spatial language model, [SpaBERT](https://github.com/knowledge-computing/spabert), to learn and extract contextual region embeddings for NC zip code areas and GA census blocks. Next, we apply an autoencoder for dimensionality reduction of the extracted region embeddings and employ a multi-layer perceptron to train and predict the EV registration count for each area.

### Usage

- `python ./src/train_predict_ev_count.py`

  - Arguments
    - `input_nc_emb_path`: Learned region embeddings for NC zip code and EV registration count `./data/output/NC_EV_Registrations_ZIP_Count_embedding_autoenc_64.csv`

    - `input_ga_emb_path`: Learned region embeddings for GA census block `./data/output/georgia_place_infrastructure_emb_zip_code_avg_enc64.csv`

    - `input_census_block_shp_path`: Census blocks in GA `./data/GA_cb_2018_13_bg_500k`

    - `output_csv`: Predicted EV registration count per GA census block `./data/output/georgia_place_infrastructure_emb_zip_code_avg_enc64_predicted_ev.csv`


## 2. Predicting EV charging demand at the census block for GA

### Description
We use the OD matrix from GA DOT to estimate the EV charging demand per census block. Based on the EV registration estimated from step 2, we know how many EV trips originated from each census block. Then we proportionally assign those EV trips according to the trip distribution from the OD matrix. Since the OD matrix is sparse, we only keep top 5 destinations with the most trips per origin. At last, the EV charging demand is the maximum of the EV registration number and the EV trips, considering the charging needs for both home and travel.  

### Usage  
- Jupyter Notebook `./src/charging_demand_prediction.ipynb`
  - Input Dataset
    - Provided NHTS NextGen GA OD data

  - Output
    - Estimated EV demand per census block in GA `./data/output/ev_demand.csv`


## 3. Assigning EV charging stations based on demand for each census block

### Description
We assign EV charging station locations based on estimated demand and a ranked list of relevant POI types, ordered as follows: `EV stations, Parking, Shopping Centers, Offices, Institutes, Hotels, Parks, Restaurants, Companies, Museums, and Golf Courses` EV charging stations are categorized into three capacity levels—1, 4, and 8—based on EV demand quantiles: 50%, 60%-80%, and 90%, respectively. 

### Usage

- `python ./src/assign_ev_station.py`

  - Arguments
    - `input_ev_demand_path`: Estimated EV demand per census block in GA `./data/output/ev_demand.csv`

    - `input_poi_path`: Filtered Overture Maps data in GA `./data/overturemap_georgia_potential_ev_stations.csv`

    - `input_ev_station_path`: Existing EV charging stations in GA `./data/georgia_ev_stations.csv`

    - `input_census_block_shp_path`: Census blocks in GA `./data/GA_cb_2018_13_bg_500k`

    - `output_gpkg`: Assigned EV charging stations (GPKG format) `./data/output/ev_charging_stations_UMN_UL.gpkg`

    - `output_ev_demand_assigned`: Updated EV demand per census file with assigned station count `./data/output/ev_demand_assigned.csv`


## 4. Adjusting EV charging stations for disadvantaged communities

### Description
We estimate the charging station distributions between disadvantaged communities (DACs) and non-disadvantaged communities (Non-DACs) according to the definition of Justice40. 

The comparison of the final estimated number of charging stations is as follows:

- DAC: 2.45 (Mean) / 4.70 (Std)
- Non-DAC: 2.62 (Mean) / 5.08 (Std)

These results suggest a fairly balanced distribution of charging stations.

### Usage
- Jupyter Notebook `./src/charging_demand_prediction.ipynb`
  - Input Dataset
    - Justice40 disadvantage census tracts `./data/Georgia-Justice40-map`


