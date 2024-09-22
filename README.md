# ACM SIGSPATIAL Cup 2024

The code for the 13th SIGSPATIAL Cup competition (GISCUP 2024) [https://sigspatial2024.sigspatial.org/giscup/index.html](https://sigspatial2024.sigspatial.org/giscup/index.html).

The goal is to optimize the placement of EV charging stations. Our UMN-UL team proposes an approach that analyzes traffic demands and leverages point-of-interest contextual embeddings using a language model to predict EV registration rates per census block. 

## 1.

### Description

### Usage

- Code

  - Input Arguments

  - Output


## 2. Learning region embeddings to predict EV registration rate for each census block
### Description
We contextualize regions using one of the spatial language models, [SpaBERT](https://github.com/knowledge-computing/spabert). We use publicly available open map data, [Overture Maps](https://overturemaps.org/), to retrieve points of interest (POI) and infrastructure data for North Carolina and Georgia.

### Usage

- Code `python ./src/train_predict_ev_count.py`

  - Input Arguments
    - Overture Maps data in North Carolina and Georgia
      `./data/overturemap_{georgia/north_carolina}_{place/infrastructure}.csv`

    - SpaBERT's pretrained weights using North Carolina data
      `./poi_contextual/weights`

    - Learned region embeddings for each zip code in North Carolina and census block in Georgia
      `./poi_contextual/embeddings`
  - Output
    - Prediction results of EV count for each census block in Georgia


## 3.

### Description

### Usage
- Code
  - Input Arguments
  - Output
    - Estimated EV demand per census block in Georgia
      `./data/output/ev_demand.csv`


## 4. Assigning EV charging stations based on demand for each census block

### Description
We assign EV charging station locations based on estimated demand and a ranked list of relevant POI types, ordered as follows: `EV stations, Parking, Shopping Centers, Offices, Institutes, Hotels, Parks, Restaurants, Companies, Museums, and Golf Courses` EV charging stations are categorized into three capacity levels—1, 4, and 8—based on EV demand quantiles: 50%, 60%-80%, and 90%, respectively. 

### Usage

- `python ./src/assign_ev_station.py` --input_ev_demand_path <path_to_ev_demand_file> --input_poi_path <path_to_potential_ev_charging_station_file>

  - Arguments
    - `input_ev_demand_path`: Estimated EV demand per census block in Georgia `./data/output/ev_demand.csv`

    - `input_poi_path`: Filtered Overture Maps data in Georgia `./data/overturemap_georgia_potential_ev_stations.csv`

    - `input_ev_station_path`: Existing EV charging stations  in Georgia `./data/overturemap_georgia_potential_ev_stations.csv`

    - `input_census_block_shp_path`: Census blocks in Georgia `./data/cb_2018_13_bg_500k`

    - `output_gpkg`: Assigned EV charging stations (GPKG format) `./data/output/ev_charging_stations_UMN_UL.gpkg`

    - `output_ev_demand_assigned`: Updated EV demand per census file with assigned station count `./data/output/ev_demand_assigned.csv`


## 5. Adjusting EV charging stations for disadvantaged communities

### Description

### Usage
- Code
  - Input Arguments
  - Output
