# ACM SIGSPATIAL Cup 2024

The code for the 13th SIGSPATIAL Cup competition (GISCUP 2024) [https://sigspatial2024.sigspatial.org/giscup/index.html](https://sigspatial2024.sigspatial.org/giscup/index.html).

The goal is to optimize the placement of EV charging stations. Our UMN-UL team proposes an approach that analyzes traffic demands and leverages point-of-interest contextual embeddings using a language model to predict EV registration rates per census block. 

## 1.


## 2. Learning region embeddings to predict EV registration rate for each census block
### Description
We contextualize regions using one of the spatial language models [SpaBERT](https://github.com/knowledge-computing/spabert). We use publicly available open map data, [Overture Maps](https://overturemaps.org/), to retrieve points of interest (POI) and infrastructure data for North Carolina and Georgia.

### Usage
- Directories
  - Overture Maps data for North Carolina and Georgia
    `./data/overturemap_{STATE_NAME}_{place/infrastructure}`

  - SpaBERT's pretrained weights using North Carolina data
    `./weights`
    
- Code


## 3.


## 4.


## 5.
