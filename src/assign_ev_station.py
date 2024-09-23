import argparse
import random

import pandas as pd
import geopandas as gpd

from shapely import wkt

def convert_to_point(geometry):
    if geometry.geom_type == 'Polygon':
        return geometry.centroid
    elif geometry.geom_type == 'LineString':
        return geometry.interpolate(0.5, normalized=True)
    else:
        return geometry 

    
def estimated_ev_chargers(x):
    if x <= 2:  # for the 0.5 quantile
        return x / 1
    elif x <= 6:  # between 0.6 and 0.8 quantile
        return x / 4
    elif x > 6:  # 0.9 quantile and above
        return x / 8

    
def assign_points_to_polygons(polygons_df, points_df, count_column, type_priority):
    assigned_points = []

    for index, polygon_row in polygons_df.iterrows():
        polygon = polygon_row['geometry']
        count = polygon_row[count_column]
        points_within_polygon = points_df[points_df.within(polygon)]

        if count <= 0 or points_within_polygon.empty:
            continue

        selected_points = pd.DataFrame()
        
        for point_type in type_priority:
            points_of_type = points_within_polygon[points_within_polygon['type'] == point_type]
            remaining_count = count - len(selected_points)

            if remaining_count <= 0:
                break

            selected_points = pd.concat([
                selected_points, 
                points_of_type.head(remaining_count)
            ])
            
        if len(selected_points) < count:
            remaining_count = count - len(selected_points)
            substrings = ['restaurant', 'company', '_park', 'museum', 'golf_course']
            
            for substring in substrings:
                if remaining_count <= 0:
                    break

                matching_points = points_within_polygon[points_within_polygon['type'].str.contains(substring, case=False, na=False)]
                
                selected_points = pd.concat([
                    selected_points,
                    matching_points.head(remaining_count)
                ])
                
                remaining_count = count - len(selected_points)
                

        polygons_df.loc[index, 'assigned_ev_station'] = len(selected_points)

        if not selected_points.empty:
            if count <= 2:
                selected_points['capacity'] = 1
            elif count <= 6:
                selected_points['capacity'] = 4
            elif count > 6:
                selected_points['capacity'] = 8
                
            assigned_points.append(selected_points)
            
    assigned_points_gdf = gpd.GeoDataFrame(pd.concat(assigned_points, ignore_index=True), crs=points_df.crs)
    
    return assigned_points_gdf
    
def main():
    parser = argparse.ArgumentParser(description='EV Station Assignment')
    parser.add_argument('--input_ev_demand_path', type=str, default='./data/output/ev_demand.csv', help='Path to EV demand file')
    parser.add_argument('--input_poi_path', type=str, default='./data/overturemap_georgia_potential_ev_stations.csv', help='Path to POI file')
    parser.add_argument('--input_ev_station_path', type=str, default='./data/georgia_ev_stations.csv', help='Path to POI file')
    parser.add_argument('--input_census_block_shp_path', type=str, default='./data/GA_cb_2018_13_bg_500k', help='Path to POI file')
    
    parser.add_argument('--output_gpkg', type=str, default='./data/output/ev_charging_stations_UMN_UL.gpkg', help='Path to output GPKG')
    parser.add_argument('--output_ev_demand_assigned', type=str, default='./data/output/ev_demand_assigned.csv', help='Path to output GPKG')
    
    args = parser.parse_args()
    
    osm_df = pd.read_csv(args.input_poi_path)
    osm_df['geometry'] = osm_df['geometry'].apply(wkt.loads)
    osm_df = gpd.GeoDataFrame(osm_df, geometry='geometry')
    osm_df = osm_df.drop_duplicates(['ogc_fid'])
    osm_df['geometry'] = osm_df['geometry'].apply(convert_to_point)

    ev_df = pd.read_csv(args.input_ev_station_path)
    ev_df = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(ev_df.Longitude, ev_df.Latitude, crs="EPSG:4326"), data=ev_df
    )
    ev_df = ev_df.drop_duplicates(['geometry'])
    ev_df['table_name'] = 'doe_ev_stations'
    ev_df['type'] = 'doe_ev_stations'
    ev_df = ev_df.rename({'ID': 'ogc_fid'}, axis=1)
    ev_df = ev_df[['ogc_fid', 'geometry', 'table_name', 'type']]

    points_df = pd.concat([ev_df, osm_df])
    points_df = points_df.drop_duplicates(['geometry'])
    points_df['geometry'] = points_df['geometry'].apply(convert_to_point)
    points_df = points_df[points_df.geometry.type == 'Point']

    demand_df = pd.read_csv(args.input_ev_demand_path)
    demand_df.geohash = demand_df.geohash.astype(str)
    census_block_df = gpd.read_file(args.input_census_block_shp_path)
    census_block_df.GEOID = census_block_df.GEOID.astype(str)

    polygons_df = demand_df.merge(census_block_df[['GEOID','geometry']], how='left', left_on='geohash', right_on='GEOID')
    polygons_df = polygons_df.dropna()

    polygons_df['estimated_demand'] = polygons_df['demand'] - polygons_df['chargers']
    polygons_df['estimated_ev_chargers'] = polygons_df['estimated_demand'].apply(estimated_ev_chargers)
    polygons_df['estimated_ev_chargers'] = polygons_df['estimated_ev_chargers'].apply(lambda x: round(x))

    type_priority = ['doe_ev_stations', 'parking', 'shopping_center', 'central_government_office', 'local_and_state_government_offices', 'corporate_office', 'research_institute', 'educational_research_institute', 'hotel', 'park']

    assigned_points_gdf = assign_points_to_polygons(polygons_df, points_df, 'estimated_ev_chargers', type_priority)

    polygons_df['assigned_ev_station'] = polygons_df['assigned_ev_station'].fillna(0)
    polygons_df.to_csv(args.output_ev_demand_assigned, index=False)

    assigned_points_gdf = gpd.GeoDataFrame(assigned_points_gdf, geometry='geometry', crs=assigned_points_gdf.crs)
    assigned_points_gdf['longitude'] = assigned_points_gdf.geometry.x
    assigned_points_gdf['latitude'] = assigned_points_gdf.geometry.y
    assigned_points_gdf['id'] = assigned_points_gdf.index
    assigned_points_gdf[['id', 'longitude', 'latitude', 'capacity','geometry']].to_file(args.output_gpkg, layer='points', driver="GPKG")


if __name__ == '__main__':
    main()