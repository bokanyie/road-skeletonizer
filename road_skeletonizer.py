import pandas as pd
import geopandas as gpd
import contextily as ctx

from shapely.geometry import LineString, Polygon, Point, MultiLineString
import matplotlib.pyplot as plt
import pyproj
import osmnx as ox

import pandas as pd
from shapely import wkb
import os

import networkx as nx
import pygeoops
import folium
from folium import plugins

class RoadSkeletonizer:
    def __init__(
            self, 
            polygon, 
            buffer_size = 100, 
            road_tags = {"highway": ["motorway","motorway_link","trunk","trunk_link"]}
        ):
        """
        Given a cbsa code and a buffer size, create a skeleton
        of the highways and roads within the CBSA area.

        It is possible to edit the type of roads to include from OSM.
        Defult is motorway, motorway_link, trunk, trunk_link from highway tag.

        Args:
            cbsac_code (str): The CBSA code to query.
            buffer_size (int, optional): The buffer size in meters. Defaults to 100.
            road_tags (dict, optional): The OSM road tags to include. Defaults to {"highway": ["motorway","motorway_link","trunk","trunk_link"]}.
        """
        # TODO set Euclidean projection based on polygon location or external input

        self.polygon = polygon   
        self.bbox = polygon.bounds  
        self.buffer_size = buffer_size
        self.road_tags = road_tags

        self.highways = self.get_highways()
        self.buffered_shape = self.get_buffered_shape()

        self.linestring_skeleton = self.create_polygon_skeleton()
        self.G, self.nodes, self.segments = self.get_graph_from_polygon_skeleton()
        self.error_message = None

    
    def get_roads(self):
        """
        Get the roads matching the given tags within the polygon.

        Returns:
            GeoDataFrame: A GeoDataFrame containing the highways and roads.
        """
        try:
            roads = ox.features_from_bbox(
                self.bbox,
                tags=self.road_tags
            )
        except:
            # TODO this does not work with the more general approach
            print("Error fetching roads. Trying with less tags.")
            try:
                roads = ox.features_from_bbox(
                    self.bbox,
                    tags={"highway": ["motorway","motorway_link"]}
                )
                self.error_message = "Error fetching highways with all tags. Fetched without trunk and trunk_link."
            except:
                print("Error fetching highways. Trying with only motorway.")
                try:
                    roads = ox.features_from_bbox(
                        self.bbox,
                        tags={"highway": "motorway"}
                    )
                    self.error_message = "Error fetching highways with all tags. Fetched only motorway."
                except:
                    print("Error fetching highways. Returning empty GeoDataFrame.")
                    roads = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
                    self.error_message = "Error fetching highways. Returning empty GeoDataFrame."
            
        return roads
    
    def get_buffered_shape(self):
        if len(self.highways.index) == 0:
            print("No highways found. Returning empty buffered shape.")
            buffered_shape = Polygon()
        else:
            # unioning a buffered polygon so that every lane merges
            buffered_shape = self.highways.to_crs(epsg=3857).buffer(self.buffer_size).union_all()
        return buffered_shape
    
    def create_polygon_skeleton(self):
        # check if the buffered shape is empty
        if self.buffered_shape.area == 0:
            print("Buffered shape is empty. Returning None.")
            linestring_skeleton = None
        # create a skeleton of the buffered shape
        try:
            linestring_skeleton = pygeoops.centerline(self.buffered_shape)
        except:
            print("Error creating skeleton. Returning None.")
            self.error_message = "Error creating skeleton. Returning None."
            linestring_skeleton = None
        
        # converting to a list of LineStrings
        if type(linestring_skeleton) == MultiLineString:
            linestring_skeleton = [l for l in linestring_skeleton.geoms]
        elif type(linestring_skeleton) == LineString:
            linestring_skeleton = [linestring_skeleton]
        
        return linestring_skeleton
    
    def get_graph_from_polygon_skeleton(self):
        
        if self.linestring_skeleton is None:
            print("No linestring skeleton found. Returning empty graph.")
            return None, None, None
        
        segments = gpd.GeoDataFrame(geometry=list(self.linestring_skeleton),crs=3857)
        
        # print(f"Number of segments before intersection with polygon: {len(segments.index)}")
        # project polygon to 3857
        polygon = gpd.GeoDataFrame({"geometry":[self.polygon]},crs=4326).to_crs(epsg=3857).iloc[0]["geometry"]
        segments = segments[segments.intersects(polygon)]
        # number of segments after intersection with polygon
        # print(f"Number of segments after intersection with polygon: {len(segments.index)}")
    
        segments['length'] = segments.geometry.length
        segments["start"] = segments.geometry.apply(lambda x: x.coords[0])
        segments["end"] = segments.geometry.apply(lambda x: x.coords[-1])
        
        nodes = pd.DataFrame(pd.concat([segments["start"], segments["end"]])\
                            .drop_duplicates()\
                            .reset_index(drop=True).sort_index())\
                            .rename(columns={0:"coords"})
        
        nodes["x"] = nodes["coords"].apply(lambda x: x[0])
        nodes["y"] = nodes["coords"].apply(lambda x: x[1])
        nodes["geometry"] = nodes["coords"].apply(lambda x: Point(x))
        nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs=3857)
        # print(f"Number of nodes: {len(nodes.index)}")
        # print(f"Number of segments: {len(segments.index)}")
        
        segments["source"] = segments["start"].map(lambda x: nodes[nodes["coords"]==x].index[0])
        segments["target"] = segments["end"].map(lambda x: nodes[nodes["coords"]==x].index[0])
        segments["key"] = segments.groupby(["source","target"]).cumcount()
        segments.set_index(["source","target","key"], inplace=True)

        G = ox.graph_from_gdfs(nodes,segments)

        return G, nodes, segments
    
    def plot_polygon_skeleton(self, plot_highways=True, plot_buffered_shape=True, plot_polygon=True):
        fig, ax = plt.subplots(figsize=(15, 15))
        if plot_highways:
            self.highways.to_crs(epsg=3857).plot(ax=ax, color='black', linewidth=0.5, zorder=30)
        if plot_buffered_shape:
            gpd.GeoSeries(self.buffered_shape, crs=3857).plot(ax=ax,zorder=20,alpha=0.2,color="black")

        _, nodes_G, edges_G = self.G, self.nodes, self.segments
        if plot_buffered_shape:
            nodes_G.plot(ax=ax, color='white', markersize=100, zorder=40)
        else:
            nodes_G.plot(ax=ax, color='lightgrey', markersize=100, zorder=40)
        edges_G.plot(ax=ax, color='red', linewidth=2, zorder=25)
        ctx.add_basemap(ax, crs=3857, source=ctx.providers.CartoDB.Positron)
        if plot_polygon:
            gpd.GeoDataFrame({"geometry":[self.polygon]},crs=4326).to_crs(epsg=3857).plot(ax=ax, color='none', edgecolor='blue', linewidth=2, zorder=50)
        plt.show()

        return fig, ax
    
    def plot_polygon_skeleton_folium(self, plot_highways=True, plot_buffered_shape=True, plot_polygon=True):
        # Calculate center of the polygon
        polygon_gdf = gpd.GeoDataFrame({"geometry":[self.polygon]}, crs=4326)
        center = [polygon_gdf.centroid.y.iloc[0], polygon_gdf.centroid.x.iloc[0]]
        
        # Create folium map
        m = folium.Map(location=center, zoom_start=12, tiles='CartoDB positron')
        
        if plot_highways:
            highways_4326 = self.highways.to_crs(epsg=4326)
            folium.GeoJson(
            highways_4326,
            style_function=lambda x: {'color': 'black', 'weight': 1, 'opacity': 0.7}
            ).add_to(m)
        
        if plot_buffered_shape:
            buffered_4326 = gpd.GeoSeries(self.buffered_shape, crs=3857).to_crs(epsg=4326)
            folium.GeoJson(
            buffered_4326,
            style_function=lambda x: {'fillColor': 'black', 'color': 'none', 'weight': 1, 'fillOpacity': 0.2}
            ).add_to(m)
        
        # Plot edges
        edges_4326 = self.segments.to_crs(epsg=4326)
        folium.GeoJson(
            edges_4326,
            style_function=lambda x: {'color': 'red', 'weight': 3, 'opacity': 0.8}
        ).add_to(m)
        
        # Plot nodes
        nodes_4326 = self.nodes.to_crs(epsg=4326)
        node_color = 'white' if plot_buffered_shape else 'lightgrey'
        for idx, row in nodes_4326.iterrows():
            folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color=node_color,
            fill=True,
            fillColor=node_color,
            fillOpacity=1
            ).add_to(m)
        
        if plot_polygon:
            folium.GeoJson(
            polygon_gdf,
            style_function=lambda x: {'fillColor': 'none', 'color': 'blue', 'weight': 2, 'fillOpacity': 0}
            ).add_to(m)
        
        return m
