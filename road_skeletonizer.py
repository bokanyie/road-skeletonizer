import pandas as pd
import geopandas as gpd
import contextily as ctx

from shapely.geometry import LineString, Polygon, Point, MultiLineString
from shapely.ops import linemerge, unary_union
import matplotlib.pyplot as plt
import pyproj
import osmnx as ox

import pandas as pd
from shapely import wkb
import os

import networkx as nx
import time
from centerline.geometry import Centerline
import folium
from folium import plugins

class RoadSkeletonizer:
    def __init__(
            self, 
            polygon, 
            buffer_size = 100, 
            road_tags = {"highway": ["motorway","motorway_link","trunk","trunk_link"]},
            verbose=False,
            timing: bool = False
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

        self.verbose = verbose
        self.timing = timing
        self.polygon = polygon   
        self.bbox = polygon.bounds  
        self.buffer_size = buffer_size
        self.road_tags = road_tags

        # Fetch highways first; buffered shape depends on them
        self.highways = self.get_highways()
        self.buffered_shape = self.get_buffered_shape()

        self.linestring_skeleton = self.create_polygon_skeleton()
        self.G, self.nodes, self.segments = self.get_graph_from_polygon_skeleton()
        self.simplify_graph_skeleton(min_length=200, merge_distance=100)
        self.error_message = None

    
    def get_roads(self):
        """
        Get the roads matching the given tags within the polygon.

        Returns:
            GeoDataFrame: A GeoDataFrame containing the highways and roads.
        """
        if self.verbose:
            print("Fetching highways with tags:", self.road_tags)

        try:
            t_attempt = time.perf_counter()
            roads = ox.features_from_bbox(
                self.bbox,
                tags=self.road_tags
            )
            if self.timing or self.verbose:
                print(f"OSM query (full tags) took {time.perf_counter()-t_attempt:.3f}s; features={len(roads)}")
        except:
            try:
                t_attempt = time.perf_counter()
                roads = ox.features_from_bbox(
                    self.bbox,
                    tags={"highway": ["motorway","motorway_link"]}
                )
                self.error_message = "Error fetching highways with all tags. Fetched without trunk and trunk_link."
                if self.verbose:
                    print(f"OSM query (motorway+link) took {time.perf_counter()-t_attempt:.3f}s; features={len(roads)}")
            except:
                try:
                    t_attempt = time.perf_counter()
                    roads = ox.features_from_bbox(
                        self.bbox,
                        tags={"highway": "motorway"}
                    )
                    self.error_message = "Error fetching highways with all tags. Fetched only motorway."
                    if self.verbose:
                        print(f"OSM query (motorway only) took {time.perf_counter()-t_attempt:.3f}s; features={len(roads)}")
                except:
                    roads = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
                    self.error_message = "Error fetching highways. Returning empty GeoDataFrame."
                    if self.verbose:
                        print("OSM query failed across attempts; returning empty GeoDataFrame")
            
        return roads
    
    def get_buffered_shape(self):
        """Create a merged buffer around all highway geometries.

        Returns an empty Polygon if no highways were found. Buffer performed in Web Mercator (EPSG:3857).
        """
        if self.verbose:
            print("Creating buffered shape with buffer size:", self.buffer_size)
        if len(self.highways.index) == 0:
            buffered_shape = Polygon()
        else:
            # unioning a buffered polygon so that every lane merges
            buffered_shape = self.highways.to_crs(epsg=3857).buffer(self.buffer_size).union_all()
        return buffered_shape
    
    def create_polygon_skeleton(self):
        """Generate a centerline skeleton (list of LineStrings) from the buffered highway polygon.

        Returns None if the buffered shape is empty or skeleton creation fails.
        """
        if self.verbose:
            print("Creating polygon skeleton from buffered shape")
        # check if the buffered shape is empty
        if self.buffered_shape.area == 0:
            linestring_skeleton = None
        # create a skeleton of the buffered shape
        try:
            c = Centerline(self.buffered_shape,interpolation_distance=20)
            linestring_skeleton = MultiLineString([g for g in c.geometry.geoms])

        except:
            self.error_message = "Error creating skeleton. Returning None."
            linestring_skeleton = None
        
        # converting to a list of LineStrings
        if type(linestring_skeleton) == MultiLineString:
            linestring_skeleton = [l for l in linestring_skeleton.geoms]
        elif type(linestring_skeleton) == LineString:
            linestring_skeleton = [linestring_skeleton]
        
        return linestring_skeleton
    
    def get_graph_from_polygon_skeleton(self):
        """Convert the skeleton LineStrings into an undirected NetworkX graph via osmnx.

        Performance optimizations:
        - Build a coordinate→node-id mapping dictionary to avoid per-row DataFrame lookups.
        - Use vectorized point creation via `geopandas.points_from_xy`.

        Returns:
            (Graph, GeoDataFrame, GeoDataFrame): graph, nodes gdf, segments gdf
        """
        if self.verbose:
            print("Creating graph from polygon skeleton")

        # Segments GeoDataFrame
        segments = gpd.GeoDataFrame(geometry=list(self.linestring_skeleton), crs=3857)
        segments["start"] = segments.geometry.map(lambda geom: geom.coords[0])
        segments["end"] = segments.geometry.map(lambda geom: geom.coords[-1])

        # Build unique nodes and a fast mapping from coordinate tuple → node id
        coord_series = pd.concat([segments["start"], segments["end"]], ignore_index=True)
        unique_coords = pd.Index(coord_series.unique())
        coord_to_id = {coord: i for i, coord in enumerate(unique_coords)}

        # Map segment endpoints to node ids (vectorized via Series.map on dict)
        segments["source"] = segments["start"].map(coord_to_id).astype("int32")
        segments["target"] = segments["end"].map(coord_to_id).astype("int32")
        segments["key"] = segments.groupby(["source", "target"]).cumcount()

        # Create nodes GeoDataFrame from unique coordinates
        x_vals = [c[0] for c in unique_coords]
        y_vals = [c[1] for c in unique_coords]
        nodes = gpd.GeoDataFrame(
            {
                "coords": list(unique_coords),
                "x": x_vals,
                "y": y_vals,
            },
            geometry=gpd.points_from_xy(x_vals, y_vals, crs=3857),
            crs=3857,
        )

        # Index required by osmnx's graph_from_gdfs
        segments.set_index(["source", "target", "key"], inplace=True)

        # Build undirected graph
        t_graph = time.perf_counter()
        G = ox.graph_from_gdfs(nodes, segments).to_undirected()
        if self.timing or self.verbose:
            print(f"Graph creation took {time.perf_counter()-t_graph:.3f}s; nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        return G, nodes, segments
    
    def simplify_graph_skeleton(self, min_length=300, merge_distance=200):
        """Simplify the highway skeleton graph in several stages.

        Steps:
        1. Collapse all degree-2 nodes (merge their adjacent edges into a single LineString).
        2. Iteratively remove dangling edges shorter than min_length.
        3. Merge consecutive short edges (< min_length) by building linear chains between junctions/endpoints.
        4. Merge remaining short edges (< merge_distance) depending on endpoint degree patterns, including
           creating a new midpoint node when both endpoints are high-degree (>2).

        Args:
            min_length (float): Length threshold (in projected units) for initial pruning/chain building.
            merge_distance (float): Length threshold for final merging heuristics.
        """
        if self.G is None:
            raise ValueError("No graph created yet.")
        t_total_start = time.perf_counter()
        if self.verbose:
            print("Starting graph simplification with", self.G.number_of_nodes(), "nodes and", self.G.number_of_edges(), "edges.")
            print("Remove all degree-2 nodes")
        t1 = time.perf_counter()
        deg2_nodes = [n for n in self.G.nodes() if self.G.degree(n) == 2]        
        subG = self.G.subgraph(deg2_nodes).copy()
        # get connected components
        conn_comp = nx.connected_components(subG)
        chains = []
        merged_components = 0
        for comp in conn_comp:
            comp = list(comp)
            if len(comp) < 3:
                continue  # skip short components
            if self.verbose:
                print("  Processing component:", comp)
            # find endpoints in the full graph
            endpoints = [n for n in comp if subG.degree(n) != 2]
            if self.verbose:
                print("    Endpoints in full graph:", endpoints)
                print("Degrees:", [subG.degree(n) for n in endpoints])
            u = [k for k in self.G.neighbors(endpoints[0]) if k not in comp][0]
            v = [k for k in self.G.neighbors(endpoints[-1]) if k not in comp][0]
            # get geometries along the full chain
            chain = [u] + nx.shortest_path(subG, source=endpoints[0], target=endpoints[-1]) + [v]
            chains.append(chain)
            geoms = []
            for prev,next in zip(chain[:-1], chain[1:]):
                geom = self.G.get_edge_data(prev, next)[0]['geometry']
                geoms.append(geom)
            # Robust merge of a chain into a single LineString
            merged_geom = linemerge(unary_union(geoms))
            if merged_geom.geom_type == 'MultiLineString':
                merged_geom = LineString([pt for line in merged_geom.geoms for pt in line.coords])
            # remove intermediate nodes and edges
            for n in comp:
                self.G.remove_node(n)
            # add new edge
            self.G.add_edge(u, v, geometry=merged_geom)
            merged_components += 1
        if self.timing or self.verbose:
            print(f"Collapsed degree-2 components: {merged_components} in {time.perf_counter()-t1:.3f}s")

        if self.verbose:
            print("Remove dangling edges shorter than min_length")
        t2 = time.perf_counter()
        removed_dangling = 0
        dangling = True
        while dangling:
            dangling = False
            deg1_nodes = [n for n in self.G.nodes() if self.G.degree(n) == 1]
            for n in deg1_nodes:
                nbr = list(self.G.neighbors(n))
                if len(nbr) == 0:
                    self.G.remove_node(n)
                    dangling = True
                    removed_dangling += 1
                    continue
                else:
                    u = nbr[0]
                    edge_data = self.G.get_edge_data(n, u)[0]
                    length = edge_data['geometry'].length
                    if length < min_length:
                        self.G.remove_node(n)
                        dangling = True
                        removed_dangling += 1
        if self.timing or self.verbose:
            print(f"Removed dangling short edges: {removed_dangling} in {time.perf_counter()-t2:.3f}s")

        if self.verbose:
            print("Merge consecutive edges shorter than min_length by forming chains")
        t3 = time.perf_counter()

        # get subgraph of edges smaller than min_length
        small_edges = [(u, v, 0) for u, v, k in self.G.edges(data=True) if k['geometry'].length < min_length]
        subG = self.G.edge_subgraph(small_edges).copy()

        # create chains in this subgraph
        chains = []
        visited = set()
        while True:
            # find a starting node with degree != 2 in subG (endpoints or junctions)
            start_node = None
            for n in subG.nodes():
                if n in visited:
                    continue
                deg = subG.degree(n)
                if deg != 2:  # Only start from endpoints (deg=1) or junctions (deg>2)
                    start_node = n
                    break
            # all valid starting nodes visited
            if start_node is None:
                break
            
            visited.add(start_node)
            
            # explore chain from start_node in each direction
            for neighbor in list(subG.neighbors(start_node)):
                if neighbor in visited:
                    continue
                if self.G.degree(neighbor) > 2:
                    continue  # Skip high-degree neighbors in the full graph
                
                chain = [start_node, neighbor]
                visited.add(neighbor)
                current_node = neighbor
                prev_node = start_node
                
                max_chain_length = len(subG.nodes())
                while len(chain) < max_chain_length:
                    # potential next directions in the subgraph
                    next_nodes = [nbr for nbr in subG.neighbors(current_node) if nbr != prev_node and nbr not in visited]
                    if not next_nodes:
                        break
                    next_node = next_nodes[0]
                    chain.append(next_node)
                    visited.add(next_node)
                    # stop at intersection in the full graph (not just subgraph)
                    if self.G.degree(next_node) > 2:
                        break
                    prev_node = current_node
                    current_node = next_node
                
                if len(chain) > 2:
                    chains.append(chain)
        if self.timing or self.verbose:
            print(f"Formed {len(chains)} chains in {time.perf_counter()-t3:.3f}s")

        # process chains that have been found
        t3b = time.perf_counter()
        removed_in_chains = 0
        for chain in chains:
            u = chain[0]
            v = chain[-1]
            geoms = []
            for i in range(len(chain) - 1):
                edge_data = self.G.get_edge_data(chain[i], chain[i + 1])[0]
                geoms.append(edge_data['geometry'])

            # Robust merge of a chain into a single LineString
            merged_geom = linemerge(unary_union(geoms))
            if merged_geom.geom_type == 'MultiLineString':
                merged_geom = LineString([pt for line in merged_geom.geoms for pt in line.coords])
            # remove intermediate nodes and edges
            for i in range(1, len(chain) - 1):
                self.G.remove_node(chain[i])
                removed_in_chains += 1
            # add new edge
            self.G.add_edge(u, v, geometry=merged_geom)
        if self.timing or self.verbose:
            print(f"Merged chains and removed {removed_in_chains} intermediate nodes in {time.perf_counter()-t3b:.3f}s")

        # list edges shorter than merge_distance
        if self.verbose:
            print("Process remaining short edges (< merge_distance)")
        t4 = time.perf_counter()
        ops_short_edges = 0
        short_edges = [(u, v) for u, v, k in self.G.edges(data=True) if k['geometry'].length < merge_distance]
        for u, v in short_edges:
            # Skip if nodes were already removed by previous operations
            if u not in self.G.nodes() or v not in self.G.nodes():
                continue
            if not self.G.has_edge(u, v):
                continue
            
            if self.verbose:
                print("  Edge", u, v, "Degrees", self.G.degree(u), self.G.degree(v))
            deg_u = self.G.degree(u)
            deg_v = self.G.degree(v)
            geom_edge = self.G.get_edge_data(u, v, 0)['geometry']
            
            # if both degrees are 2, search for shorter attaching edge and merge
            if deg_u == 2 and deg_v == 2:
                next_node_u = [nbr for nbr in self.G.neighbors(u) if nbr != v][0]
                next_node_v = [nbr for nbr in self.G.neighbors(v) if nbr != u][0]
                geom_u = self.G.get_edge_data(u, next_node_u)[0]['geometry']
                geom_v = self.G.get_edge_data(v, next_node_v)[0]['geometry']
                if geom_u.length < geom_v.length:
                    segment_new = linemerge(unary_union([geom_edge, geom_u]))
                    if segment_new.geom_type == 'MultiLineString':
                        segment_new = LineString([pt for line in segment_new.geoms for pt in line.coords])
                    self.G.remove_node(u)
                    self.G.add_edge(next_node_u, v, geometry=segment_new, color="magenta")
                    ops_short_edges += 1
                else:
                    segment_new = linemerge(unary_union([geom_edge, geom_v]))
                    if segment_new.geom_type == 'MultiLineString':
                        segment_new = LineString([pt for line in segment_new.geoms for pt in line.coords])
                    self.G.remove_node(v)
                    self.G.add_edge(next_node_v, u, geometry=segment_new, color="magenta")
                    ops_short_edges += 1
            # if one degree is 2, merge with edge attaching from that direction
            elif deg_u == 2 and deg_v != 2:
                next_node_u = [nbr for nbr in self.G.neighbors(u) if nbr != v][0]
                geom_u = self.G.get_edge_data(u, next_node_u)[0]['geometry']
                segment_new = linemerge(unary_union([geom_edge, geom_u]))
                if segment_new.geom_type == 'MultiLineString':
                    segment_new = LineString([pt for line in segment_new.geoms for pt in line.coords])
                self.G.remove_node(u)
                self.G.add_edge(next_node_u, v, geometry=segment_new, color="magenta")
                ops_short_edges += 1
            elif deg_u != 2 and deg_v == 2:
                next_node_v = [nbr for nbr in self.G.neighbors(v) if nbr != u][0]
                geom_v = self.G.get_edge_data(v, next_node_v)[0]['geometry']
                segment_new = linemerge(unary_union([geom_edge, geom_v]))
                if segment_new.geom_type == 'MultiLineString':
                    segment_new = LineString([pt for line in segment_new.geoms for pt in line.coords])
                self.G.remove_node(v)
                self.G.add_edge(next_node_v, u, geometry=segment_new, color="magenta")
                ops_short_edges += 1
            # if both degrees are >2, merge nodes into geometric center and continue all attaching edges to that node!
            elif deg_u > 2 and deg_v > 2:
                pos_u_x = self.G.nodes()[u]['x']
                pos_u_y = self.G.nodes()[u]['y']
                pos_v_x = self.G.nodes()[v]['x']
                pos_v_y = self.G.nodes()[v]['y']
                center_x = (pos_u_x + pos_v_x) / 2
                center_y = (pos_u_y + pos_v_y) / 2
                new_node_id = f"merged_{u}_{v}"
                self.G.add_node(new_node_id, x=center_x, y=center_y, geometry=Point(center_x, center_y))
                
                # Redirect all edges from u (except u-v)
                for nbr in list(self.G.neighbors(u)):
                    if nbr != v:
                        geom_old = self.G.get_edge_data(u, nbr)[0]['geometry']
                        # Adjust geometry to connect to new center
                        coords_old = list(geom_old.coords)
                        if coords_old[0] == (pos_u_x, pos_u_y):
                            new_coords = [(center_x, center_y)] + coords_old[1:]
                        elif coords_old[-1] == (pos_u_x, pos_u_y):
                            new_coords = coords_old[:-1] + [(center_x, center_y)]
                        else:
                            # Fallback: add center as new endpoint
                            new_coords = [(center_x, center_y)] + coords_old
                        new_geom = LineString(new_coords)
                        self.G.add_edge(new_node_id, nbr, geometry=new_geom, color="magenta")
                
                # Redirect all edges from v (except u-v)
                for nbr in list(self.G.neighbors(v)):
                    if nbr != u:
                        geom_old = self.G.get_edge_data(v, nbr)[0]['geometry']
                        coords_old = list(geom_old.coords)
                        if coords_old[0] == (pos_v_x, pos_v_y):
                            new_coords = [(center_x, center_y)] + coords_old[1:]
                        elif coords_old[-1] == (pos_v_x, pos_v_y):
                            new_coords = coords_old[:-1] + [(center_x, center_y)]
                        else:
                            new_coords = [(center_x, center_y)] + coords_old
                        new_geom = LineString(new_coords)
                        self.G.add_edge(new_node_id, nbr, geometry=new_geom, color="magenta")
                
                self.G.remove_node(u)
                self.G.remove_node(v)
                ops_short_edges += 1
        if self.timing or self.verbose:
            print(f"Processed remaining short edges: {ops_short_edges} operations in {time.perf_counter()-t4:.3f}s")
        if self.timing or self.verbose:
            print(f"Total simplify time: {time.perf_counter()-t_total_start:.3f}s")

    def get_simplified_skeleton(self):
        # get G, nodes, edges similarly to get_graph_from_polygon_skeleton
        # create nodes from G.nodes()
        if self.G is None:
            raise ValueError("No graph created yet.")
        # Create nodes GeoDataFrame from G.nodes()
        node_data = list(self.G.nodes(data=True))
        nodes = gpd.GeoDataFrame(
            {
            "node_id": [node_id for node_id, _ in node_data],
            "x": [data['x'] for _, data in node_data],
            "y": [data['y'] for _, data in node_data],
            "coords": [(data['x'], data['y']) for _, data in node_data],  # Include coords
            },
            geometry=[Point(data['x'], data['y']) for _, data in node_data],
            crs=3857,
        )
        # Create edges GeoDataFrame from G.edges()
        edge_data = list(self.G.edges(data=True, keys=True))
        edges = gpd.GeoDataFrame(
            {
            "source": [u for u, v, k, _ in edge_data],
            "target": [v for u, v, k, _ in edge_data],
            "key": [k for u, v, k, _ in edge_data],
            "start": [data['geometry'].coords[0] for _, _, _, data in edge_data],
            "end": [data['geometry'].coords[-1] for _, _, _, data in edge_data],
            },
            geometry=[data['geometry'] for _, _, _, data in edge_data],
            crs=3857,
        )
        edges.set_index(["source", "target", "key"], inplace=True)
        return self.G, nodes, edges 
    
    def plot_polygon_skeleton(self, plot_highways=True, plot_buffered_shape=True, plot_polygon=True):
        """Plot the polygon, buffered shape, highways, and simplified graph using matplotlib."""
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
    
    def plot_polygon_skeleton_folium(self, 
            plot_highways=True, 
            plot_buffered_shape=True, 
            plot_linestring = True, 
            plot_polygon=True
        ):
        """Interactive folium map of the polygon, highways, buffered shape, skeleton lines, and simplified graph."""
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

        if plot_linestring:
            linestring_4326 = gpd.GeoDataFrame(geometry=self.linestring_skeleton, crs=3857).to_crs(epsg=4326)
            folium.GeoJson(
            linestring_4326,
            style_function=lambda x: {'color': 'orange', 'weight': 10, 'opacity': 0.7}
            ).add_to(m)
        
        if plot_buffered_shape:
            buffered_4326 = gpd.GeoSeries(self.buffered_shape, crs=3857).to_crs(epsg=4326)
            folium.GeoJson(
            buffered_4326,
            style_function=lambda x: {'fillColor': 'black', 'color': 'none', 'weight': 1, 'fillOpacity': 0.2}
            ).add_to(m)

        # # add original nodes in dark gray with radius 3 (filled)
        # folium.GeoJson(
        #     self.nodes.to_crs(epsg=4326).to_json(),
        #     marker=folium.CircleMarker(radius=3, color='red', fill=True, fill_color='orange',fillOpacity=0.7)
        # ).add_to(m)

        # add edges in red width 2
        folium.GeoJson(
            ox.graph_to_gdfs(self.G, nodes=False, edges=True).to_crs(epsg=4326).to_json(),
            style_function=lambda x: {'color': 'red', 'weight': 5}).add_to(m)
        # add nodes in white with radius 5 (filled)
        folium.GeoJson(
            ox.graph_to_gdfs(self.G, nodes=True, edges=False).to_crs(epsg=4326).to_json(),
            marker=folium.CircleMarker(radius=5, color='white', fill=True, fill_color='white',fillOpacity=1)
        ).add_to(m)
        
        if plot_polygon:
            folium.GeoJson(
                polygon_gdf,
                style_function=lambda x: {'fillColor': 'none', 'color': 'blue', 'weight': 2, 'fillOpacity': 0}
            ).add_to(m)
        
        return m
