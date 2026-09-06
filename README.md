# Road Skeletonizer

Extract and simplify a road-network skeleton from OpenStreetMap data inside a
Shapely polygon. The `RoadSkeletonizer` class uses a scikit-learn-style API:
configure an instance, call `fit()`, then inspect the resulting graph or
GeoDataFrames.

## Requirements

The project currently has no package metadata, so install the runtime
dependencies directly in the active Python environment:

```bash
pip install centerline contextily folium geopandas matplotlib networkx osmnx pandas pyproj shapely
```

An internet connection is required while `fit()` downloads road features from
OpenStreetMap through the Overpass API. The input polygon should use longitude
and latitude coordinates (`EPSG:4326`). Distances such as `buffer_size` and
the simplification thresholds are interpreted in metres after reprojection to
Web Mercator (`EPSG:3857`).

## Quick start

```python
from shapely.geometry import box

from road_skeletonizer import RoadSkeletonizer

# Example area around central Budapest: (min longitude, min latitude,
# max longitude, max latitude).
area = box(19.03, 47.48, 19.08, 47.52)

skeletonizer = RoadSkeletonizer(
		buffer_size=100,
		road_tags={"highway": ["motorway", "motorway_link", "trunk", "trunk_link"]},
		simplify_min_length=300,
		simplify_merge_distance=250,
		verbose=True,
)
skeletonizer.fit(area)

# Simplified results.
graph, nodes, edges = skeletonizer.get_simplified_skeleton()
print(graph.number_of_nodes(), graph.number_of_edges())

# Interactive visualization.
map_view = skeletonizer.plot_polygon_skeleton_folium()
map_view.save("road_skeleton_map.html")
```

The example performs a live OSM query. Keep the requested area reasonably
small and respect OpenStreetMap and Overpass usage policies.

## Main parameters

| Parameter | Description | Default |
| --- | --- | ---: |
| `buffer_size` | Buffer around downloaded road geometries, in metres | `100` |
| `road_tags` | OSM feature tags passed to OSMnx | motorway, motorway_link, trunk, trunk_link |
| `simplify_min_length` | Threshold for pruning and short-chain merging, in metres | `300` |
| `simplify_merge_distance` | Distance threshold for nearby-node merging, in metres | `250` |
| `verbose` | Print pipeline progress | `False` |
| `timing` | Print timing information | `False` |

## Results and helper methods

After `fit(polygon)`, the instance exposes the intermediate pipeline state:

- `roads`: downloaded road features as a GeoDataFrame
- `buffered_shape`: merged buffered geometry in `EPSG:3857`
- `linestring_skeleton`: generated centerline geometries
- `G`: simplified undirected NetworkX graph
- `nodes`, `segments`: GeoDataFrames matching the final simplified graph
- `error_message`: message recorded when a fallback OSM or skeleton step is used

Use `get_simplified_skeleton()` for GeoDataFrames reconstructed from the final
graph. The plotting helpers are `plot_polygon_skeleton()` for Matplotlib and
`plot_polygon_skeleton_folium()` for an interactive Folium map.

## Notes

- OSM queries can fail because of network errors, Overpass limits, or invalid
	tags. The class retries with narrower motorway filters and records a message
	in `error_message`.
- Web Mercator is convenient for this workflow but is not an accurate local
	metric CRS for precise surveying over large areas.
- There is currently no automated test suite or published package definition;
	`test.ipynb` contains the worked Budapest example.
