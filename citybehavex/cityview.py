from __future__ import annotations

import duckdb
import geopandas as gpd
import pandas as pd
import shapely
import typer
from shapely import wkb
from shapely.geometry import MultiPolygon, Polygon

# Layers pulled from Overture Maps. Each entry is a (theme/type path, extra WHERE clause).
_BUILDINGS_PATH = "theme=buildings/type=*/*"
_ROADS_PATH = "theme=transportation/type=segment/*"
_GREEN_PATH = "theme=base/type=land_use/*"
_GREEN_SUBTYPES = ("park", "garden", "forest", "grass", "recreation_ground")

# Buffer width (EPSG:3857 metres) used to turn road centre-lines into thin ribbons before
# triangulation. ~6 projected units ≈ 4 real metres at Paris latitude.
_ROAD_BUFFER = 6.0


def _connect() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("SET s3_region='us-west-2';")
    return conn


def _read_layer(
    conn: duckdb.DuckDBPyConnection,
    overture_release: str,
    path: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    where: str = "",
) -> gpd.GeoDataFrame:
    query = f"""
        SELECT id, geometry
        FROM read_parquet(
            's3://overturemaps-us-west-2/release/{overture_release}/{path}',
            hive_partitioning=1
        )
        WHERE bbox.xmin > {min_lon}
          AND bbox.xmax < {max_lon}
          AND bbox.ymin > {min_lat}
          AND bbox.ymax < {max_lat}
          {where}
    """
    df = conn.execute(query).df()
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326").to_crs(epsg=3857)
    df["geometry"] = df["geometry"].apply(lambda x: wkb.loads(bytes(x)))
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf.to_crs(epsg=3857)


def load_overture_layers(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    overture_release: str,
) -> dict[str, gpd.GeoDataFrame]:
    """Load buildings, roads, and green-space geometries from Overture Maps (EPSG:3857)."""
    conn = _connect()
    try:
        buildings = _read_layer(
            conn, overture_release, _BUILDINGS_PATH, min_lon, min_lat, max_lon, max_lat
        )
        roads = _read_layer(
            conn,
            overture_release,
            _ROADS_PATH,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            where="AND subtype = 'road'",
        )
        subtypes = ", ".join(f"'{s}'" for s in _GREEN_SUBTYPES)
        green = _read_layer(
            conn,
            overture_release,
            _GREEN_PATH,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            where=f"AND subtype IN ({subtypes})",
        )
    finally:
        conn.close()
    return {"building": buildings, "road": roads, "green": green}


def triangulate_geometry(geom) -> MultiPolygon | None:
    """Triangulate a (Multi)Polygon into a MultiPolygon whose parts are triangles.

    Uses constrained Delaunay triangulation, which honours the polygon boundary and holes.
    Returns ``None`` for empty/invalid input.
    """
    if geom is None or geom.is_empty:
        return None
    polygons: list[Polygon]
    if geom.geom_type == "Polygon":
        polygons = [geom]
    elif geom.geom_type == "MultiPolygon":
        polygons = list(geom.geoms)
    else:
        return None

    triangles: list[Polygon] = []
    for poly in polygons:
        if poly.is_empty:
            continue
        result = shapely.constrained_delaunay_triangles(poly)
        triangles.extend(t for t in result.geoms if not t.is_empty)
    if not triangles:
        return None
    return MultiPolygon(triangles)


def build_cityview_file(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    overture_release: str,
    output: str,
) -> gpd.GeoDataFrame:
    """Build a single FlatGeobuf of pre-triangulated building/road/green geometries."""
    layers = load_overture_layers(min_lon, min_lat, max_lon, max_lat, overture_release)

    # Roads are line geometries: buffer them into thin ribbons so they can be triangulated
    # and rendered as filled meshes like the polygon layers.
    roads = layers["road"]
    if not roads.empty:
        roads = roads.copy()
        roads["geometry"] = roads.geometry.buffer(_ROAD_BUFFER)
        layers["road"] = roads

    parts: list[gpd.GeoDataFrame] = []
    for kind, gdf in layers.items():
        if gdf.empty:
            typer.echo(f"  {kind}: 0 features")
            continue
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.apply(triangulate_geometry)
        gdf["kind"] = kind
        gdf = gdf[gdf.geometry.notna()].reset_index(drop=True)
        typer.echo(f"  {kind}: {len(gdf):,} features")
        parts.append(gdf[["id", "kind", "geometry"]])

    if not parts:
        raise ValueError("No geometries found in the requested bounding box.")

    combined = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:3857"
    )
    combined.to_file(output, driver="FlatGeobuf")
    typer.echo(f"Saved {len(combined):,} triangulated shapes -> {output}")
    return combined
