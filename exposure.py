import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import os
from climada.entity.exposures import Exposures
from climada.util.constants import SYSTEM_DIR
import rasterio
from rasterio.features import shapes
from affine import Affine
import requests
import json
from scipy.interpolate import griddata

class ZurichFloodExposure(Exposures):
    """ZurichFloodExposure class extends CLIMADA Exposures for Zurich flood exposure.
    
    This class manages exposure data for flood risk assessment in Zurich,
    including building stock, infrastructure, and critical facilities.
    
    Attributes:
        gdf (GeoDataFrame): GeoPandas DataFrame with exposure data
        admin_boundaries (GeoDataFrame): Administrative boundaries of Zurich
        building_types (dict): Building type classification and parameters
        infrastructure_types (dict): Infrastructure type classification
        construction_years (dict): Construction years by district
        districts (GeoDataFrame): Districts of Zurich
        building_footprints (GeoDataFrame): Building footprints
    """
    
    def __init__(self):
        """Initialize Zurich flood exposure with default parameters."""
        super().__init__()
        self.admin_boundaries = None
        self.building_types = {
            'residential': {
                'max_floors': 8,
                'avg_value_per_sqm': 10000,  # CHF
                'avg_floor_area': 120,       # m²
                'proportion': 0.7
            },
            'commercial': {
                'max_floors': 10,
                'avg_value_per_sqm': 15000,  # CHF
                'avg_floor_area': 500,       # m²
                'proportion': 0.15
            },
            'industrial': {
                'max_floors': 3,
                'avg_value_per_sqm': 8000,   # CHF
                'avg_floor_area': 2000,      # m²
                'proportion': 0.05
            },
            'public': {
                'max_floors': 5,
                'avg_value_per_sqm': 12000,  # CHF
                'avg_floor_area': 1000,      # m²
                'proportion': 0.1
            }
        }
        self.infrastructure_types = {
            'road': {
                'primary': {'value_per_km': 10000000, 'width': 15},
                'secondary': {'value_per_km': 5000000, 'width': 10},
                'tertiary': {'value_per_km': 2000000, 'width': 8},
                'residential': {'value_per_km': 1000000, 'width': 6}
            },
            'railway': {'value_per_km': 20000000, 'width': 5},
            'bridge': {'value_per_unit': 50000000},
            'tunnel': {'value_per_km': 100000000},
            'utility': {
                'electricity': {'value_per_km': 5000000},
                'water': {'value_per_km': 3000000},
                'gas': {'value_per_km': 4000000}
            }
        }
        self.construction_years = {
            'district_1': {'mean': 1890, 'std': 30},  # Old town
            'district_2': {'mean': 1910, 'std': 20},
            'district_3': {'mean': 1930, 'std': 25},
            'district_4': {'mean': 1950, 'std': 30},
            'district_5': {'mean': 1970, 'std': 25},
            'district_6': {'mean': 1990, 'std': 20},
            'district_7': {'mean': 1960, 'std': 30},
            'district_8': {'mean': 1980, 'std': 25},
            'district_9': {'mean': 2000, 'std': 15},
            'district_10': {'mean': 2010, 'std': 10},  # Newest development
            'district_11': {'mean': 1970, 'std': 30},
            'district_12': {'mean': 1950, 'std': 35}
        }
        self.districts = None
        self.building_footprints = None
    
    def set_from_openstreetmap(self, bounds=None):
        """Load exposure data from OpenStreetMap for Zurich.
        
        Parameters:
            bounds (tuple, optional): Bounding box (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            ZurichFloodExposure: self
        """
        # Default bounds for Zurich
        if bounds is None:
            bounds = (8.4, 47.3, 8.6, 47.4)
        
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # This would use the Overpass API in a real implementation
        # For demonstration, we'll create synthetic data
        
        # Create districts (synthetic data)
        if self.districts is None:
            self.districts = self._create_synthetic_districts()
        
        # Generate building footprints
        building_geometries = []
        building_attributes = []
        
        # Number of buildings to generate (scaled for demonstration)
        num_buildings = 5000
        
        for i in range(num_buildings):
            # Random point within bounds
            lon = min_lon + (max_lon - min_lon) * np.random.random()
            lat = min_lat + (max_lat - min_lat) * np.random.random()
            point = Point(lon, lat)
            
            # Find which district the point is in
            district = None
            district_id = None
            for idx, dist in self.districts.iterrows():
                if dist.geometry.contains(point):
                    district = dist.name
                    district_id = idx + 1
                    break
            
            if district is None:
                continue  # Skip if not in any district
            
            # Building type based on proportions
            r = np.random.random()
            cum_prop = 0
            building_type = None
            for btype, params in self.building_types.items():
                cum_prop += params['proportion']
                if r <= cum_prop:
                    building_type = btype
                    break
            
            # Building size and shape
            width = np.random.normal(20, 5) if building_type != 'industrial' else np.random.normal(50, 15)
            length = np.random.normal(20, 5) if building_type != 'industrial' else np.random.normal(50, 15)
            rotation = np.random.random() * np.pi
            
            # Create building polygon
            corners = []
            for dx, dy in [(-width/2, -length/2), (width/2, -length/2), 
                           (width/2, length/2), (-width/2, length/2)]:
                rotated_x = dx * np.cos(rotation) - dy * np.sin(rotation)
                rotated_y = dx * np.sin(rotation) + dy * np.cos(rotation)
                corner_lon = lon + rotated_x * 0.00001  # Approximate conversion to degrees
                corner_lat = lat + rotated_y * 0.00001
                corners.append((corner_lon, corner_lat))
            
            building_geometry = Polygon(corners)
            
            # Number of floors
            max_floors = self.building_types[building_type]['max_floors']
            floors = np.random.randint(1, max_floors + 1)
            
            # Building value
            value_per_sqm = self.building_types[building_type]['avg_value_per_sqm']
            floor_area = self.building_types[building_type]['avg_floor_area']
            building_value = value_per_sqm * floor_area * floors
            
            # Construction year based on district
            if district_id in range(1, 13):
                district_key = f'district_{district_id}'
                mean_year = self.construction_years[district_key]['mean']
                std_year = self.construction_years[district_key]['std']
                year = int(np.random.normal(mean_year, std_year))
                year = max(1800, min(2023, year))  # Constrain to realistic range
            else:
                year = np.random.randint(1900, 2023)
            
            # Building attributes
            building_attributes.append({
                'id': i + 1,
                'type': building_type,
                'floors': floors,
                'area': floor_area * floors,
                'value': building_value,
                'year': year,
                'district': district,
                'district_id': district_id
            })
            
            building_geometries.append(building_geometry)
        
        # Create GeoDataFrame
        self.building_footprints = gpd.GeoDataFrame(
            building_attributes,
            geometry=building_geometries,
            crs="EPSG:4326"
        )
        
        # Generate roads
        road_geometries = []
        road_attributes = []
        
        # Generate some main roads
        for district_id in range(1, 13):
            # Get district center
            district = self.districts.iloc[district_id-1]
            center = district.geometry.centroid
            
            # Generate roads radiating from center
            for angle in np.linspace(0, 2*np.pi, 5):
                road_type = 'primary' if angle < np.pi/2 else 'secondary'
                
                # Road length
                length = np.random.normal(0.02, 0.005)  # In degrees
                
                # Road endpoint
                end_lon = center.x + length * np.cos(angle)
                end_lat = center.y + length * np.sin(angle)
                
                # Create road line
                road_geom = [Point(center.x, center.y), Point(end_lon, end_lat)]
                
                road_attributes.append({
                    'id': len(road_attributes) + 1,
                    'type': 'road',
                    'subtype': road_type,
                    'value': length * self.infrastructure_types['road'][road_type]['value_per_km'] * 100, # km to degree conversion
                    'district_id': district_id
                })
                
                road_geometries.append(road_geom)
        
        # Add infrastructure data to building data
        # For simplicity, we'll combine all data into a single GeoDataFrame
        all_geometries = building_geometries + road_geometries
        all_attributes = building_attributes + road_attributes
        
        # Create final GeoDataFrame
        self.gdf = gpd.GeoDataFrame(
            all_attributes,
            geometry=all_geometries,
            crs="EPSG:4326"
        )
        
        # Convert to CLIMADA format
        self.set_gdf(self.gdf)
        
        # Override value column with building or infrastructure value
        self.value = np.array([item.get('value', 0) for item in all_attributes])
        
        print(f"Loaded {len(building_geometries)} buildings and {len(road_geometries)} roads")
        return self
    
    def set_from_census(self, census_file=None):
        """Set exposure from census data.
        
        Parameters:
            census_file (str, optional): Path to census data file
            
        Returns:
            ZurichFloodExposure: self
        """
        # In a real implementation, this would load actual census data
        # For demonstration, we'll use synthetic data based on districts
        
        if self.districts is None:
            self.districts = self._create_synthetic_districts()
        
        # Generate population data by district
        district_data = []
        
        for idx, district in self.districts.iterrows():
            district_id = idx + 1
            district_name = district.name
            
            # Population density depends on district (higher in central districts)
            base_density = 15000  # persons/km²
            density_factor = 1.0
            
            if district_id <= 5:  # Central districts
                density_factor = 1.5
            elif district_id <= 10:  # Middle districts
                density_factor = 1.0
            else:  # Outer districts
                density_factor = 0.7
            
            density = base_density * density_factor
            
            # Calculate area in km²
            area = district.geometry.area * 111**2 * np.cos(np.radians(47.4))  # Approximate conversion
            
            # Calculate population
            population = int(density * area)
            
            # Calculate household statistics
            avg_household_size = np.random.normal(2.1, 0.2)
            num_households = int(population / avg_household_size)
            
            # Calculate income and wealth statistics
            if district_id <= 3:  # High-income central districts
                avg_income = np.random.normal(120000, 20000)
                avg_wealth = avg_income * np.random.uniform(8, 12)
            elif district_id <= 8:  # Middle-income districts
                avg_income = np.random.normal(90000, 15000)
                avg_wealth = avg_income * np.random.uniform(6, 10)
            else:  # Lower-income outer districts
                avg_income = np.random.normal(70000, 10000)
                avg_wealth = avg_income * np.random.uniform(4, 8)
            
            district_data.append({
                'district_id': district_id,
                'district_name': district_name,
                'population': population,
                'density': density,
                'households': num_households,
                'avg_household_size': avg_household_size,
                'avg_income': avg_income,
                'avg_wealth': avg_wealth,
                'geometry': district.geometry
            })
        
        # Create census GeoDataFrame
        census_gdf = gpd.GeoDataFrame(district_data, geometry='geometry', crs="EPSG:4326")
        
        # If we have building footprints, distribute census data to buildings
        if self.building_footprints is not None:
            # Filter residential buildings
            residential = self.building_footprints[self.building_footprints.type == 'residential']
            
            # Calculate population per building by district
            for idx, row in census_gdf.iterrows():
                district_id = row['district_id']
                district_pop = row['population']
                
                # Get buildings in this district
                district_buildings = residential[residential.district_id == district_id]
                
                if len(district_buildings) > 0:
                    # Distribute population proportionally to building area
                    total_area = district_buildings.area.sum()
                    
                    for bidx, building in district_buildings.iterrows():
                        # Calculate building population based on area proportion
                        building_pop = district_pop * (building.area / total_area)
                        
                        # Update building data
                        self.building_footprints.at[bidx, 'population'] = building_pop
                        
                        # Calculate building economic value based on occupants
                        avg_wealth = row['avg_wealth']
                        self.building_footprints.at[bidx, 'content_value'] = building_pop * avg_wealth * 0.3
        
        # Store census data
        self.census_data = census_gdf
        
        print(f"Loaded census data for {len(census_gdf)} districts with total population of {census_gdf.population.sum():.0f}")
        return self
    
    def set_elevation_data(self, dem_data):
        """Set elevation data for exposure points.
        
        Parameters:
            dem_data (ndarray): Digital Elevation Model data
            
        Returns:
            ZurichFloodExposure: self
        """
        if dem_data is None:
            raise ValueError("DEM data is required")
        
        # Get elevation for each exposure point
        if hasattr(self, 'gdf') and 'geometry' in self.gdf.columns:
            # Create coordinate grid for DEM data
            ny, nx = dem_data.shape
            lats = np.linspace(47.3, 47.4, ny)
            lons = np.linspace(8.4, 8.6, nx)
            grid_x, grid_y = np.meshgrid(lons, lats)
            
            # Get coordinates of exposure points
            point_coords = []
            for geom in self.gdf.geometry:
                if hasattr(geom, 'centroid'):
                    point_coords.append((geom.centroid.x, geom.centroid.y))
                elif isinstance(geom, list) and all(isinstance(p, Point) for p in geom):
                    # For lines (e.g., roads), use the midpoint
                    mid_idx = len(geom) // 2
                    point_coords.append((geom[mid_idx].x, geom[mid_idx].y))
            
            # Extract x and y coordinates
            points_x = np.array([p[0] for p in point_coords])
            points_y = np.array([p[1] for p in point_coords])
            
            # Interpolate elevation at exposure points
            elevations = griddata((grid_x.flatten(), grid_y.flatten()), 
                                  dem_data.flatten(), 
                                  (points_x, points_y), 
                                  method='linear')
            
            # Add elevation to GeoDataFrame
            self.gdf['elevation'] = elevations
            
            # Update CLIMADA exposure attributes
            self.elevation = elevations
            
            print(f"Added elevation data to {len(elevations)} exposure points")
        
        return self
    
    def calculate_floor_height(self):
        """Calculate first floor height based on building type and age.
        
        Returns:
            ZurichFloodExposure: self
        """
        if 'type' in self.gdf.columns and 'year' in self.gdf.columns:
            # Create floor height array
            floor_heights = np.zeros(len(self.gdf))
            
            for i, (_, row) in enumerate(self.gdf.iterrows()):
                if row.get('type') in self.building_types:
                    # Base height depends on building type
                    if row['type'] == 'residential':
                        base_height = 0.5  # 50 cm above ground
                    elif row['type'] == 'commercial':
                        base_height = 0.2  # 20 cm above ground (for accessibility)
                    elif row['type'] == 'industrial':
                        base_height = 0.3  # 30 cm above ground
                    else:  # public
                        base_height = 0.4  # 40 cm above ground
                    
                    # Adjust for construction year
                    year = row['year']
                    if year < 1950:
                        # Older buildings often have higher first floors
                        year_factor = 0.5  # Additional 50 cm
                    elif year < 1980:
                        year_factor = 0.3  # Additional 30 cm
                    elif year < 2000:
                        year_factor = 0.1  # Additional 10 cm
                    else:
                        year_factor = 0.0  # Modern buildings follow standard
                    
                    # Check if building is in flood-prone district
                    district_id = row.get('district_id')
                    district_factor = 0.0
                    
                    if district_id in [3, 4, 9]:  # Districts with historical flooding
                        district_factor = 0.2  # Additional 20 cm
                    
                    # Total floor height
                    floor_heights[i] = base_height + year_factor + district_factor
            
            # Add floor height to GeoDataFrame
            self.gdf['floor_height'] = floor_heights
            
            # Add to CLIMADA exposure attributes
            self.floor_height = floor_heights
            
            print(f"Calculated floor heights for {len(floor_heights)} buildings")
        
        return self
    
    def assign_vulnerability_classes(self):
        """Assign vulnerability classes based on building characteristics.
        
        Returns:
            ZurichFloodExposure: self
        """
        if 'type' in self.gdf.columns and 'year' in self.gdf.columns:
            # Initialize vulnerability class column
            vuln_classes = np.zeros(len(self.gdf), dtype=int)
            
            for i, (_, row) in enumerate(self.gdf.iterrows()):
                if 'type' in row and row['type'] in self.building_types:
                    building_type = row['type']
                    year = row.get('year', 2000)
                    
                    # Base vulnerability class (1-5, with 5 being most vulnerable)
                    if building_type == 'residential':
                        base_class = 3
                    elif building_type == 'commercial':
                        base_class = 4
                    elif building_type == 'industrial':
                        base_class = 5
                    else:  # public
                        base_class = 2
                    
                    # Modify based on construction year
                    if year < 1950:
                        year_modifier = 1  # Older buildings more vulnerable
                    elif year < 1980:
                        year_modifier = 0
                    elif year < 2000:
                        year_modifier = -1
                    else:
                        year_modifier = -2  # Modern buildings less vulnerable
                    
                    # Final class (constrained to 1-5 range)
                    vuln_class = max(1, min(5, base_class + year_modifier))
                    vuln_classes[i] = vuln_class
                else:
                    # For infrastructure
                    if row.get('type') == 'road':
                        if row.get('subtype') == 'primary':
                            vuln_classes[i] = 2
                        else:
                            vuln_classes[i] = 3
                    else:
                        vuln_classes[i] = 4
            
            # Add vulnerability class to GeoDataFrame
            self.gdf['vuln_class'] = vuln_classes
            
            # Add to CLIMADA exposure attributes
            self.vulnerability_class = vuln_classes
            
            print(f"Assigned vulnerability classes to {len(vuln_classes)} assets")
        
        return self
    
    def _create_synthetic_districts(self):
        """Create synthetic districts for Zurich.
        
        Returns:
            GeoDataFrame: Synthetic district boundaries
        """
        # Create concentric districts from center
        center_x, center_y = 8.54, 47.37  # Approximate center of Zurich
        
        district_polygons = []
        district_names = []
        
        # Create 12 districts (Zurich has 12 Kreise)
        for i in range(12):
            angle_start = i * np.pi/6
            angle_end = (i+1) * np.pi/6
            
            # Inner city districts are smaller
            inner_radius = 0.02 if i < 5 else 0.04
            outer_radius = 0.04 if i < 5 else 0.08
            
            # Create district polygon
            angles = np.linspace(angle_start, angle_end, 20)
            inner_points = [(center_x + inner_radius*np.cos(a), center_y + inner_radius*np.sin(a)) for a in angles]
            outer_points = [(center_x + outer_radius*np.cos(a), center_y + outer_radius*np.sin(a)) for a in reversed(angles)]
            all_points = inner_points + outer_points + [inner_points[0]]
            
            district_polygons.append(Polygon(all_points))
            district_names.append(f"District {i+1}")
        
        # Create GeoDataFrame
        districts = gpd.GeoDataFrame({
            'name': district_names,
            'geometry': district_polygons
        }, crs="EPSG:4326")
        
        return districts
    
    def set_from_raster_population(self, population_raster=None):
        """Set exposure from a population density raster.
        
        Parameters:
            population_raster (str): Path to population density raster
            
        Returns:
            ZurichFloodExposure: self
        """
        # In a real implementation, this would load actual population data
        # For demonstration, we'll create synthetic data
        
        if self.districts is None:
            self.districts = self._create_synthetic_districts()
        
        # Create a synthetic population density grid
        grid_size = 100
        lons = np.linspace(8.4, 8.6, grid_size)
        lats = np.linspace(47.3, 47.4, grid_size)
        xx, yy = np.meshgrid(lons, lats)
        
        # Initialize population grid
        population = np.zeros((grid_size, grid_size))
        
        # Create population density based on distance from center
        center_x, center_y = 8.54, 47.37
        for i in range(grid_size):
            for j in range(grid_size):
                dist = np.sqrt((xx[i, j] - center_x)**2 + (yy[i, j] - center_y)**2)
                
                # Higher density in city center, decreasing outward
                if dist < 0.02:  # Inner city
                    base_density = np.random.normal(20000, 5000)
                elif dist < 0.04:  # Middle districts
                    base_density = np.random.normal(10000, 3000)
                elif dist < 0.08:  # Outer districts
                    base_density = np.random.normal(5000, 2000)
                else:  # Outskirts
                    base_density = np.random.normal(2000, 1000)
                
                # Convert to number of people in cell
                cell_area = 0.002 * 0.001 * 111**2 * np.cos(np.radians(47.4))  # Approximate km²
                cell_population = base_density * cell_area
                
                population[i, j] = max(0, cell_population)
        
        # Convert population grid to points
        points = []
        values = []
        ids = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                if population[i, j] > 0:
                    points.append(Point(xx[i, j], yy[i, j]))
                    values.append(population[i, j])
                    ids.append(len(points))
        
        # Create GeoDataFrame
        population_gdf = gpd.GeoDataFrame({
            'id': ids,
            'population': values,
            'geometry': points
        }, crs="EPSG:4326")
        
        # Calculate exposure value based on population
        # Assuming 500,000 CHF per person (total assets)
        population_gdf['value'] = population_gdf['population'] * 500000
        
        # Assign to districts
        district_ids = []
        for _, point in population_gdf.iterrows():
            district_id = 0
            for idx, dist in self.districts.iterrows():
                if dist.geometry.contains(point.geometry):
                    district_id = idx + 1
                    break
            district_ids.append(district_id)
        
        population_gdf['district_id'] = district_ids
        
        # Set as exposure
        self.gdf = population_gdf
        self.set_gdf(population_gdf)
        
        # Override value
        self.value = np.array(values)
        
        print(f"Created population-based exposure with {len(points)} points and "
              f"total population of {sum(values):.0f}")
        
        return self
    
    def plot_exposure(self, column=None, ax=None, **kwargs):
        """Plot exposure data.
        
        Parameters:
            column (str, optional): Column to plot
            ax (matplotlib.axes, optional): Axes to plot on
            **kwargs: Additional arguments to pass to plot
            
        Returns:
            matplotlib.axes: Axes with plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        
        # Plot districts as background
        if self.districts is not None:
            self.districts.boundary.plot(ax=ax, color='black', linewidth=0.5)
        
        # Plot exposure
        if column is None:
            column = 'value'
        
        if column in self.gdf.columns:
            # Normalize values for color scaling
            norm_values = self.gdf[column] / self.gdf[column].max()
            
            # Plot each geometry with color based on value
            for idx, row in self.gdf.iterrows():
                geom = row.geometry
                value = norm_values.iloc[idx]
                
                if hasattr(geom, 'exterior'):
                    # Polygon
                    ax.fill(*geom.exterior.xy, alpha=0.6, color=plt.cm.viridis(value))
                elif isinstance(geom, Point):
                    # Point - size based on value
                    ax.scatter(geom.x, geom.y, s=value*100, color=plt.cm.viridis(value), alpha=0.6)
                elif isinstance(geom, list) and all(isinstance(p, Point) for p in geom):
                    # Line
                    x = [p.x for p in geom]
                    y = [p.y for p in geom]
                    ax.plot(x, y, color=plt.cm.viridis(value), linewidth=2+value*3, alpha=0.6)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=column)
        
        # Set title and labels
        ax.set_title(f"Zurich Exposure: {column}")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        return ax
