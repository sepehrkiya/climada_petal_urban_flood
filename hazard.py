import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from numba import jit, prange
import matplotlib.pyplot as plt
from climada.hazard import Hazard
from climada.util.constants import SYSTEM_DIR
import os
import requests
import multiprocessing as mp
from pathlib import Path
import datetime
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

class ZurichFloodHazard(Hazard):
    """ZurichFloodHazard class extends CLIMADA Hazard for urban flood modeling in Zurich.
    
    This hazard module implements a sophisticated 2D shallow water model with
    Green-Ampt infiltration for the city of Zurich, Switzerland. It handles
    the 10 municipal districts and accounts for the lake and river systems.
    
    Attributes:
        intensity (sparse.csr_matrix): Flood depth in meters
        fraction (sparse.csr_matrix): Fractional impact (1 for flood)
        event_id (np.array): Unique identifier for each event
        frequency (np.array): Annual frequency of each event
        date (np.array): Date of each event
        orig (np.array): Origin of data source
        units (str): Units of intensity (default: 'm')
        centroids (Centroids): Centroids container
        event_name (list): Name of each event
        tag (TagHazard): Tag information about the source
        pool (BoundedThreadPoolExecutor): Thread pool for parallel computation
        rainfall_data (xarray.Dataset): Precipitation data
        dem_data (np.array): Digital Elevation Model of Zurich
        districts (gpd.GeoDataFrame): Boundaries of Zurich districts
        drainage_network (gpd.GeoDataFrame): Storm water drainage network
        soil_properties (pd.DataFrame): Soil infiltration parameters
        lake_zurich (Polygon): Lake Zurich boundary polygon
        rivers (gpd.GeoDataFrame): Rivers in Zurich area
    """
    
    def __init__(self):
        """Initialize Zurich flood hazard model with default parameters."""
        super().__init__(haz_type='FL')
        self.units = 'm'
        self.pool = None
        self.rainfall_data = None
        self.dem_data = None
        self.districts = None
        self.drainage_network = None
        self.soil_properties = None
        self.lake_zurich = None
        self.rivers = None
        self.model_params = {
            'dx': 10.0,  # Grid resolution in meters
            'dt': 1.0,   # Time step in seconds
            'g': 9.81,   # Gravity acceleration
            'manning_n': 0.03,  # Manning roughness coefficient
            'theta': 0.7,  # Time integration parameter (0.5-1.0)
            'max_iter': 10000,  # Maximum iterations
            'convergence_threshold': 1e-5,  # Convergence criterion
            'cfl_number': 0.7,  # Courant–Friedrichs–Lewy condition
            'output_interval': 3600,  # Output interval in seconds
            'sim_duration': 86400,  # Simulation duration in seconds
            'infiltration': {
                'initial_moisture': 0.3,  # Initial moisture content
                'saturated_moisture': 0.5,  # Saturated moisture content
                'suction_head': 0.1667,  # Suction head in meters
                'hydraulic_conductivity': {
                    'urban': 5e-7,      # Urban areas (m/s)
                    'suburban': 1e-6,   # Suburban areas (m/s)
                    'parks': 5e-6,      # Parks and gardens (m/s)
                    'forest': 1e-5      # Forest areas (m/s)
                }
            },
            'drainage': {
                'capacity': 0.02,  # Base drainage capacity in m/h
                'upgrade_factor': {  # Capacity multiplier by district
                    'district_1': 1.2,  # Central district
                    'district_2': 1.0,
                    'district_3': 0.9,
                    'district_4': 1.1,
                    'district_5': 1.0,
                    'district_6': 0.8,
                    'district_7': 1.1,
                    'district_8': 1.0,
                    'district_9': 0.9,
                    'district_10': 0.7, # Newest development
                    'district_11': 1.0,
                    'district_12': 0.8
                }
            }
        }
        
    def set_from_copernicus_cds(self, start_date, end_date, api_key=None):
        """Load precipitation data from Copernicus Climate Data Store.
        
        Parameters:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            api_key (str, optional): Copernicus CDS API key
        
        Returns:
            ZurichFloodHazard: self
        """
        # Check if API key is provided or exists in environment
        if api_key is None:
            api_key = os.environ.get('COPERNICUS_CDS_API_KEY')
            if api_key is None:
                raise ValueError("No Copernicus CDS API key provided")
        
        # Format request to Copernicus CDS
        url = "https://cds.climate.copernicus.eu/api/v2"
        request_params = {
            'dataset_short_name': 'reanalysis-era5-land',
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['total_precipitation'],
            'year': [d.split('-')[0] for d in [start_date, end_date]],
            'month': list(range(1, 13)),
            'day': list(range(1, 32)),
            'time': [f"{h:02d}:00" for h in range(24)],
            'area': [47.4, 8.4, 47.3, 8.6],  # Zurich bounding box
        }
        
        # Log request details
        print(f"Requesting precipitation data from Copernicus CDS for {start_date} to {end_date}")
        
        # This is a simplified placeholder - actual implementation would use the CDS API client
        # For demonstration only - in production code, use proper CDS API client
        # self.rainfall_data = xr.open_dataset(response_file)
        
        # Placeholder simulated data (in actual implementation, this would come from CDS)
        time_range = pd.date_range(start=start_date, end=end_date, freq='1H')
        lats = np.linspace(47.3, 47.4, 20)
        lons = np.linspace(8.4, 8.6, 20)
        
        # Create synthetic precipitation data with realistic spatio-temporal pattern
        # Rainstorm events with higher intensity in summer months
        precip_data = np.zeros((len(time_range), len(lats), len(lons)))
        
        for i, t in enumerate(time_range):
            month = t.month
            hour = t.hour
            
            # More intense rain in summer (May-September)
            seasonal_factor = 2.0 if 5 <= month <= 9 else 0.8
            
            # Diurnal cycle - more rain in afternoon/evening
            diurnal_factor = 1.5 if 14 <= hour <= 20 else 0.9
            
            # Random storm events (approximately 10% of time steps)
            if np.random.random() < 0.1:
                storm_center_lat = np.random.choice(range(len(lats)))
                storm_center_lon = np.random.choice(range(len(lons)))
                
                # Create storm pattern
                for lat_idx in range(len(lats)):
                    for lon_idx in range(len(lons)):
                        distance = np.sqrt((lat_idx - storm_center_lat)**2 + 
                                          (lon_idx - storm_center_lon)**2)
                        intensity = 10 * np.exp(-0.5 * (distance/3)**2) * seasonal_factor * diurnal_factor
                        precip_data[i, lat_idx, lon_idx] = max(0, intensity + np.random.normal(0, 0.2))
            else:
                # Background precipitation
                precip_data[i, :, :] = np.random.exponential(0.5) * seasonal_factor * diurnal_factor
        
        # Create xarray dataset
        self.rainfall_data = xr.Dataset(
            data_vars={
                "tp": (["time", "latitude", "longitude"], precip_data)
            },
            coords={
                "time": time_range,
                "latitude": lats,
                "longitude": lons
            }
        )
        
        # Add metadata
        self.rainfall_data.tp.attrs["units"] = "mm/h"
        self.rainfall_data.tp.attrs["long_name"] = "Total Precipitation"
        
        print(f"Successfully loaded precipitation data with {len(time_range)} timestamps")
        return self
    
    def load_zurich_data(self, data_dir=None):
        """Load Zurich specific data: DEM, districts, drainage network, etc.
        
        Parameters:
            data_dir (str, optional): Directory with Zurich data files
            
        Returns:
            ZurichFloodHazard: self
        """
        if data_dir is None:
            data_dir = os.path.join(SYSTEM_DIR, 'data', 'zurich')
        
        # In a real implementation, this would load actual data files
        # For demonstration, we'll create synthetic data
        
        # Create synthetic DEM with realistic Zurich topography
        # Zurich has elevation ranging from ~400m to ~850m
        grid_size = 500
        x = np.linspace(0, 10000, grid_size)  # 10 km grid
        y = np.linspace(0, 10000, grid_size)  # 10 km grid
        
        # Base elevation
        base_elevation = 400 + np.zeros((grid_size, grid_size))
        
        # Lake Zurich (south-east)
        lake_mask = np.zeros((grid_size, grid_size), dtype=bool)
        for i in range(grid_size):
            for j in range(grid_size):
                if (i-grid_size*0.7)**2 + (j-grid_size*0.7)**2 < (grid_size*0.3)**2:
                    lake_mask[i, j] = True
        
        base_elevation[lake_mask] = 399  # Lake level
        
        # Hills (Uetliberg in west, Zürichberg in east)
        xx, yy = np.meshgrid(x, y)
        
        # Uetliberg (west side)
        uetliberg = 400 * np.exp(-0.5 * ((xx-2000)**2 + (yy-5000)**2) / 2000**2)
        
        # Zürichberg (east side)
        zurichberg = 300 * np.exp(-0.5 * ((xx-8000)**2 + (yy-5000)**2) / 2000**2)
        
        # Combining topographic features
        self.dem_data = base_elevation + uetliberg + zurichberg
        
        # Synthetic river path (Limmat river running from lake through city center)
        river_y = np.linspace(grid_size*0.7, 0, 100)
        river_x = grid_size*0.7 + np.sin(np.linspace(0, np.pi*2, 100)) * grid_size*0.05
        
        # Carve river into DEM
        for i, (x, y) in enumerate(zip(river_x, river_y)):
            x_idx = int(x)
            y_idx = int(y)
            if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                # River channel with declining elevation from lake to north
                river_elevation = 398 - i*0.1
                # Create a river bed approx 30m wide
                for dx in range(-15, 16):
                    for dy in range(-15, 16):
                        if x_idx+dx >= 0 and x_idx+dx < grid_size and y_idx+dy >= 0 and y_idx+dy < grid_size:
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist < 15:
                                self.dem_data[y_idx+dy, x_idx+dx] = min(
                                    self.dem_data[y_idx+dy, x_idx+dx],
                                    river_elevation + dist*0.2
                                )
        
        # Create synthetic district boundaries
        # In a real implementation, this would load from a GeoJSON file
        district_polygons = []
        district_names = []
        
        # Create 12 districts (Zurich has 12 Kreise)
        for i in range(12):
            angle_start = i * np.pi/6
            angle_end = (i+1) * np.pi/6
            
            # Inner city districts are smaller
            inner_radius = 2000 if i < 5 else 4000
            outer_radius = 4000 if i < 5 else 8000
            
            # Create district polygon
            angles = np.linspace(angle_start, angle_end, 20)
            inner_points = [(5000 + inner_radius*np.cos(a), 5000 + inner_radius*np.sin(a)) for a in angles]
            outer_points = [(5000 + outer_radius*np.cos(a), 5000 + outer_radius*np.sin(a)) for a in reversed(angles)]
            all_points = inner_points + outer_points + [inner_points[0]]
            
            district_polygons.append(Polygon(all_points))
            district_names.append(f"district_{i+1}")
        
        # Create GeoDataFrame
        self.districts = gpd.GeoDataFrame({
            'name': district_names,
            'geometry': district_polygons,
            'population': [1000*i + 30000 for i in range(12)],  # Synthetic population
            'build_year': [1850 + 10*i for i in range(12)]  # Synthetic construction years
        })
        
        # Set lake boundary
        lake_boundary = []
        for theta in np.linspace(0, 2*np.pi, 100):
            x = 5000 + 3000 * np.cos(theta) * 1.5  # Elongated in x direction
            y = 5000 + 3000 * np.sin(theta) * 0.8  # Compressed in y direction
            lake_boundary.append((x, y))
        
        self.lake_zurich = Polygon(lake_boundary)
        
        print("Successfully loaded Zurich geographic data")
        return self
    
    @jit(nopython=True, parallel=True)
    def _shallow_water_step(self, h, u, v, z, dx, dt, g, n, rainfall):
        """Compute one time step of the 2D shallow water equations.
        
        Parameters:
            h (ndarray): Water depth
            u (ndarray): x-direction velocity
            v (ndarray): y-direction velocity
            z (ndarray): Elevation (DEM)
            dx (float): Grid spacing
            dt (float): Time step
            g (float): Gravity acceleration
            n (float): Manning roughness coefficient
            rainfall (ndarray): Rainfall intensity
            
        Returns:
            tuple: Updated h, u, v
        """
        ny, nx = h.shape
        h_new = np.zeros_like(h)
        u_new = np.zeros_like(u)
        v_new = np.zeros_like(v)
        
        # Compute fluxes
        for i in prange(1, ny-1):
            for j in range(1, nx-1):
                # Skip if the cell is not in the computational domain
                if h[i, j] < 1e-6 and rainfall[i, j] < 1e-9:
                    continue
                
                # Water surface elevation
                wse = h[i, j] + z[i, j]
                
                # x-direction flux
                if j < nx-1:
                    # East face
                    h_e = 0.5 * (h[i, j] + h[i, j+1])
                    if h_e > 1e-6:
                        # Compute momentum flux
                        u_e = 0.5 * (u[i, j] + u[i, j+1])
                        flux_e = h_e * u_e
                        
                        # Manning friction term
                        n_term = g * n**2 * u_e * np.abs(u_e) / h_e**(1/3)
                        
                        # Source term (pressure gradient)
                        source_e = -g * h_e * (z[i, j+1] + h[i, j+1] - z[i, j] - h[i, j]) / dx
                        
                        # Update velocity
                        u_new[i, j] = u[i, j] + dt * (source_e - n_term)
                        
                        # Update water depth
                        h_new[i, j] -= dt * flux_e / dx
                        h_new[i, j+1] += dt * flux_e / dx
                
                # y-direction flux
                if i < ny-1:
                    # North face
                    h_n = 0.5 * (h[i, j] + h[i+1, j])
                    if h_n > 1e-6:
                        # Compute momentum flux
                        v_n = 0.5 * (v[i, j] + v[i+1, j])
                        flux_n = h_n * v_n
                        
                        # Manning friction term
                        n_term = g * n**2 * v_n * np.abs(v_n) / h_n**(1/3)
                        
                        # Source term (pressure gradient)
                        source_n = -g * h_n * (z[i+1, j] + h[i+1, j] - z[i, j] - h[i, j]) / dx
                        
                        # Update velocity
                        v_new[i, j] = v[i, j] + dt * (source_n - n_term)
                        
                        # Update water depth
                        h_new[i, j] -= dt * flux_n / dx
                        h_new[i+1, j] += dt * flux_n / dx
                
                # Add rainfall
                h_new[i, j] += dt * rainfall[i, j]
        
        return h_new, u_new, v_new
    
    @jit(nopython=True)
    def _green_ampt_infiltration(self, h, soil_moisture, K_s, psi_f, theta_i, theta_s, dt):
        """Calculate infiltration using Green-Ampt method.
        
        Parameters:
            h (ndarray): Water depth
            soil_moisture (ndarray): Current soil moisture content
            K_s (ndarray): Saturated hydraulic conductivity
            psi_f (float): Suction head at wetting front
            theta_i (float): Initial moisture content
            theta_s (float): Saturated moisture content
            dt (float): Time step
            
        Returns:
            tuple: Infiltration amount, updated soil moisture
        """
        ny, nx = h.shape
        infiltration = np.zeros_like(h)
        new_moisture = soil_moisture.copy()
        
        # Available storage in soil
        available_storage = theta_s - soil_moisture
        
        # Only infiltrate where there's water
        wet_cells = h > 1e-6
        
        # Potential infiltration rate
        F = np.zeros_like(h)
        F[wet_cells] = K_s[wet_cells] * (1 + psi_f * (theta_s - theta_i) / F[wet_cells])
        
        # Actual infiltration limited by available water and storage
        infiltration = np.minimum(h, F * dt)
        infiltration = np.minimum(infiltration, available_storage * dt)
        
        # Update soil moisture
        new_moisture += infiltration / dt  # Convert volume to content
        
        return infiltration, new_moisture
    
    def _simulate_drainage(self, h, district_map, dt):
        """Simulate water drainage through urban drainage system.
        
        Parameters:
            h (ndarray): Water depth
            district_map (ndarray): Map of district indices
            dt (float): Time step
            
        Returns:
            ndarray: Drainage amount
        """
        # Base drainage capacity
        base_capacity = self.model_params['drainage']['capacity'] * (dt / 3600)  # Convert to m per timestep
        
        # Initialize drainage array
        drainage = np.zeros_like(h)
        
        # Apply drainage capacity by district
        for district_id, factor in self.model_params['drainage']['upgrade_factor'].items():
            district_num = int(district_id.split('_')[1])
            district_mask = district_map == district_num
            
            # Scale by district specific factor
            district_capacity = base_capacity * factor
            
            # Drainage limited by available water
            drainage[district_mask] = np.minimum(h[district_mask], district_capacity)
        
        return drainage
    
    def set_from_intensity(self, intensity_file, return_periods=None):
        """Set hazard from intensity file (NetCDF or raster).
        
        Parameters:
            intensity_file (str): Path to intensity file
            return_periods (list): List of return periods to extract
            
        Returns:
            ZurichFloodHazard: self
        """
        # Load intensity data
        try:
            if intensity_file.endswith('.nc'):
                dataset = xr.open_dataset(intensity_file)
                # Extract flood depths for given return periods
                if return_periods is None:
                    return_periods = [10, 30, 100, 300]
                
                # Process datasets and construct event set
                event_ids = []
                event_names = []
                frequencies = []
                intensity_list = []
                
                for rp in return_periods:
                    event_ids.append(int(rp))
                    event_names.append(f"RP{rp}")
                    frequencies.append(1.0/rp)
                    
                    # Extract flood depth for this return period
                    if f"depth_rp{rp}" in dataset:
                        depth = dataset[f"depth_rp{rp}"].values
                    else:
                        # If specific return period not available, interpolate
                        available_rps = [int(v.split('_rp')[1]) for v in dataset.variables 
                                        if v.startswith('depth_rp')]
                        available_rps.sort()
                        
                        if rp < min(available_rps):
                            depth = dataset[f"depth_rp{min(available_rps)}"].values * (rp / min(available_rps))
                        elif rp > max(available_rps):
                            depth = dataset[f"depth_rp{max(available_rps)}"].values * (rp / max(available_rps))
                        else:
                            # Linear interpolation
                            lower_rp = max([r for r in available_rps if r < rp])
                            upper_rp = min([r for r in available_rps if r > rp])
                            lower_depth = dataset[f"depth_rp{lower_rp}"].values
                            upper_depth = dataset[f"depth_rp{upper_rp}"].values
                            
                            weight = (rp - lower_rp) / (upper_rp - lower_rp)
                            depth = lower_depth * (1-weight) + upper_depth * weight
                    
                    # Flatten and store
                    intensity_list.append(depth.flatten())
                
                # Get grid coordinates
                lats = dataset['latitude'].values
                lons = dataset['longitude'].values
                
                # Create centroids
                xx, yy = np.meshgrid(lons, lats)
                coord_x = xx.flatten()
                coord_y = yy.flatten()
                
                # Set hazard attributes
                self.centroids.set_lat_lon(coord_y, coord_x)
                self.event_id = np.array(event_ids)
                self.event_name = event_names
                self.frequency = np.array(frequencies)
                self.intensity = csr_matrix(np.vstack(intensity_list))
                self.fraction = self.intensity.copy()
                self.fraction.data[:] = 1
                
            elif intensity_file.endswith(('.tif', '.asc')):
                # For raster files, assume it represents a single event
                with rasterio.open(intensity_file) as src:
                    depth = src.read(1)
                    transform = src.transform
                    
                    # Create a grid of coordinates
                    height, width = depth.shape
                    rows, cols = np.mgrid[0:height, 0:width]
                    x, y = rasterio.transform.xy(transform, rows, cols)
                    
                    # Flatten arrays
                    depth_flat = depth.flatten()
                    x_flat = np.array(x).flatten()
                    y_flat = np.array(y).flatten()
                    
                    # Filter out no-data values
                    valid = depth_flat != src.nodata
                    depth_flat = depth_flat[valid]
                    x_flat = x_flat[valid]
                    y_flat = y_flat[valid]
                    
                    # Set centroids
                    self.centroids.set_lat_lon(y_flat, x_flat)
                    
                    # Set single event
                    self.event_id = np.array([1])
                    self.event_name = ["Flood event"]
                    self.frequency = np.array([0.01])  # Assuming 100yr event
                    self.intensity = csr_matrix(depth_flat.reshape(1, -1))
                    self.fraction = self.intensity.copy()
                    self.fraction.data[:] = 1
            else:
                raise ValueError(f"Unsupported file format: {intensity_file}")
                
        except Exception as e:
            raise ValueError(f"Failed to load intensity file: {str(e)}")
        
        return self
    
    def simulate_events(self, num_events=10):
        """Simulate flood events using the shallow water model.
        
        Parameters:
            num_events (int): Number of events to simulate
            
        Returns:
            ZurichFloodHazard: self
        """
        if self.rainfall_data is None:
            raise ValueError("Rainfall data not loaded. Call set_from_copernicus_cds first.")
        
        if self.dem_data is None:
            raise ValueError("DEM data not loaded. Call load_zurich_data first.")
        
        # Initialize multi-processing pool
        if self.pool is None:
            self.pool = mp.Pool(processes=min(num_events, mp.cpu_count()))
        
        # Create district map for drainage simulation
        district_map = np.zeros_like(self.dem_data, dtype=int)
        
        # This would use a more sophisticated method in a real implementation
        # For demonstration, we use a simple radial pattern
        ny, nx = self.dem_data.shape
        center_y, center_x = ny // 2, nx // 2
        
        for y in range(ny):
            for x in range(nx):
                # Calculate angle from center
                angle = np.arctan2(y - center_y, x - center_x)
                # Normalize to 0-12 range
                district_idx = int(((angle + np.pi) / (2 * np.pi) * 12) % 12) + 1
                district_map[y, x] = district_idx
        
        # Sample rainfall events
        event_ids = []
        event_names = []
        frequencies = []
        date_stamps = []
        intensity_list = []
        
        # Select random time slices for events
        time_indices = np.random.choice(len(self.rainfall_data.time), size=num_events, replace=False)
        
        # Parallel simulation of events
        event_args = [(i, self.rainfall_data.isel(time=i).tp.values, 
                      self.dem_data, district_map, self.model_params) 
                      for i in time_indices]
        
        # Execute simulations in parallel
        results = self.pool.map(self._simulate_single_event, event_args)
        
        # Process results
        for i, (max_depth, event_time) in enumerate(results):
            event_date = self.rainfall_data.time[time_indices[i]].values
            
            event_ids.append(i+1)
            event_names.append(f"Event_{event_date}")
            
            # Calculate frequency (more intense events are rarer)
            mean_rainfall = self.rainfall_data.isel(time=time_indices[i]).tp.mean().values
            if mean_rainfall > 10:  # Heavy rain
                frequency = 1/100
            elif mean_rainfall > 5:  # Moderate rain
                frequency = 1/30
            else:  # Light rain
                frequency = 1/10
                
            frequencies.append(frequency)
            date_stamps.append(np.datetime64(event_date))
            
            # Store max depth
            intensity_list.append(max_depth.flatten())
        
        # Get grid coordinates
        ny, nx = self.dem_data.shape
        y = np.linspace(47.3, 47.4, ny)
        x = np.linspace(8.4, 8.6, nx)
        xx, yy = np.meshgrid(x, y)
        
        # Set hazard attributes
        self.centroids.set_lat_lon(yy.flatten(), xx.flatten())
        self.event_id = np.array(event_ids)
        self.event_name = event_names
        self.frequency = np.array(frequencies)
        self.date = np.array(date_stamps)
        self.intensity = csr_matrix(np.vstack(intensity_list))
        self.fraction = self.intensity.copy()
        self.fraction.data[:] = 1
        
        print(f"Successfully simulated {num_events} flood events")
        return self
    
    def _simulate_single_event(self, args):
        """Simulate a single flood event (called by multiprocessing).
        
        Parameters:
            args (tuple): Tuple containing (event_idx, rainfall, dem, district_map, model_params)
            
        Returns:
            tuple: (max_depth, event_time)
        """
        event_idx, rainfall, dem, district_map, model_params = args
        
        # Extract model parameters
        dx = model_params['dx']
        dt = model_params['dt']
        g = model_params['g']
        n = model_params['manning_n']
        sim_duration = model_params['sim_duration']
        
        # Initialize state variables
        ny, nx = dem.shape
        h = np.zeros((ny, nx))  # Water depth
        u = np.zeros((ny, nx))  # x-velocity
        v = np.zeros((ny, nx))  # y-velocity
        
        # Initialize soil moisture
        soil_moisture = np.ones((ny, nx)) * model_params['infiltration']['initial_moisture']
        
        # Initialize infiltration parameters
        theta_i = model_params['infiltration']['initial_moisture']
        theta_s = model_params['infiltration']['saturated_moisture']
        psi_f = model_params['infiltration']['suction_head']
        
        # Create hydraulic conductivity map based on districts
        k_s = np.zeros((ny, nx))
        for y in range(ny):
            for x in range(nx):
                district = district_map[y, x]
                if district <= 5:  # Inner city
                    k_s[y, x] = model_params['infiltration']['hydraulic_conductivity']['urban']
                elif district <= 10:  # Outer districts
                    k_s[y, x] = model_params['infiltration']['hydraulic_conductivity']['suburban']
                else:  # Outskirts
                    k_s[y, x] = model_params['infiltration']['hydraulic_conductivity']['parks']
        
        # Storage for maximum depth
        max_depth = np.zeros((ny, nx))
        
        # Convert rainfall from mm/h to m/s
        rainfall_rate = rainfall / 1000 / 3600
        
        # Main simulation loop
        num_steps = int(sim_duration / dt)
        for step in range(num_steps):
            t = step * dt
            
            # Calculate infiltration
            infiltration, soil_moisture = self._green_ampt_infiltration(
                h, soil_moisture, k_s, psi_f, theta_i, theta_s, dt)
            
            # Calculate drainage
            drainage = self._simulate_drainage(h, district_map, dt)
            
            # Update water depth due to infiltration and drainage
            h -= (infiltration + drainage)
            h = np.maximum(h, 0)  # Ensure non-negative depth
            
            # Shallow water update
            h, u, v = self._shallow_water_step(h, u, v, dem, dx, dt, g, n, rainfall_rate)
            
            # Update maximum depth
            max_depth = np.maximum(max_depth, h)
            
            # CFL condition check
            water_cells = h > 1e-6
            if water_cells.any():
                max_vel = max(np.max(np.abs(u[water_cells])), np.max(np.abs(v[water_cells])))
                if max_vel > 0:
                    wave_speed = np.sqrt(g * np.max(h[water_cells]))
                    cfl = (max_vel + wave_speed) * dt / dx
                    if cfl > model_params['cfl_number']:
                        # Reduce timestep and retry
                        dt_new = model_params['cfl_number'] * dx / (max_vel + wave_speed)
                        dt = max(dt_new * 0.8, dt/2)  # Apply safety factor
            
        # Return maximum depth and event time
        return max_depth, t
    
    def plot_hazard_map(self, event_id=None, ax=None, cmap='Blues', **kwargs):
        """Plot flood hazard map for a specified event.
        
        Parameters:
            event_id (int, optional): Event ID to plot. If None, plots the first event.
            ax (matplotlib.axes, optional): Axes to plot on.
            cmap (str): Colormap name
            **kwargs: Additional arguments to pass to matplotlib
            
        Returns:
            matplotlib.axes: Axes with the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        
        if event_id is None:
            event_idx = 0
        else:
            event_idx = np.where(self.event_id == event_id)[0][0]
        
        # Extract event data
        event_intensity = self.intensity[event_idx].toarray().reshape(-1)
        
        # Get coordinates
        coords_x = self.centroids.lon
        coords_y = self.centroids.lat
        
        # Create grid for plotting
        grid_size = int(np.sqrt(len(coords_x)))
        grid_x = coords_x.reshape(grid_size, grid_size)
        grid_y = coords_y.reshape(grid_size, grid_size)
        grid_z = event_intensity.reshape(grid_size, grid_size)
        
        # Plot flood depth
        im = ax.pcolormesh(grid_x, grid_y, grid_z, cmap=cmap, **kwargs)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Flood Depth (m)')
        
        # Add districts if available
        if self.districts is not None:
            self.districts.boundary.plot(ax=ax, color='black', linewidth=0.5)
            
            # Add district labels
            for idx, row in self.districts.iterrows():
                ax.annotate(row['name'], xy=row.geometry.centroid.coords[0], 
                          ha='center', fontsize=8)
        
        # Add lake if available
        if self.lake_zurich is not None:
            ax.fill(*self.lake_zurich.exterior.xy, color='lightblue', alpha=0.6, 
                    label='Lake Zurich')
        
        # Set title and labels
        ax.set_title(f"Flood Hazard Map: {self.event_name[event_idx]}")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Add legend
        ax.legend()
        
        return ax
