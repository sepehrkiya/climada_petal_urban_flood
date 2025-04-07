import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import pickle
from climada.entity import Entity, Exposures, DiscRates
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.entity.measures import MeasureSet, Measure
from climada.util.constants import SYSTEM_DIR

class ZurichFloodEntity(Entity):
    """ZurichFloodEntity class extends CLIMADA Entity for Zurich flood risk assessment.
    
    This class manages the full entity configuration for Zurich flood risk assessment,
    including exposure, impact functions, discount rates, and adaptation measures.
    
    Attributes:
        exposures (Exposures): Exposures instance for Zurich
        impact_funcs (ImpactFuncSet): Impact functions for flooding
        disc_rates (DiscRates): Discount rates for cost-benefit calculations
        measure_set (MeasureSet): Adaptation measures
        residential_vfs (list): Vulnerability functions for residential buildings
        commercial_vfs (list): Vulnerability functions for commercial buildings
        industrial_vfs (list): Vulnerability functions for industrial buildings
        infrastructure_vfs (list): Vulnerability functions for infrastructure
    """
    
    def __init__(self):
        """Initialize Zurich flood entity with default parameters."""
        super().__init__()
        self.residential_vfs = None
        self.commercial_vfs = None
        self.industrial_vfs = None
        self.infrastructure_vfs = None
    
    def set_default_impact_functions(self):
        """Set default flood impact functions for different asset types.
        
        Returns:
            ZurichFloodEntity: self
        """
        # Create impact function set
        impact_funcs = ImpactFuncSet()
        
        # Generate vulnerability functions for different building types and ages
        
        # Residential vulnerability functions (1-5 classes from least to most vulnerable)
        self.residential_vfs = []
        
        # Class 1: Modern residential buildings (2000+) with flood protection
        res_vf1 = ImpactFunc()
        res_vf1.haz_type = 'FL'
        res_vf1.id = 1
        res_vf1.name = 'Residential - Class 1 (Modern)'
        # Depth-damage relationship (depth in m, damage as fraction of value)
        depths = np.array([0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
        mdf = np.array([0.00, 0.05, 0.12, 0.25, 0.35, 0.45, 0.55, 0.60, 0.65])
        res_vf1.set_step_from_arrays(depths, mdf)
        impact_funcs.append(res_vf1)
        self.residential_vfs.append(res_vf1)
        
        # Class 2: High-quality residential buildings (1980-2000)
        res_vf2 = ImpactFunc()
        res_vf2.haz_type = 'FL'
        res_vf2.id = 2
        res_vf2.name = 'Residential - Class 2 (High quality)'
        mdf = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70])
        res_vf2.set_step_from_arrays(depths, mdf)
        impact_funcs.append(res_vf2)
        self.residential_vfs.append(res_vf2)
        
        # Class 3: Medium-quality residential buildings (1950-1980)
        res_vf3 = ImpactFunc()
        res_vf3.haz_type = 'FL'
        res_vf3.id = 3
        res_vf3.name = 'Residential - Class 3 (Medium quality)'
        mdf = np.array([0.00, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.70, 0.75])
        res_vf3.set_step_from_arrays(depths, mdf)
        impact_funcs.append(res_vf3)
        self.residential_vfs.append(res_vf3)
        
        # Class 4: Lower-quality residential buildings (1900-1950)
        res_vf4 = ImpactFunc()
        res_vf4.haz_type = 'FL'
        res_vf4.id = 4
        res_vf4.name = 'Residential - Class 4 (Lower quality)'
        mdf = np.array([0.00, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80])
        res_vf4.set_step_from_arrays(depths, mdf)
        impact_funcs.append(res_vf4)
        self.residential_vfs.append(res_vf4)
        
        # Class 5: Old residential buildings (pre-1900)
        res_vf5 = ImpactFunc()
        res_vf5.haz_type = 'FL'
        res_vf5.id = 5
        res_vf5.name = 'Residential - Class 5 (Old)'
        mdf = np.array([0.00, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.80, 0.85])
        res_vf5.set_step_from_arrays(depths, mdf)
        impact_funcs.append(res_vf5)
        self.residential_vfs.append(res_vf5)
        
        # Commercial vulnerability functions
        self.commercial_vfs = []
        
        # Class 1: Modern commercial (2000+)
        com_vf1 = ImpactFunc()
        com_vf1.haz_type = 'FL'
        com_vf1.id = 11
        com_vf1.name = 'Commercial - Class 1 (Modern)'
        mdf = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70])
        com_vf1.set_step_from_arrays(depths, mdf)
        impact_funcs.append(com_vf1)
        self.commercial_vfs.append(com_vf1)
        
        # Class 2: High-quality commercial (1980-2000)
        com_vf2 = ImpactFunc()
        com_vf2.haz_type = 'FL'
        com_vf2.id = 12
        com_vf2.name = 'Commercial - Class 2 (High quality)'
        mdf = np.array([0.00, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.70, 0.75])
        com_vf2.set_step_from_arrays(depths, mdf)
        impact_funcs.append(com_vf2)
        self.commercial_vfs.append(com_vf2)
        
        # Class 3: Medium-quality commercial (1950-1980)
        com_vf3 = ImpactFunc()
        com_vf3.haz_type = 'FL'
        com_vf3.id = 13
        com_vf3.name = 'Commercial - Class 3 (Medium quality)'
        mdf = np.array([0.00, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80])
        com_vf3.set_step_from_arrays(depths, mdf)
        impact_funcs.append(com_vf3)
        self.commercial_vfs.append(com_vf3)
        
        # Class 4: Lower-quality commercial (1900-1950)
        com_vf4 = ImpactFunc()
        com_vf4.haz_type = 'FL'
        com_vf4.id = 14
        com_vf4.name = 'Commercial - Class 4 (Lower quality)'
        mdf = np.array([0.00, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.80, 0.85])
        com_vf4.set_step_from_arrays(depths, mdf)
        impact_funcs.append(com_vf4)
        self.commercial_vfs.append(com_vf4)
        
        # Class 5: Old commercial (pre-1900)
        com_vf5 = ImpactFunc()
        com_vf5.haz_type = 'FL'
        com_vf5.id = 15
        com_vf5.name = 'Commercial - Class 5 (Old)'
        mdf = np.array([0.00, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90])
        com_vf5.set_step_from_arrays(depths, mdf)
        impact_funcs.append(com_vf5)
        self.commercial_vfs.append(com_vf5)
        
        # Industrial vulnerability functions
        self.industrial_vfs = []
        
        # Class 1: Modern industrial (2000+)
        ind_vf1 = ImpactFunc()
        ind_vf1.haz_type = 'FL'
        ind_vf1.id = 21
        ind_vf1.name = 'Industrial - Class 1 (Modern)'
        mdf = np.array([0.00, 0.10, 0.20, 0.35, 0.45, 0.55, 0.65, 0.70, 0.75])
        ind_vf1.set_step_from_arrays(depths, mdf)
        impact_funcs.append(ind_vf1)
        self.industrial_vfs.append(ind_vf1)
        
        # Class 2: High-quality industrial (1980-2000)
        ind_vf2 = ImpactFunc()
        ind_vf2.haz_type = 'FL'
        ind_vf2.id = 22
        ind_vf2.name = 'Industrial - Class 2 (High quality)'
        mdf = np.array([0.00, 0.15, 0.30, 0.45, 0.55, 0.65, 0.75, 0.80, 0.85])
        ind_vf2.set_step_from_arrays(depths, mdf)
        impact_funcs.append(ind_vf2)
        self.industrial_vfs.append(ind_vf2)
        
        # Class 3: Medium-quality industrial (1950-1980)
        ind_vf3 = ImpactFunc()
        ind_vf3.haz_type = 'FL'
        ind_vf3.id = 23
        ind_vf3.name = 'Industrial - Class 3 (Medium quality)'
        mdf = np.array([0.00, 0.20, 0.35, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90])
        ind_vf3.set_step_from_arrays(depths, mdf)
        impact_funcs.append(ind_vf3)
        self.industrial_vfs.append(ind_vf3)
        
        # Class 4: Lower-quality industrial (1900-1950)
        ind_vf4 = ImpactFunc()
        ind_vf4.haz_type = 'FL'
        ind_vf4.id = 24
        ind_vf4.name = 'Industrial - Class 4 (Lower quality)'
        mdf = np.array([0.00, 0.25, 0.40, 0.55, 0.65, 0.75, 0.85, 0.90, 0.95])
        ind_vf4.set_step_from_arrays(depths, mdf)
        impact_funcs.append(ind_vf4)
        self.industrial_vfs.append(ind_vf4)
        
        # Class 5: Old industrial (pre-1900)
        ind_vf5 = ImpactFunc()
        ind_vf5.haz_type = 'FL'
        ind_vf5.id = 25
        ind_vf5.name = 'Industrial - Class 5 (Old)'
        mdf = np.array([0.00, 0.30, 0.45, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00])
        ind_vf5.set_step_from_arrays(depths, mdf)
        impact_funcs.append(ind_vf5)
        self.industrial_vfs.append(ind_vf5)
        
        # Infrastructure vulnerability functions
        self.infrastructure_vfs = []
        
        # Roads
        road_vf = ImpactFunc()
        road_vf.haz_type = 'FL'
        road_vf.id = 31
        road_vf.name = 'Roads'
        mdf = np.array([0.00, 0.05, 0.15, 0.30, 0.50, 0.65, 0.80, 0.90, 0.95])
        road_vf.set_step_from_arrays(depths, mdf)
        impact_funcs.append(road_vf)
        self.infrastructure_vfs.append(road_vf)
        
        # Bridges
        bridge_vf = ImpactFunc()
        bridge_vf.haz_type = 'FL'
        bridge_vf.id = 32
        bridge_vf.name = 'Bridges'
        mdf = np.array([0.00, 0.00, 0.05, 0.10, 0.20, 0.35, 0.55, 0.75, 0.90])
        bridge_vf.set_step_from_arrays(depths, mdf)
        impact_funcs.append(bridge_vf)
        self.infrastructure_vfs.append(bridge_vf)
        
        # Railways
        railway_vf = ImpactFunc()
        railway_vf.haz_type = 'FL'
        railway_vf.id = 33
        railway_vf.name = 'Railways'
        mdf = np.array([0.00, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95, 1.00])
        railway_vf.set_step_from_arrays(depths, mdf)
        impact_funcs.append(railway_vf)
        self.infrastructure_vfs.append(railway_vf)
        
        # Utilities
        utility_vf = ImpactFunc()
        utility_vf.haz_type = 'FL'
        utility_vf.id = 34
        utility_vf.name = 'Utilities'
        mdf = np.array([0.00, 0.05, 0.20, 0.40, 0.60, 0.75, 0.85, 0.95, 1.00])
        utility_vf.set_step_from_arrays(depths, mdf)
        impact_funcs.append(utility_vf)
        self.infrastructure_vfs.append(utility_vf)
        
        # Set impact function set
        self.impact_funcs = impact_funcs
        
        print(f"Initialized {len(impact_funcs.get_func())} impact functions")
        return self
    
    def set_default_discount_rates(self):
        """Set default discount rates for cost-benefit calculations.
        
        Returns:
            ZurichFloodEntity: self
        """
        # Initialize discount rates
        disc_rates = DiscRates()
        
        # Set rates for different time periods
        # Short term (0-10 years): 3%
        # Medium term (11-30 years): 2.5%
        # Long term (31+ years): 2%
        years = np.arange(0, 101)
        rates = np.zeros(len(years))
        rates[0:11] = 0.03  # 3% for years 0-10
        rates[11:31] = 0.025  # 2.5% for years 11-30
        rates[31:] = 0.02  # 2% for years 31+
        
        disc_rates.years = years
        disc_rates.rates = rates
        
        self.disc_rates = disc_rates
        
        print("Set default discount rates")
        return self
    
    def set_adaptation_measures(self):
        """Set adaptation measures for flood risk reduction.
        
        Returns:
            ZurichFloodEntity: self
        """
        # Initialize measure set
        measure_set = MeasureSet()
        
        # 1. Property-level measures
        # Dry floodproofing
        dry_floodproof = Measure()
        dry_floodproof.name = 'Dry floodproofing'
        dry_floodproof.haz_type = 'FL'
        dry_floodproof.color_rgb = np.array([0.6, 0.1, 0.1])
        
        # Cost parameters
        dry_floodproof.cost = 15000  # CHF per building (average)
        dry_floodproof.mdd_impact = (0, -0.3)  # Reduces mean damage by 30%
        dry_floodproof.paa_impact = (0, -0.1)  # Reduces affected probability by 10%
        dry_floodproof.hazard_inten_imp = (0, 0)  # No direct impact on hazard intensity
        
        # Apply to buildings with depths under 1m
        def apply_dry_floodproof(exposure, hazard, impf_set):
            # New impact function with reduced damage for shallow flooding
            for impf in impf_set.get_func():
                # Create a copy of the original impact function
                new_impf = ImpactFunc()
                new_impf.haz_type = impf.haz_type
                new_impf.id = impf.id + 1000  # New ID
                new_impf.name = impf.name + " (dry floodproofed)"
                
                # Modify intensity_unit
                new_impf.intensity_unit = impf.intensity_unit
                
                # Get original damage functions
                orig_inten = impf.intensity
                orig_mdd = impf.mdd
                orig_paa = impf.paa
                
                # Reduce damage for shallow flooding (< 1m)
                new_mdd = orig_mdd.copy()
                for i in range(len(orig_inten)):
                    if orig_inten[i] < 1.0:
                        new_mdd[i] = max(0, orig_mdd[i] * 0.7)  # 30% reduction
                
                # Set the modified impact function
                new_impf.intensity = orig_inten
                new_impf.mdd = new_mdd
                new_impf.paa = orig_paa
                
                # Add to impact function set
                impf_set.append(new_impf)
            
            return exposure, hazard, impf_set
        
        dry_floodproof.apply = apply_dry_floodproof
        measure_set.append(dry_floodproof)
        
        # 2. Wet floodproofing
        wet_floodproof = Measure()
        wet_floodproof.name = 'Wet floodproofing'
        wet_floodproof.haz_type = 'FL'
        wet_floodproof.color_rgb = np.array([0.1, 0.5, 0.1])
        
        # Cost parameters
        wet_floodproof.cost = 25000  # CHF per building
        wet_floodproof.mdd_impact = (0, -0.4)  # Reduces mean damage by 40%
        wet_floodproof.paa_impact = (0, 0)  # No reduction in affected probability
        
        def apply_wet_floodproof(exposure, hazard, impf_set):
            # Similar to dry floodproofing but effective for deeper floods
            for impf in impf_set.get_func():
                new_impf = ImpactFunc()
                new_impf.haz_type = impf.haz_type
                new_impf.id = impf.id + 2000
                new_impf.name = impf.name + " (wet floodproofed)"
                new_impf.intensity_unit = impf.intensity_unit
                
                orig_inten = impf.intensity
                orig_mdd = impf.mdd
                orig_paa = impf.paa
                
                # Reduce damage for all flood depths
                new_mdd = orig_mdd.copy() * 0.6  # 40% reduction
                
                new_impf.intensity = orig_inten
                new_impf.mdd = new_mdd
                new_impf.paa = orig_paa
                
                impf_set.append(new_impf)
            
            return exposure, hazard, impf_set
        
        wet_floodproof.apply = apply_wet_floodproof
        measure_set.append(wet_floodproof)
        
        # 3. Neighborhood flood barriers
        flood_barriers = Measure()
        flood_barriers.name = 'Neighborhood flood barriers'
        flood_barriers.haz_type = 'FL'
        flood_barriers.color_rgb = np.array([0.1, 0.1, 0.6])
        
        # Cost parameters (higher cost but more effective)
        flood_barriers.cost = 5000000  # CHF per neighborhood
        flood_barriers.mdd_impact = (0, 0)  # No direct MDD impact
        flood_barriers.paa_impact = (0, 0)  # No direct PAA impact
        
        def apply_flood_barriers(exposure, hazard, impf_set):
            # Modify hazard by reducing flood depths
            if hazard.intensity.shape[0] > 0:
                # Reduce flood depths by 0.5m (up to complete elimination)
                intensity_copy = hazard.intensity.copy()
                intensity_copy.data = np.maximum(0, intensity_copy.data - 0.5)
                hazard.intensity = intensity_copy
            
            return exposure, hazard, impf_set
        
        flood_barriers.apply = apply_flood_barriers
        measure_set.append(flood_barriers)
        
        # 4. Improved drainage system
        improved_drainage = Measure()
        improved_drainage.name = 'Improved drainage'
        improved_drainage.haz_type = 'FL'
        improved_drainage.color_rgb = np.array([0.6, 0.6, 0.1])
        
        # Cost parameters
        improved_drainage.cost = 10000000  # CHF per district
        improved_drainage.mdd_impact = (0, 0)  # No direct MDD impact
        improved_drainage.paa_impact = (0, 0)  # No direct PAA impact
        
        def apply_improved_drainage(exposure, hazard, impf_set):
            # Reduce flooding for smaller events more than larger events
            if hazard.intensity.shape[0] > 0:
                intensity_copy = hazard.intensity.copy()
                
                # Apply different reductions based on event frequency
                for i, freq in enumerate(hazard.frequency):
                    reduction_factor = min(0.8, freq * 10)  # More reduction for frequent events
                    event_intensity = intensity_copy[i].copy()
                    event_intensity.data = event_intensity.data * (1 - reduction_factor)
                    intensity_copy[i] = event_intensity
                
                hazard.intensity = intensity_copy
            
            return exposure, hazard, impf_set
        
        improved_drainage.apply = apply_improved_drainage
        measure_set.append(improved_drainage)
        
        # 5. Retention basins
        retention_basins = Measure()
        retention_basins.name = 'Retention basins'
        retention_basins.haz_type = 'FL'
        retention_basins.color_rgb = np.array([0.1, 0.6, 0.6])
        
        # Cost parameters
        retention_basins.cost = 20000000  # CHF per installation
        retention_basins.mdd_impact = (0, 0)  # No direct MDD impact
        retention_basins.paa_impact = (0, 0)  # No direct PAA impact
        
        def apply_retention_basins(exposure, hazard, impf_set):
            # Reduce peak flood depths for all events
            if hazard.intensity.shape[0] > 0:
                intensity_copy = hazard.intensity.copy()
                
                # Reduce all flood depths by 30%
                intensity_copy.data = intensity_copy.data * 0.7
                
                hazard.intensity = intensity_copy
            
            return exposure, hazard, impf_set
        
        retention_basins.apply = apply_retention_basins
        measure_set.append(retention_basins)
        
        # Set the measure set
        self.measure_set = measure_set
        
        print(f"Initialized {len(measure_set.get_measure())} adaptation measures")
        return self
    
    def assign_impact_functions(self, exposure):
        """Assign appropriate impact functions to exposure points based on attributes.
        
        Parameters:
            exposure (Exposures): Exposure instance
            
        Returns:
            Exposures: Updated exposure with impact function IDs
        """
        if exposure.gdf is None:
            raise ValueError("Exposure has no GeoDataFrame")
        
        if self.impact_funcs is None:
            raise ValueError("Impact functions not initialized")
        
        # Initialize impact function ID array
        impact_ids = np.zeros(exposure.gdf.shape[0], dtype=int)
        
        for i, (_, row) in enumerate(exposure.gdf.iterrows()):
            asset_type = row.get('type', 'residential')
            vuln_class = row.get('vuln_class', 3)  # Default to medium vulnerability
            
            # Cap vulnerability class between 1-5
            vuln_class = max(1, min(5, vuln_class))
            
            # Assign impact function ID based on asset type and vulnerability class
            if asset_type == 'residential':
                impact_ids[i] = vuln_class  # IDs 1-5
            elif asset_type == 'commercial':
                impact_ids[i] = 10 + vuln_class  # IDs 11-15
            elif asset_type == 'industrial':
                impact_ids[i] = 20 + vuln_class  # IDs 21-25
            elif asset_type == 'road':
                impact_ids[i] = 31  # Road ID
            elif asset_type == 'railway':
                impact_ids[i] = 33  # Railway ID
            elif asset_type == 'bridge':
                impact_ids[i] = 32  # Bridge ID
            elif asset_type == 'utility':
                impact_ids[i] = 34  # Utility ID
            else:
                # Default to medium residential if type is unknown
                impact_ids[i] = 3
        
        # Add impact function IDs to exposure
        exposure.gdf['impf_FL'] = impact_ids
        exposure.impact_funcs = {'FL': impact_ids}
        
        print(f"Assigned impact functions to {len(impact_ids)} exposure points")
        return exposure
    
    def plot_impact_functions(self, asset_type=None, ax=None):
        """Plot impact functions for the specified asset type.
        
        Parameters:
            asset_type (str, optional): Asset type to plot ('residential', 'commercial', 
                                        'industrial', or 'infrastructure')
            ax (matplotlib.axes, optional): Axes to plot on
            
        Returns:
            matplotlib.axes: Axes with the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        if self.impact_funcs is None:
            raise ValueError("Impact functions not initialized")
        
        # Select which functions to plot
        if asset_type == 'residential' and self.residential_vfs:
            funcs_to_plot = self.residential_vfs
            title = 'Residential Building Vulnerability Functions'
        elif asset_type == 'commercial' and self.commercial_vfs:
            funcs_to_plot = self.commercial_vfs
            title = 'Commercial Building Vulnerability Functions'
        elif asset_type == 'industrial' and self.industrial_vfs:
            funcs_to_plot = self.industrial_vfs
            title = 'Industrial Building Vulnerability Functions'
        elif asset_type == 'infrastructure' and self.infrastructure_vfs:
            funcs_to_plot = self.infrastructure_vfs
            title = 'Infrastructure Vulnerability Functions'
        else:
            # Plot all functions
            funcs_to_plot = self.impact_funcs.get_func()
            title = 'All Vulnerability Functions'
        
        # Plot each function
        for func in funcs_to_plot:
            ax.plot(func.intensity, func.mdd, label=func.name)
        
        ax.set_xlabel('Flood Depth (m)')
        ax.set_ylabel('Mean Damage Ratio')
        ax.set_title(title)
        ax.grid(True)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1)
        ax.legend()
        
        return ax
    
    def save(self, file_path=None):
        """Save the entity to a file.
        
        Parameters:
            file_path (str, optional): Path to save the entity
            
        Returns:
            str: Path where entity was saved
        """
        if file_path is None:
            file_path = os.path.join(SYSTEM_DIR, 'data', 'zurich_flood_entity.pkl')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save entity
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Entity saved to {file_path}")
        return file_path
    
    @classmethod
    def load(cls, file_path=None):
        """Load entity from a file.
        
        Parameters:
            file_path (str, optional): Path to the entity file
            
        Returns:
            ZurichFloodEntity: Loaded entity
        """
        if file_path is None:
            file_path = os.path.join(SYSTEM_DIR, 'data', 'zurich_flood_entity.pkl')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Entity file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            entity = pickle.load(f)
        
        if not isinstance(entity, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        
        print(f"Loaded entity from {file_path}")
        return entity
        
        
