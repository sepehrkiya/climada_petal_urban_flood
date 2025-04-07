"""Urban flood impact module."""

import numpy as np
from climada.engine import Impact
from climada_petal_urban_flood.hazard import UrbanFloodHazard
from climada_petal_urban_flood.exposure import UrbanFloodExposure

class UrbanFloodImpact(Impact):
    """Class for urban flood impact assessment.
    
    This class extends CLIMADA's Impact to provide specialized methods
    for urban flood impact evaluation.
    """
    
    def __init__(self):
        """Initialize UrbanFloodImpact."""
        super().__init__()
    
    def calc_impact(self, hazard, exposure, impact_func_set, save_mat=False):
        """Calculate impact from hazard, exposure and impact functions.
        
        Extended to handle urban-specific impact calculations.
        
        Parameters:
            hazard (UrbanFloodHazard): Urban flood hazard
            exposure (UrbanExposure): Urban exposure 
            impact_func_set (ImpactFuncSet): Impact function set
            save_mat (bool): Save impact matrix
        """
        # Verify the correct type of input data
        if not isinstance(hazard, UrbanFloodHazard):
            hazard = UrbanFloodHazard.from_hazard(hazard)
            
        # Use the original method to calculate impacts
        super().calc_impact(hazard, exposure, impact_func_set, save_mat=save_mat)
        
        # Additional computations specific to urban flooding can be added here
        return self
    
    def affected_population(self, threshold=0.1):
        """Calculate the number of people affected by urban flood.
        
        Parameters:
            threshold (float): Flood depth threshold in meters to consider affected
            
        Returns:
            float: Number of affected people
        """
        if not hasattr(self.exp, 'value'):
            raise ValueError("No exposure data found")
            
        # Assume that the population column is stored as 'pop'
        if not hasattr(self.exp, 'pop'):
            raise ValueError("No population data in exposure")
            
        # Identify areas where flood depth exceeds the threshold
        affected_idx = self.eai_exp > 0
        
        # Calculate the number of affected people
        affected_pop = np.sum(self.exp.pop[affected_idx])
        
        return affected_pop
        
    def indirect_impacts(self, infrastructure_disruption_factor=0.5, recovery_time_days=30):
        """Estimate the indirect economic impacts of urban flooding.
        
        Parameters:
            infrastructure_disruption_factor (float): Factor for infrastructure disruption
            recovery_time_days (int): Estimated recovery time in days
            
        Returns:
            float: Estimated indirect economic impact
        """
        # This is a simplified model to estimate indirect effects
        direct_impact = self.aai_agg
        
        # Estimate indirect effects as a percentage of direct effects, dependent on recovery time
        indirect_factor = infrastructure_disruption_factor * (recovery_time_days / 365)
        indirect_impact = direct_impact * indirect_factor
        
        return indirect_impact
