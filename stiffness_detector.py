"""
Complete Stiffness Detection Module for Combustion Systems

This module provides robust stiffness analysis for Cantera gas objects using
expert-recommended numerical methods with adaptive handling for different
chemistry regimes (frozen, slow, active).

Author: Based on expert recommendations for combustion stiffness analysis
"""

import numpy as np
import warnings

class StiffnessDetector:
    """
    Complete stiffness detection module for combustion systems
    
    Features:
    - Expert-recommended Jacobian computation (two-species compensation, central differences)
    - Adaptive handling for frozen/slow/active chemistry regimes
    - Robust eigenvalue analysis with constraint filtering
    - Quality indicators and fallback methods
    - Comprehensive metrics output
    """
    
    def __init__(self, 
                 rtol=1e-6,
                 frozen_threshold=1e-6,
                 slow_threshold=1e-3,
                 min_mass_fraction=1e-10,
                 eigenvalue_threshold=1e-10,
                 bath_species='N2',
                 frozen_stiffness_method='interpolated'):
        """
        Initialize stiffness detector
        
        Parameters:
        -----------
        rtol : float
            Relative tolerance for Jacobian finite differences (default: 1e-6)
        frozen_threshold : float
            Max reaction rate below which chemistry is "frozen" [mol/m³/s] (default: 1e-6)
        slow_threshold : float
            Max reaction rate below which chemistry is "slow" [mol/m³/s] (default: 1e-3)
        min_mass_fraction : float
            Minimum mass fraction for species inclusion (default: 1e-10)
        eigenvalue_threshold : float
            Threshold for filtering small eigenvalues (default: 1e-10)
        bath_species : str
            Preferred bath species for compensation (default: 'N2')
        frozen_stiffness_method : str
            Method for frozen chemistry: 'nominal', 'temperature', 'concentration', 'interpolated'
        """
        self.rtol = rtol
        self.frozen_threshold = frozen_threshold
        self.slow_threshold = slow_threshold
        self.min_mass_fraction = min_mass_fraction
        self.eigenvalue_threshold = eigenvalue_threshold
        self.bath_species = bath_species
        self.frozen_stiffness_method = frozen_stiffness_method
        
        # Internal state for diagnostics
        self._last_computation_info = {}
    
    def analyze_stiffness(self, gas):
        """
        Complete stiffness analysis for a given gas state
        
        Parameters:
        -----------
        gas : cantera.Solution
            Cantera gas object at current state
            
        Returns:
        --------
        dict : Complete stiffness analysis containing:
            - jacobian : numpy.ndarray or None
            - stiffness_ratio : float
            - chemistry_state : str ('frozen', 'slow', 'moderate', 'fast')
            - quality : str ('computed_full', 'computed_reduced', 'estimated', 'fallback')
            - eigenvalues : numpy.ndarray or None
            - time_scales : dict with 'fastest' and 'slowest' [seconds]
            - reaction_metrics : dict with reaction rate info
            - species_info : dict with species distribution
            - solver_recommendations : dict with suggested parameters
            - diagnostic_info : dict with computation details
        """
        try:
            # Step 1: Analyze current state
            state_info = self._analyze_gas_state(gas)
            
            # Step 2: Determine chemistry regime
            chemistry_state = self._classify_chemistry_state(state_info['max_reaction_rate'])
            
            # Step 3: Compute stiffness based on regime
            if chemistry_state == 'frozen':
                result = self._handle_frozen_chemistry(gas, state_info)
            elif chemistry_state == 'slow':
                result = self._handle_slow_chemistry(gas, state_info)
            else:
                result = self._handle_active_chemistry(gas, state_info, chemistry_state)
            
            # Step 4: Add comprehensive metadata
            result.update({
                'chemistry_state': chemistry_state,
                'reaction_metrics': state_info,
                'species_info': self._get_species_distribution(gas),
                'solver_recommendations': self._generate_solver_recommendations(result),
                'diagnostic_info': self._last_computation_info.copy()
            })
            
            return result
            
        except Exception as e:
            # Emergency fallback
            warnings.warn(f"Stiffness analysis failed: {e}. Using fallback estimate.")
            return self._emergency_fallback(gas, str(e))
    
    def _analyze_gas_state(self, gas):
        """Analyze current gas state for reaction activity"""
        wdot = gas.net_production_rates
        Y = gas.Y
        
        return {
            'temperature': gas.T,
            'pressure': gas.P,
            'density': gas.density,
            'max_reaction_rate': np.max(np.abs(wdot)),
            'mean_reaction_rate': np.mean(np.abs(wdot)),
            'active_species_count': np.sum(np.abs(wdot) > self.frozen_threshold),
            'production_rates': wdot.copy(),
            'mass_fractions': Y.copy()
        }
    
    def _classify_chemistry_state(self, max_rate):
        """Classify chemistry state based on reaction rates"""
        if max_rate < self.frozen_threshold:
            return 'frozen'
        elif max_rate < self.slow_threshold:
            return 'slow'
        elif max_rate < 1.0:
            return 'moderate'
        else:
            return 'fast'
    
    def _handle_frozen_chemistry(self, gas, state_info):
        """Handle frozen chemistry case with multiple estimation methods"""
        max_rate = state_info['max_reaction_rate']
        
        # Different estimation methods
        stiffness_estimates = {
            'nominal': 1.0,
            'temperature': max(1.0, np.exp(-24000 / gas.T)),
            'concentration': self._concentration_based_stiffness(gas),
            'interpolated': self._interpolated_stiffness(max_rate, -10, -6, 0, 3)
        }
        
        chosen_stiffness = stiffness_estimates[self.frozen_stiffness_method]
        
        self._last_computation_info = {
            'method': f'frozen_{self.frozen_stiffness_method}',
            'all_estimates': stiffness_estimates,
            'chosen_estimate': chosen_stiffness,
            'max_rate': max_rate
        }
        
        return {
            'jacobian': None,
            'stiffness_ratio': chosen_stiffness,
            'quality': 'estimated',
            'eigenvalues': None,
            'time_scales': {'fastest': np.inf, 'slowest': np.inf},
            'n_species_analyzed': 0,
            'method': f'frozen_{self.frozen_stiffness_method}'
        }
    
    def _handle_slow_chemistry(self, gas, state_info):
        """Handle slow chemistry with reduced system"""
        try:
            # Use reduced system for stability
            Y = gas.Y
            significant_mask = Y > max(self.min_mass_fraction, 1e-8)
            n_significant = np.sum(significant_mask)
            
            if n_significant >= 2:
                J_reduced = self._compute_reduced_jacobian(gas, significant_mask)
                metrics = self._analyze_eigenvalues(J_reduced, robust=True)
                
                # Cap stiffness for slow chemistry to avoid numerical artifacts
                stiffness_capped = min(metrics['stiffness_ratio'], 1e6)
                
                self._last_computation_info = {
                    'method': 'reduced_jacobian',
                    'n_species_reduced': n_significant,
                    'original_stiffness': metrics['stiffness_ratio'],
                    'capped_stiffness': stiffness_capped
                }
                
                return {
                    'jacobian': J_reduced,
                    'stiffness_ratio': stiffness_capped,
                    'quality': 'computed_reduced',
                    'eigenvalues': metrics['eigenvalues'],
                    'time_scales': {'fastest': metrics['fastest_time_scale'], 
                                  'slowest': metrics['slowest_time_scale']},
                    'n_species_analyzed': n_significant,
                    'method': 'reduced_jacobian'
                }
            else:
                # Fall back to frozen handling
                return self._handle_frozen_chemistry(gas, state_info)
                
        except Exception as e:
            # Fallback to interpolation
            max_rate = state_info['max_reaction_rate']
            stiffness = self._interpolated_stiffness(max_rate, -6, -3, 1, 4)
            
            self._last_computation_info = {
                'method': 'slow_chemistry_fallback',
                'error': str(e),
                'interpolated_stiffness': stiffness
            }
            
            return {
                'jacobian': None,
                'stiffness_ratio': stiffness,
                'quality': 'fallback_estimate',
                'eigenvalues': None,
                'time_scales': {'fastest': 1/stiffness if stiffness > 0 else np.inf, 
                              'slowest': 1.0},
                'n_species_analyzed': 0,
                'method': 'interpolated_fallback'
            }
    
    def _handle_active_chemistry(self, gas, state_info, chemistry_state):
        """Handle active chemistry with full or reduced computation"""
        try:
            # Try full Jacobian first
            J_full = self._compute_jacobian_expert(gas)
            metrics_full = self._analyze_eigenvalues(J_full, robust=True)
            
            # Check if result is reasonable
            if metrics_full['stiffness_ratio'] < 1e12 and not np.isnan(metrics_full['stiffness_ratio']):
                # Full computation successful
                self._last_computation_info = {
                    'method': 'full_jacobian_expert',
                    'n_species_full': gas.n_species,
                    'stiffness_ratio': metrics_full['stiffness_ratio']
                }
                
                return {
                    'jacobian': J_full,
                    'stiffness_ratio': metrics_full['stiffness_ratio'],
                    'quality': 'computed_full',
                    'eigenvalues': metrics_full['eigenvalues'],
                    'time_scales': {'fastest': metrics_full['fastest_time_scale'],
                                  'slowest': metrics_full['slowest_time_scale']},
                    'n_species_analyzed': gas.n_species,
                    'method': 'expert_jacobian_full'
                }
            else:
                # Full system too stiff, use reduced
                Y = gas.Y
                active_mask = Y > max(self.min_mass_fraction, 1e-10)
                J_reduced = self._compute_reduced_jacobian(gas, active_mask)
                metrics_reduced = self._analyze_eigenvalues(J_reduced, robust=True)
                
                self._last_computation_info = {
                    'method': 'reduced_after_full_failed',
                    'full_stiffness': metrics_full['stiffness_ratio'],
                    'reduced_stiffness': metrics_reduced['stiffness_ratio'],
                    'n_species_reduced': np.sum(active_mask)
                }
                
                return {
                    'jacobian': J_reduced,
                    'stiffness_ratio': metrics_reduced['stiffness_ratio'],
                    'quality': 'computed_reduced',
                    'eigenvalues': metrics_reduced['eigenvalues'],
                    'time_scales': {'fastest': metrics_reduced['fastest_time_scale'],
                                  'slowest': metrics_reduced['slowest_time_scale']},
                    'n_species_analyzed': np.sum(active_mask),
                    'method': 'expert_jacobian_reduced'
                }
                
        except Exception as e:
            # Final fallback to rate-based estimation
            max_rate = state_info['max_reaction_rate']
            stiffness = self._interpolated_stiffness(max_rate, -3, 2, 3, 6)
            
            self._last_computation_info = {
                'method': 'active_chemistry_fallback',
                'error': str(e),
                'interpolated_stiffness': stiffness
            }
            
            return {
                'jacobian': None,
                'stiffness_ratio': stiffness,
                'quality': 'fallback_estimate', 
                'eigenvalues': None,
                'time_scales': {'fastest': 1/stiffness if stiffness > 0 else np.inf,
                              'slowest': 1.0},
                'n_species_analyzed': 0,
                'method': 'interpolated_fallback'
            }
    
    def _compute_jacobian_expert(self, gas):
        """
        Expert-recommended Jacobian computation with all fixes:
        - Two-species compensation
        - Central differences  
        - Consistent density calculation
        - Smart step sizing
        """
        n_species = gas.n_species
        species_names = gas.species_names
        
        # Find bath species
        try:
            j_bath = species_names.index(self.bath_species)
        except ValueError:
            j_bath = np.argmax(gas.Y)  # Most abundant as fallback
        
        # Store baseline state
        Y0 = gas.Y.copy()
        T0, P0 = gas.T, gas.P
        MW = gas.molecular_weights
        
        # Initialize Jacobian
        J = np.zeros((n_species, n_species))
        
        for j in range(n_species):
            # Choose compensation species
            comp_idx = j_bath if j != j_bath else np.argmax([gas.Y[k] if k != j_bath else -1 for k in range(n_species)])
            
            # Smart step sizing
            dY = max(1e-8, self.rtol * max(1.0, abs(Y0[j])))
            dY = min(dY, Y0[comp_idx] - 1e-12)  # Ensure compensation species stays positive
            
            if dY <= 1e-15:
                continue
            
            try:
                # Central difference computation
                rates_plus = rates_minus = None
                
                for direction in [+1, -1]:
                    # Two-species compensation
                    Y_pert = Y0.copy()
                    Y_pert[j] += direction * dY
                    Y_pert[comp_idx] -= direction * dY
                    
                    # Safety checks
                    Y_pert = np.clip(Y_pert, 0.0, 1.0)
                    sum_Y = np.sum(Y_pert)
                    if abs(sum_Y - 1.0) > 1e-12:
                        Y_pert /= sum_Y  # Minimal renormalization
                    
                    # Set state and compute rates with consistent density
                    gas.TPY = T0, P0, Y_pert
                    wdot_pert = gas.net_production_rates
                    rho_pert = gas.density  # Consistent density
                    
                    # Convert to mass fraction rates
                    Ydot_pert = (wdot_pert * MW) / rho_pert
                    
                    if direction > 0:
                        rates_plus = Ydot_pert
                    else:
                        rates_minus = Ydot_pert
                
                # Central difference
                if rates_plus is not None and rates_minus is not None:
                    J[:, j] = (rates_plus - rates_minus) / (2.0 * dY)
                    
            except Exception:
                continue  # Skip problematic species
        
        # Restore original state
        gas.TPY = T0, P0, Y0
        
        # Clean up Jacobian
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
        
        return J
    
    def _compute_reduced_jacobian(self, gas, species_mask):
        """Compute Jacobian for subset of species"""
        significant_indices = np.where(species_mask)[0]
        n_sig = len(significant_indices)
        
        if n_sig < 2:
            raise ValueError("Need at least 2 significant species")
        
        # Store state
        Y0 = gas.Y.copy()
        T0, P0 = gas.T, gas.P
        MW = gas.molecular_weights[significant_indices]
        
        # Find bath among significant species
        bath_idx_local = np.argmax(Y0[significant_indices])
        bath_idx_global = significant_indices[bath_idx_local]
        
        J = np.zeros((n_sig, n_sig))
        
        for j_local, j_global in enumerate(significant_indices):
            # Compensation logic
            comp_idx = bath_idx_global if j_global != bath_idx_global else significant_indices[0]
            
            # Smart step
            dY = max(1e-8, self.rtol * Y0[j_global])
            dY = min(dY, Y0[comp_idx] * 0.9)
            
            if dY < 1e-12:
                continue
            
            try:
                # Central difference
                for direction in [+1, -1]:
                    Y_pert = Y0.copy()
                    Y_pert[j_global] += direction * dY
                    Y_pert[comp_idx] -= direction * dY
                    Y_pert = np.clip(Y_pert, 0, 1)
                    Y_pert /= np.sum(Y_pert)
                    
                    gas.TPY = T0, P0, Y_pert
                    wdot_pert = gas.net_production_rates[significant_indices]
                    rho_pert = gas.density
                    
                    if direction > 0:
                        rates_plus = (wdot_pert * MW) / rho_pert
                    else:
                        rates_minus = (wdot_pert * MW) / rho_pert
                
                J[:, j_local] = (rates_plus - rates_minus) / (2.0 * dY)
                
            except Exception:
                continue
        
        # Restore state
        gas.TPY = T0, P0, Y0
        return J
    
    def _analyze_eigenvalues(self, J, robust=True):
        """
        Robust eigenvalue analysis with constraint filtering
        
        Parameters:
        -----------
        J : numpy.ndarray
            Jacobian matrix
        robust : bool
            Use robust percentile-based analysis
            
        Returns:
        --------
        dict : Eigenvalue analysis results
        """
        try:
            eigenvals = np.linalg.eigvals(J)
        except np.linalg.LinAlgError:
            return {
                'stiffness_ratio': 1e3,
                'eigenvalues': np.array([]),
                'fastest_time_scale': np.inf,
                'slowest_time_scale': np.inf,
                'n_stable_modes': 0,
                'analysis_failed': True
            }
        
        # Filter eigenvalues
        real_parts = np.real(eigenvals)
        magnitudes = np.abs(eigenvals)
        
        # Focus on stable chemistry modes (negative real part, significant magnitude)
        stable_mask = (magnitudes > self.eigenvalue_threshold) & (real_parts < -self.eigenvalue_threshold)
        stable_mags = magnitudes[stable_mask]
        
        if len(stable_mags) < 2:
            return {
                'stiffness_ratio': 1.0,
                'eigenvalues': eigenvals,
                'fastest_time_scale': np.inf,
                'slowest_time_scale': np.inf,
                'n_stable_modes': len(stable_mags),
                'insufficient_modes': True
            }
        
        # Compute stiffness ratio
        if robust and len(stable_mags) >= 10:
            # Percentile-based (expert recommendation)
            max_mag = np.percentile(stable_mags, 95)
            min_mag = np.percentile(stable_mags, 5)
            ratio_type = "95th/5th percentile"
        else:
            # Standard max/min
            max_mag = np.max(stable_mags)
            min_mag = np.min(stable_mags)
            ratio_type = "max/min"
        
        stiffness_ratio = max_mag / min_mag if min_mag > 0 else 1e6
        
        return {
            'stiffness_ratio': stiffness_ratio,
            'eigenvalues': eigenvals,
            'fastest_time_scale': 1.0 / max_mag if max_mag > 0 else np.inf,
            'slowest_time_scale': 1.0 / min_mag if min_mag > 0 else np.inf,
            'n_stable_modes': len(stable_mags),
            'ratio_type': ratio_type,
            'max_eigenvalue_magnitude': max_mag,
            'min_eigenvalue_magnitude': min_mag
        }
    
    def _concentration_based_stiffness(self, gas):
        """Estimate stiffness based on concentration span"""
        Y = gas.Y
        Y_nonzero = Y[Y > 1e-16]
        if len(Y_nonzero) > 1:
            conc_span = np.max(Y_nonzero) / np.min(Y_nonzero)
            return min(conc_span, 1e3)  # Cap at reasonable value
        return 1.0
    
    def _interpolated_stiffness(self, rate, log_rate_min, log_rate_max, log_stiff_min, log_stiff_max):
        """Interpolate stiffness based on reaction rate"""
        if rate <= 0:
            return 10**log_stiff_min
        
        log_rate = np.log10(rate)
        log_stiffness = np.interp(log_rate, [log_rate_min, log_rate_max], [log_stiff_min, log_stiff_max])
        return 10**log_stiffness
    
    def _get_species_distribution(self, gas):
        """Get species distribution statistics"""
        Y = gas.Y
        return {
            'n_major': np.sum(Y > 1e-3),
            'n_minor': np.sum((Y > 1e-8) & (Y <= 1e-3)),
            'n_trace': np.sum((Y > 1e-15) & (Y <= 1e-8)),
            'n_negligible': np.sum(Y <= 1e-15),
            'max_mass_fraction': np.max(Y),
            'min_nonzero_mass_fraction': np.min(Y[Y > 0]) if np.any(Y > 0) else 0,
            'concentration_span': np.max(Y) / np.min(Y[Y > 1e-16]) if np.any(Y > 1e-16) else 1
        }
    
    def _generate_solver_recommendations(self, stiffness_result):
        """Generate solver recommendations based on stiffness"""
        stiffness_ratio = stiffness_result['stiffness_ratio']
        fastest_dt = stiffness_result['time_scales']['fastest']
        
        if stiffness_ratio > 1e6:
            solver_type = "implicit_high_order"
            recommended_solvers = ["CVODE_BDF", "RADAU5", "DASPK"]
            rtol_suggestion = 1e-9
            atol_suggestion = 1e-12
        elif stiffness_ratio > 1e3:
            solver_type = "implicit_medium_order"
            recommended_solvers = ["CVODE_BDF", "LSODA", "DOPRI5"]
            rtol_suggestion = 1e-8
            atol_suggestion = 1e-11
        elif stiffness_ratio > 1e2:
            solver_type = "adaptive"
            recommended_solvers = ["LSODA", "DOPRI5", "RK45"]
            rtol_suggestion = 1e-7
            atol_suggestion = 1e-10
        else:
            solver_type = "explicit"
            recommended_solvers = ["RK45", "DOPRI5", "RK23"]
            rtol_suggestion = 1e-6
            atol_suggestion = 1e-9
        
        # Time step suggestions
        if fastest_dt < np.inf and fastest_dt > 0:
            max_dt_suggestion = fastest_dt / 10  # Safety factor
            initial_dt_suggestion = fastest_dt / 100
        else:
            max_dt_suggestion = 1e-6
            initial_dt_suggestion = 1e-8
        
        return {
            'solver_type': solver_type,
            'recommended_solvers': recommended_solvers,
            'tolerance_suggestions': {
                'rtol': rtol_suggestion,
                'atol': atol_suggestion
            },
            'time_step_suggestions': {
                'max_dt': max_dt_suggestion,
                'initial_dt': initial_dt_suggestion
            },
            'stiffness_classification': self._classify_stiffness(stiffness_ratio)
        }
    
    def _classify_stiffness(self, ratio):
        """Classify stiffness level"""
        if ratio > 1e6:
            return "very_stiff"
        elif ratio > 1e3:
            return "stiff"
        elif ratio > 1e2:
            return "moderately_stiff"
        else:
            return "not_stiff"
    
    def _emergency_fallback(self, gas, error_msg):
        """Emergency fallback when all methods fail"""
        return {
            'jacobian': None,
            'stiffness_ratio': 1e3,  # Conservative estimate
            'chemistry_state': 'unknown',
            'quality': 'emergency_fallback',
            'eigenvalues': None,
            'time_scales': {'fastest': np.inf, 'slowest': np.inf},
            'reaction_metrics': {'error': error_msg},
            'species_info': {'error': error_msg},
            'solver_recommendations': {
                'solver_type': 'implicit_medium_order',
                'recommended_solvers': ['CVODE_BDF'],
                'tolerance_suggestions': {'rtol': 1e-8, 'atol': 1e-11},
                'time_step_suggestions': {'max_dt': 1e-6, 'initial_dt': 1e-8},
                'stiffness_classification': 'stiff'
            },
            'diagnostic_info': {
                'method': 'emergency_fallback',
                'error': error_msg
            },
            'n_species_analyzed': 0,
            'method': 'emergency_fallback'
        }


# Convenience function for quick analysis
def analyze_stiffness(gas, **kwargs):
    """
    Convenience function for quick stiffness analysis
    
    Parameters:
    -----------
    gas : cantera.Solution
        Cantera gas object
    **kwargs : dict
        Additional arguments passed to StiffnessDetector
        
    Returns:
    --------
    dict : Complete stiffness analysis results
    """
    detector = StiffnessDetector(**kwargs)
    return detector.analyze_stiffness(gas)


# Example usage and demonstration
if __name__ == "__main__":
    import cantera as ct
    
    print("🔧 COMPLETE STIFFNESS DETECTION MODULE DEMO")
    print("="*60)
    
    try:
        # Create test case
        gas = ct.Solution('gri30.yaml')
        gas.TPX = 1000, 101325, 'CH4:1, O2:2, N2:7.52'
        
        # Method 1: Using the class
        print("\n1. Using StiffnessDetector class:")
        detector = StiffnessDetector(frozen_stiffness_method='interpolated')
        result = detector.analyze_stiffness(gas)
        
        print(f"   Stiffness ratio: {result['stiffness_ratio']:.2e}")
        print(f"   Chemistry state: {result['chemistry_state']}")
        print(f"   Quality: {result['quality']}")
        print(f"   Method: {result['method']}")
        print(f"   Solver recommendation: {result['solver_recommendations']['solver_type']}")
        
        # Method 2: Using convenience function
        print("\n2. Using convenience function:")
        result_quick = analyze_stiffness(gas, frozen_stiffness_method='temperature')
        print(f"   Quick analysis stiffness: {result_quick['stiffness_ratio']:.2e}")
        
        # Method 3: Test with active chemistry
        print("\n3. Testing with higher temperature (active chemistry):")
        gas.TPX = 1500, 101325, 'CH4:1, O2:2, N2:7.52'
        reactor = ct.IdealGasReactor(gas)
        net = ct.ReactorNet([reactor])
        net.advance(1e-6)  # Brief reaction
        
        result_active = detector.analyze_stiffness(gas)
        print(f"   Active chemistry stiffness: {result_active['stiffness_ratio']:.2e}")
        print(f"   Time scale range: {result_active['time_scales']['fastest']:.2e} to {result_active['time_scales']['slowest']:.2e} s")
        print(f"   Species analyzed: {result_active['n_species_analyzed']}")
        
        # Show comprehensive output
        print(f"\n4. Complete analysis output structure:")
        print(f"   Available keys: {list(result.keys())}")
        print(f"   Reaction metrics: {list(result['reaction_metrics'].keys())}")
        print(f"   Species info: {result['species_info']}")
        print(f"   Diagnostic info: {result['diagnostic_info']}")
        
        print(f"\n✅ Module successfully demonstrated!")
        print(f"   - Handles frozen, slow, and active chemistry")
        print(f"   - Provides comprehensive analysis and recommendations")
        print(f"   - Includes quality indicators and fallback methods")
        print(f"   - Ready for integration into combustion simulations")
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()