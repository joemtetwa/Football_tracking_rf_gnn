"""
Football Ball Prediction Validator
---------------------------------
This script provides functions to validate predicted ball positions
by checking for anomalies, unrealistic movements, and out-of-bounds positions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


class BallPredictionValidator:
    """
    Class for validating predicted ball positions from football tracking data.
    Performs various checks to identify potential issues in the predictions.
    """
    
    def __init__(self, predictions_path, output_dir='validation_results'):
        """
        Initialize the validator with path to predictions and output directory.
        
        Args:
            predictions_path (str): Path to the predicted ball positions CSV
            output_dir (str): Directory to save validation results
        """
        self.predictions_path = predictions_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load predictions
        self.predictions = pd.read_csv(predictions_path)
        print(f"Loaded {len(self.predictions)} predicted ball positions")
        
        # Football pitch dimensions (estimated ranges in coordinate system)
        # These should be adjusted based on your specific dataset
        self.pitch_length = 10000  # Length in x direction
        self.pitch_width = 6800    # Width in y direction
        self.min_expected_x = -5000
        self.max_expected_x = 5000
        self.min_expected_y = -3400
        self.max_expected_y = 3400
        
        # Validation parameters
        self.max_realistic_speed = 3000     # Maximum realistic ball speed (units/second)
        self.large_jump_threshold = 500     # Threshold for unrealistic position jumps
        self.time_unit_factor = 10          # Convert time units to seconds (assuming deciseconds)
    
    def run_all_validations(self, generate_report=True):
        """
        Run all validation checks and optionally generate a comprehensive report.
        
        Args:
            generate_report (bool): Whether to generate a validation report
            
        Returns:
            dict: Dictionary containing all validation results
        """
        print("Running all validation checks...")
        
        # Basic data validation
        basic_checks = self.validate_basic_data()
        
        # Position range validation
        position_checks = self.validate_position_ranges()
        
        # Movement validation
        movement_checks = self.validate_ball_movement()
        
        # Period-specific validation
        period_checks = self.validate_by_period()
        
        # Combine all results
        all_results = {
            'basic_checks': basic_checks,
            'position_checks': position_checks,
            'movement_checks': movement_checks,
            'period_checks': period_checks,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Generate report if requested
        if generate_report:
            self.generate_validation_report(all_results)
        
        return all_results
    
    def validate_basic_data(self):
        """
        Perform basic validation checks on the prediction data.
        
        Returns:
            dict: Results of basic validation checks
        """
        print("Validating basic data properties...")
        
        # Check for required columns
        required_columns = ['MatchId', 'IdPeriod', 'Time', 'ball_x', 'ball_y']
        missing_columns = [col for col in required_columns if col not in self.predictions.columns]
        
        # Check for missing values
        missing_values = {
            'ball_x': self.predictions['ball_x'].isna().sum(),
            'ball_y': self.predictions['ball_y'].isna().sum(),
            'total_rows_with_missing': self.predictions[['ball_x', 'ball_y']].isna().any(axis=1).sum()
        }
        
        # Check for duplicate timestamps
        duplicates = self.predictions.duplicated(subset=['IdPeriod', 'Time']).sum()
        
        # Check time sequence
        time_issues = []
        for period in self.predictions['IdPeriod'].unique():
            period_data = self.predictions[self.predictions['IdPeriod'] == period]
            time_diffs = period_data['Time'].diff()
            if (time_diffs.dropna() <= 0).any():
                time_issues.append({
                    'period': period,
                    'issues': len(time_diffs[time_diffs <= 0])
                })
        
        return {
            'missing_columns': missing_columns,
            'missing_values': missing_values,
            'duplicate_timestamps': duplicates,
            'time_sequence_issues': time_issues,
            'total_predictions': len(self.predictions),
            'periods': self.predictions['IdPeriod'].unique().tolist()
        }
    
    def validate_position_ranges(self):
        """
        Validate ball position ranges to identify out-of-bounds predictions.
        
        Returns:
            dict: Results of position range validation
        """
        print("Validating ball position ranges...")
        
        # Get actual ranges
        x_min, x_max = self.predictions['ball_x'].min(), self.predictions['ball_x'].max()
        y_min, y_max = self.predictions['ball_y'].min(), self.predictions['ball_y'].max()
        
        # Check for positions outside expected pitch
        out_of_bounds_x = ((self.predictions['ball_x'] < self.min_expected_x) | 
                          (self.predictions['ball_x'] > self.max_expected_x)).sum()
        out_of_bounds_y = ((self.predictions['ball_y'] < self.min_expected_y) | 
                          (self.predictions['ball_y'] > self.max_expected_y)).sum()
        
        # Calculate coverage of pitch
        x_range = x_max - x_min
        y_range = y_max - y_min
        expected_x_range = self.max_expected_x - self.min_expected_x
        expected_y_range = self.max_expected_y - self.min_expected_y
        
        # Create pitch coverage stats
        coverage_stats = {
            'x_coverage_percent': (x_range / expected_x_range) * 100,
            'y_coverage_percent': (y_range / expected_y_range) * 100,
            'area_coverage_percent': (x_range * y_range) / (expected_x_range * expected_y_range) * 100
        }
        
        # Create position distribution statistics
        x_quartiles = self.predictions['ball_x'].quantile([0.25, 0.5, 0.75]).to_dict()
        y_quartiles = self.predictions['ball_y'].quantile([0.25, 0.5, 0.75]).to_dict()
        
        return {
            'actual_ranges': {
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max
            },
            'expected_ranges': {
                'x_min': self.min_expected_x,
                'x_max': self.max_expected_x,
                'y_min': self.min_expected_y,
                'y_max': self.max_expected_y
            },
            'out_of_bounds': {
                'x_count': out_of_bounds_x,
                'x_percent': (out_of_bounds_x / len(self.predictions)) * 100,
                'y_count': out_of_bounds_y,
                'y_percent': (out_of_bounds_y / len(self.predictions)) * 100
            },
            'coverage_stats': coverage_stats,
            'position_distribution': {
                'x_quartiles': x_quartiles,
                'y_quartiles': y_quartiles
            }
        }
    
    def validate_ball_movement(self):
        """
        Validate ball movement patterns to identify unrealistic movements.
        
        Returns:
            dict: Results of ball movement validation
        """
        print("Validating ball movement patterns...")
        
        # Calculate position changes and speeds
        data = self.predictions.copy()
        data['x_diff'] = data['ball_x'].diff()
        data['y_diff'] = data['ball_y'].diff()
        
        # Handle period transitions (don't calculate diffs across periods)
        period_starts = data['IdPeriod'].ne(data['IdPeriod'].shift()).cumsum()
        data.loc[period_starts != period_starts.shift(), ['x_diff', 'y_diff']] = np.nan
        
        # Calculate distance moved and speed
        data['distance_moved'] = np.sqrt(data['x_diff']**2 + data['y_diff']**2)
        
        # Calculate speed in units per second (assuming Time is in deciseconds)
        time_diffs = data['Time'].diff() / self.time_unit_factor
        data['time_diff'] = time_diffs
        data['ball_speed'] = data['distance_moved'] / time_diffs
        
        # Replace inf values (from division by zero) with NaN
        data['ball_speed'].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Identify large position jumps
        large_jumps = data[data['distance_moved'] > self.large_jump_threshold].copy()
        
        # Identify unrealistic speeds
        unrealistic_speeds = data[data['ball_speed'] > self.max_realistic_speed].copy()
        
        # Calculate speed statistics
        speed_stats = {
            'min': data['ball_speed'].min(),
            'max': data['ball_speed'].max(),
            'mean': data['ball_speed'].mean(),
            'median': data['ball_speed'].median(),
            'std': data['ball_speed'].std(),
            'percentiles': data['ball_speed'].quantile([0.25, 0.75, 0.9, 0.95, 0.99]).to_dict()
        }
        
        # Calculate acceleration
        data['speed_diff'] = data['ball_speed'].diff()
        data['acceleration'] = data['speed_diff'] / time_diffs
        data['acceleration'].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Identify sudden accelerations/decelerations
        high_acceleration_threshold = 1000  # Adjust based on your data
        high_accelerations = data[data['acceleration'].abs() > high_acceleration_threshold].copy()
        
        # Save the processed data for potential further analysis
        data.to_csv(os.path.join(self.output_dir, 'ball_movement_data.csv'), index=False)
        
        return {
            'speed_stats': speed_stats,
            'large_jumps': {
                'count': len(large_jumps),
                'percent': (len(large_jumps) / len(data)) * 100,
                'max_jump': data['distance_moved'].max(),
                'examples': large_jumps.head(5).to_dict('records') if len(large_jumps) > 0 else []
            },
            'unrealistic_speeds': {
                'count': len(unrealistic_speeds),
                'percent': (len(unrealistic_speeds) / len(data)) * 100,
                'max_speed': data['ball_speed'].max(),
                'examples': unrealistic_speeds.head(5).to_dict('records') if len(unrealistic_speeds) > 0 else []
            },
            'acceleration_issues': {
                'count': len(high_accelerations),
                'percent': (len(high_accelerations) / len(data)) * 100,
                'max_acceleration': data['acceleration'].abs().max(),
                'examples': high_accelerations.head(5).to_dict('records') if len(high_accelerations) > 0 else []
            }
        }
    
    def validate_by_period(self):
        """
        Validate ball positions and movements by match period.
        
        Returns:
            dict: Results of period-specific validation
        """
        print("Validating ball positions by period...")
        
        # Group data by period
        period_groups = self.predictions.groupby('IdPeriod')
        
        period_results = {}
        
        for period, group in period_groups:
            # Basic statistics for each period
            period_results[period] = {
                'count': len(group),
                'position_ranges': {
                    'x_min': group['ball_x'].min(),
                    'x_max': group['ball_x'].max(),
                    'x_mean': group['ball_x'].mean(),
                    'y_min': group['ball_y'].min(),
                    'y_max': group['ball_y'].max(),
                    'y_mean': group['ball_y'].mean()
                }
            }
            
            # Calculate movement stats if we have a processed data file
            movement_file = os.path.join(self.output_dir, 'ball_movement_data.csv')
            if os.path.exists(movement_file):
                movement_data = pd.read_csv(movement_file)
                period_movement = movement_data[movement_data['IdPeriod'] == period]
                
                if len(period_movement) > 0:
                    period_results[period]['movement_stats'] = {
                        'mean_speed': period_movement['ball_speed'].mean(),
                        'max_speed': period_movement['ball_speed'].max(),
                        'large_jumps': (period_movement['distance_moved'] > self.large_jump_threshold).sum(),
                        'unrealistic_speeds': (period_movement['ball_speed'] > self.max_realistic_speed).sum()
                    }
        
        # Compare periods to identify inconsistencies
        period_ids = list(period_results.keys())
        period_comparisons = {}
        
        if len(period_ids) > 1:
            for i in range(len(period_ids)):
                for j in range(i+1, len(period_ids)):
                    p1 = period_ids[i]
                    p2 = period_ids[j]
                    
                    p1_stats = period_results[p1]
                    p2_stats = period_results[p2]
                    
                    # Compare x position means
                    x_mean_diff = abs(p1_stats['position_ranges']['x_mean'] - 
                                     p2_stats['position_ranges']['x_mean'])
                    
                    # Compare y position means
                    y_mean_diff = abs(p1_stats['position_ranges']['y_mean'] - 
                                     p2_stats['position_ranges']['y_mean'])
                    
                    # Compare movement stats if available
                    speed_diff = None
                    if ('movement_stats' in p1_stats and 'movement_stats' in p2_stats):
                        speed_diff = abs(p1_stats['movement_stats']['mean_speed'] - 
                                        p2_stats['movement_stats']['mean_speed'])
                    
                    period_comparisons[f"{p1}_vs_{p2}"] = {
                        'x_mean_diff': x_mean_diff,
                        'y_mean_diff': y_mean_diff,
                        'speed_diff': speed_diff
                    }
        
        return {
            'period_stats': period_results,
            'period_comparisons': period_comparisons
        }
    
    def generate_validation_report(self, validation_results):
        """
        Generate a comprehensive validation report based on all validation results.
        
        Args:
            validation_results (dict): Combined results from all validation checks
        """
        print("Generating validation report...")
        
        # Create a timestamp for the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"validation_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("=================================================\n")
            f.write("FOOTBALL BALL PREDICTION VALIDATION REPORT\n")
            f.write("=================================================\n")
            f.write(f"Generated: {validation_results['timestamp']}\n")
            f.write(f"Input file: {self.predictions_path}\n")
            f.write(f"Total predictions: {validation_results['basic_checks']['total_predictions']}\n\n")
            
            # Basic Data Validation Section
            f.write("1. BASIC DATA VALIDATION\n")
            f.write("------------------------\n")
            basic = validation_results['basic_checks']
            
            f.write(f"Missing columns: {basic['missing_columns'] or 'None'}\n")
            f.write(f"Missing values: {basic['missing_values']['total_rows_with_missing']} rows "
                   f"({(basic['missing_values']['total_rows_with_missing']/basic['total_predictions'])*100:.2f}%)\n")
            f.write(f"Duplicate timestamps: {basic['duplicate_timestamps']}\n")
            
            if basic['time_sequence_issues']:
                f.write("Time sequence issues detected:\n")
                for issue in basic['time_sequence_issues']:
                    f.write(f"  - Period {issue['period']}: {issue['issues']} issues\n")
            else:
                f.write("Time sequence: OK\n")
            
            f.write(f"Periods detected: {basic['periods']}\n\n")
            
            # Position Range Validation Section
            f.write("2. POSITION RANGE VALIDATION\n")
            f.write("----------------------------\n")
            pos = validation_results['position_checks']
            
            f.write("Actual position ranges:\n")
            f.write(f"  X: {pos['actual_ranges']['x_min']:.2f} to {pos['actual_ranges']['x_max']:.2f}\n")
            f.write(f"  Y: {pos['actual_ranges']['y_min']:.2f} to {pos['actual_ranges']['y_max']:.2f}\n\n")
            
            f.write("Expected position ranges:\n")
            f.write(f"  X: {pos['expected_ranges']['x_min']} to {pos['expected_ranges']['x_max']}\n")
            f.write(f"  Y: {pos['expected_ranges']['y_min']} to {pos['expected_ranges']['y_max']}\n\n")
            
            f.write("Out-of-bounds positions:\n")
            f.write(f"  X: {pos['out_of_bounds']['x_count']} points "
                   f"({pos['out_of_bounds']['x_percent']:.2f}%)\n")
            f.write(f"  Y: {pos['out_of_bounds']['y_count']} points "
                   f"({pos['out_of_bounds']['y_percent']:.2f}%)\n\n")
            
            f.write("Pitch coverage:\n")
            f.write(f"  X-range coverage: {pos['coverage_stats']['x_coverage_percent']:.2f}%\n")
            f.write(f"  Y-range coverage: {pos['coverage_stats']['y_coverage_percent']:.2f}%\n")
            f.write(f"  Area coverage: {pos['coverage_stats']['area_coverage_percent']:.2f}%\n\n")
            
            # Movement Validation Section
            f.write("3. BALL MOVEMENT VALIDATION\n")
            f.write("---------------------------\n")
            mov = validation_results['movement_checks']
            
            f.write("Ball speed statistics (units/second):\n")
            f.write(f"  Min: {mov['speed_stats']['min']:.2f}\n")
            f.write(f"  Max: {mov['speed_stats']['max']:.2f}\n")
            f.write(f"  Mean: {mov['speed_stats']['mean']:.2f}\n")
            f.write(f"  Median: {mov['speed_stats']['median']:.2f}\n")
            f.write(f"  Std Dev: {mov['speed_stats']['std']:.2f}\n")
            f.write(f"  95th percentile: {mov['speed_stats']['percentiles'][0.95]:.2f}\n")
            f.write(f"  99th percentile: {mov['speed_stats']['percentiles'][0.99]:.2f}\n\n")
            
            f.write("Large position jumps:\n")
            f.write(f"  Count: {mov['large_jumps']['count']} "
                   f"({mov['large_jumps']['percent']:.2f}%)\n")
            f.write(f"  Max jump: {mov['large_jumps']['max_jump']:.2f} units\n\n")
            
            f.write("Unrealistic speeds:\n")
            f.write(f"  Count: {mov['unrealistic_speeds']['count']} "
                   f"({mov['unrealistic_speeds']['percent']:.2f}%)\n")
            f.write(f"  Max speed: {mov['unrealistic_speeds']['max_speed']:.2f} units/second\n\n")
            
            f.write("Acceleration issues:\n")
            f.write(f"  Count: {mov['acceleration_issues']['count']} "
                   f"({mov['acceleration_issues']['percent']:.2f}%)\n")
            f.write(f"  Max acceleration: {mov['acceleration_issues']['max_acceleration']:.2f} units/second²\n\n")
            
            # Period-specific Validation Section
            f.write("4. PERIOD-SPECIFIC VALIDATION\n")
            f.write("-----------------------------\n")
            per = validation_results['period_checks']
            
            for period, stats in per['period_stats'].items():
                f.write(f"Period {period} ({stats['count']} points):\n")
                f.write(f"  X range: {stats['position_ranges']['x_min']:.2f} to "
                       f"{stats['position_ranges']['x_max']:.2f} (mean: {stats['position_ranges']['x_mean']:.2f})\n")
                f.write(f"  Y range: {stats['position_ranges']['y_min']:.2f} to "
                       f"{stats['position_ranges']['y_max']:.2f} (mean: {stats['position_ranges']['y_mean']:.2f})\n")
                
                if 'movement_stats' in stats:
                    f.write(f"  Mean speed: {stats['movement_stats']['mean_speed']:.2f} units/second\n")
                    f.write(f"  Max speed: {stats['movement_stats']['max_speed']:.2f} units/second\n")
                    f.write(f"  Large jumps: {stats['movement_stats']['large_jumps']}\n")
                    f.write(f"  Unrealistic speeds: {stats['movement_stats']['unrealistic_speeds']}\n")
                
                f.write("\n")
            
            if per['period_comparisons']:
                f.write("Period comparisons:\n")
                for comp_name, comp_stats in per['period_comparisons'].items():
                    f.write(f"  {comp_name}:\n")
                    f.write(f"    X mean difference: {comp_stats['x_mean_diff']:.2f}\n")
                    f.write(f"    Y mean difference: {comp_stats['y_mean_diff']:.2f}\n")
                    
                    if comp_stats['speed_diff'] is not None:
                        f.write(f"    Speed difference: {comp_stats['speed_diff']:.2f}\n")
                    
                    f.write("\n")
            
            # Summary and Recommendations
            f.write("5. SUMMARY AND RECOMMENDATIONS\n")
            f.write("------------------------------\n")
            
            # Count issues
            issues = []
            
            if basic['missing_values']['total_rows_with_missing'] > 0:
                issues.append(f"Missing values: {basic['missing_values']['total_rows_with_missing']} rows")
            
            if basic['time_sequence_issues']:
                issues.append(f"Time sequence issues in {len(basic['time_sequence_issues'])} periods")
            
            out_of_bounds_total = pos['out_of_bounds']['x_count'] + pos['out_of_bounds']['y_count']
            if out_of_bounds_total > 0:
                issues.append(f"Out-of-bounds positions: {out_of_bounds_total} points")
            
            if mov['large_jumps']['count'] > 0:
                issues.append(f"Large position jumps: {mov['large_jumps']['count']} jumps")
            
            if mov['unrealistic_speeds']['count'] > 0:
                issues.append(f"Unrealistic speeds: {mov['unrealistic_speeds']['count']} instances")
            
            if mov['acceleration_issues']['count'] > 0:
                issues.append(f"Acceleration issues: {mov['acceleration_issues']['count']} instances")
            
            # Write summary
            if issues:
                f.write("Issues detected:\n")
                for issue in issues:
                    f.write(f"  - {issue}\n")
            else:
                f.write("No major issues detected in the predictions.\n")
            
            f.write("\nRecommendations:\n")
            
            # Generate recommendations based on issues
            if basic['missing_values']['total_rows_with_missing'] > 0:
                f.write("  - Investigate and fill missing values using interpolation or other methods.\n")
            
            if basic['time_sequence_issues']:
                f.write("  - Fix time sequence issues to ensure proper temporal ordering.\n")
            
            if out_of_bounds_total > 0:
                f.write("  - Review out-of-bounds positions and apply boundary constraints if needed.\n")
            
            if mov['large_jumps']['count'] > mov['large_jumps']['count'] * 0.01:  # More than 1%
                f.write("  - Apply smoothing to reduce unrealistic position jumps.\n")
            
            if mov['unrealistic_speeds']['count'] > 0:
                f.write("  - Apply physical constraints to ensure realistic ball speeds.\n")
            
            if mov['acceleration_issues']['count'] > 0:
                f.write("  - Apply acceleration constraints for more natural ball movement.\n")
            
            # Add general recommendations
            f.write("  - Consider applying temporal smoothing if not already done.\n")
            f.write("  - Validate predictions against known ball movement patterns in football.\n")
            
            f.write("\n=================================================\n")
            f.write("END OF REPORT\n")
            f.write("=================================================\n")
        
        print(f"Validation report generated: {report_path}")
        
        # Generate a summary plot
        self._generate_summary_plot(validation_results)
    
    def _generate_summary_plot(self, validation_results):
        """
        Generate a summary plot showing key validation metrics.
        
        Args:
            validation_results (dict): Combined results from all validation checks
        """
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Ball Prediction Validation Summary', fontsize=16)
        
        # 1. Ball Position Distribution
        ax1 = axs[0, 0]
        ax1.hexbin(self.predictions['ball_x'], self.predictions['ball_y'], 
                  gridsize=30, cmap='Blues')
        ax1.set_title('Ball Position Distribution')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.grid(True, alpha=0.3)
        
        # Add expected pitch boundaries
        ax1.axvline(x=self.min_expected_x, color='r', linestyle='--', alpha=0.5)
        ax1.axvline(x=self.max_expected_x, color='r', linestyle='--', alpha=0.5)
        ax1.axhline(y=self.min_expected_y, color='r', linestyle='--', alpha=0.5)
        ax1.axhline(y=self.max_expected_y, color='r', linestyle='--', alpha=0.5)
        
        # 2. Ball Speed Distribution
        ax2 = axs[0, 1]
        
        # Load movement data if available
        movement_file = os.path.join(self.output_dir, 'ball_movement_data.csv')
        if os.path.exists(movement_file):
            movement_data = pd.read_csv(movement_file)
            
            # Plot speed histogram
            speeds = movement_data['ball_speed'].dropna()
            ax2.hist(speeds, bins=50, alpha=0.7, color='skyblue')
            ax2.axvline(x=self.max_realistic_speed, color='r', linestyle='--', 
                       label=f'Max Realistic Speed ({self.max_realistic_speed})')
            
            # Add statistical markers
            mean_speed = speeds.mean()
            median_speed = speeds.median()
            percentile_95 = speeds.quantile(0.95)
            
            ax2.axvline(x=mean_speed, color='g', linestyle='-', 
                       label=f'Mean Speed ({mean_speed:.2f})')
            ax2.axvline(x=median_speed, color='b', linestyle='-', 
                       label=f'Median Speed ({median_speed:.2f})')
            ax2.axvline(x=percentile_95, color='orange', linestyle='-', 
                       label=f'95th Percentile ({percentile_95:.2f})')
            
            ax2.set_title('Ball Speed Distribution')
            ax2.set_xlabel('Speed (units/second)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Movement data not available', 
                    horizontalalignment='center', verticalalignment='center')
        
        # 3. Position Ranges by Period
        ax3 = axs[1, 0]
        period_stats = validation_results['period_checks']['period_stats']
        periods = list(period_stats.keys())
        
        x_means = [period_stats[p]['position_ranges']['x_mean'] for p in periods]
        y_means = [period_stats[p]['position_ranges']['y_mean'] for p in periods]
        x_mins = [period_stats[p]['position_ranges']['x_min'] for p in periods]
        x_maxs = [period_stats[p]['position_ranges']['x_max'] for p in periods]
        y_mins = [period_stats[p]['position_ranges']['y_min'] for p in periods]
        y_maxs = [period_stats[p]['position_ranges']['y_max'] for p in periods]
        
        # Plot X position ranges
        ax3.bar(periods, x_maxs, alpha=0.3, color='skyblue', label='X Max')
        ax3.bar(periods, x_means, alpha=0.7, color='royalblue', label='X Mean')
        ax3.bar(periods, x_mins, alpha=0.3, color='lightblue', label='X Min')
        
        ax3.set_title('X Position Ranges by Period')
        ax3.set_xlabel('Period')
        ax3.set_ylabel('X Position')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Issues Summary
        ax4 = axs[1, 1]
        
        # Count different types of issues
        mov = validation_results['movement_checks']
        pos = validation_results['position_checks']
        
        issue_types = ['Missing Values', 'Out-of-Bounds', 'Large Jumps', 'Unrealistic Speeds']
        issue_counts = [
            validation_results['basic_checks']['missing_values']['total_rows_with_missing'],
            pos['out_of_bounds']['x_count'] + pos['out_of_bounds']['y_count'],
            mov['large_jumps']['count'],
            mov['unrealistic_speeds']['count']
        ]
        
        # Convert to percentages
        total_predictions = validation_results['basic_checks']['total_predictions']
        issue_percentages = [(count / total_predictions) * 100 for count in issue_counts]
        
        # Plot issues
        ax4.bar(issue_types, issue_percentages, color=['skyblue', 'royalblue', 'lightcoral', 'indianred'])
        ax4.set_title('Validation Issues (% of Total Predictions)')
        ax4.set_xlabel('Issue Type')
        ax4.set_ylabel('Percentage (%)')
        ax4.set_ylim(0, max(issue_percentages) * 1.2 if max(issue_percentages) > 0 else 1)
        
        for i, v in enumerate(issue_percentages):
            ax4.text(i, v + 0.1, f'{v:.2f}%', ha='center')
        
        ax4.grid(True, alpha=0.3)
        
        # Adjust layout and save plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = os.path.join(self.output_dir, 'validation_summary_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Validation summary plot saved to: {plot_path}")


def main():
    """
    Main function to run the validation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate football ball position predictions')
    parser.add_argument('--predictions', required=True, 
                        help='Path to predicted ball positions CSV')
    parser.add_argument('--output-dir', default='validation_results', 
                        help='Directory to save validation results')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = BallPredictionValidator(
        predictions_path=args.predictions,
        output_dir=args.output_dir
    )
    
    # Run all validations
    validation_results = validator.run_all_validations(generate_report=True)
    
    print("\nValidation complete. Check the report for detailed results.")


if __name__ == "__main__":
    main()