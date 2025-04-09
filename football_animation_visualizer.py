"""
Football Animation Visualizer
-----------------------------
This script creates an animated visualization of football tracking data with predicted 
ball positions to evaluate model accuracy.

Required libraries:
- matplotlib
- pandas
- numpy
- tqdm
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Wedge
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse


class FootballAnimationVisualizer:
    """
    Class for creating animated visualizations of football tracking data
    with predicted ball positions.
    """
    
    def __init__(self, home_data_path, away_data_path, 
                 predicted_ball_path, actual_ball_path=None, 
                 output_dir='animations'):
        """
        Initialize the visualizer with paths to data files.
        
        Args:
            home_data_path: Path to home team tracking data CSV
            away_data_path: Path to away team tracking data CSV
            predicted_ball_path: Path to predicted ball positions CSV
            actual_ball_path: Path to actual ball positions CSV (optional)
            output_dir: Directory to save animation files
        """
        self.home_data_path = home_data_path
        self.away_data_path = away_data_path
        self.predicted_ball_path = predicted_ball_path
        self.actual_ball_path = actual_ball_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Colors
        self.home_color = 'blue'
        self.away_color = 'red'
        self.predicted_ball_color = 'orange'
        self.actual_ball_color = 'white'
        
        # Pitch dimensions (adjust based on your data)
        self.pitch_length = 10000
        self.pitch_width = 6800
        
        # Match data
        self.home_data = None
        self.away_data = None
        self.predicted_ball_data = None
        self.actual_ball_data = None
        self.home_player_ids = []
        self.away_player_ids = []
        self.periods = []
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess tracking data."""
        print("Loading tracking data...")
        
        # Load team tracking data
        self.home_data = pd.read_csv(self.home_data_path)
        self.away_data = pd.read_csv(self.away_data_path)
        
        # Load ball data
        self.predicted_ball_data = pd.read_csv(self.predicted_ball_path)
        
        if self.actual_ball_path:
            self.actual_ball_data = pd.read_csv(self.actual_ball_path)
        
        # Extract player IDs
        self.home_player_ids = self._extract_player_ids(self.home_data, 'home')
        self.away_player_ids = self._extract_player_ids(self.away_data, 'away')
        
        print(f"Found {len(self.home_player_ids)} home players and {len(self.away_player_ids)} away players")
        
        # Get periods
        self.periods = self.home_data['IdPeriod'].unique()
        
        # Ensure data alignment by matching on MatchId, IdPeriod, and Time
        self._align_data()
    
    def _extract_player_ids(self, data, prefix):
        """Extract player IDs from column names."""
        player_cols = [col for col in data.columns if f'{prefix}_' in col and '_x' in col]
        player_ids = []
        
        for col in player_cols:
            # Extract ID from column name (e.g., 'home_123_x' -> 'home_123')
            parts = col.split('_')
            if len(parts) >= 2:
                player_id = f"{prefix}_{parts[1]}"
                if player_id not in player_ids:
                    player_ids.append(player_id)
        
        return player_ids
    
    def _align_data(self):
        """Ensure all datasets are aligned on time indices."""
        # Create a master index of all time points
        common_cols = ['MatchId', 'IdPeriod', 'Time']
        
        # Start with home data as the base
        aligned_idx = self.home_data[common_cols].copy()
        
        # Merge with away data
        aligned_idx = pd.merge(aligned_idx, self.away_data[common_cols], 
                              on=common_cols, how='inner')
        
        # Merge with predicted ball data
        aligned_idx = pd.merge(aligned_idx, self.predicted_ball_data[common_cols],
                             on=common_cols, how='inner')
        
        # Merge with actual ball data if available
        if self.actual_ball_data is not None:
            aligned_idx = pd.merge(aligned_idx, self.actual_ball_data[common_cols],
                                 on=common_cols, how='inner')
        
        print(f"Data aligned: {len(aligned_idx)} common time points")
        
        # Filter all datasets to only include common time points
        self.home_data = pd.merge(aligned_idx, self.home_data, on=common_cols)
        self.away_data = pd.merge(aligned_idx, self.away_data, on=common_cols)
        self.predicted_ball_data = pd.merge(aligned_idx, self.predicted_ball_data, on=common_cols)
        
        if self.actual_ball_data is not None:
            self.actual_ball_data = pd.merge(aligned_idx, self.actual_ball_data, on=common_cols)
    
    def create_animation(self, period=1, start_time=None, end_time=None, 
                         fps=10, duration=20, show_trails=True, trail_length=20):
        """
        Create an animation for a specific period and time range.
        
        Args:
            period: Match period to animate
            start_time: Start time (or None for beginning of period)
            end_time: End time (or None for end of period)
            fps: Frames per second
            duration: Duration of animation in seconds
            show_trails: Whether to show movement trails
            trail_length: Length of movement trails (in frames)
            
        Returns:
            Path to saved animation file
        """
        # Filter data for the specified period
        period_filter = (self.home_data['IdPeriod'] == period)
        home_period = self.home_data[period_filter].copy()
        away_period = self.away_data[period_filter].copy()
        pred_ball_period = self.predicted_ball_data[period_filter].copy()
        
        if self.actual_ball_data is not None:
            actual_ball_period = self.actual_ball_data[period_filter].copy()
        else:
            actual_ball_period = None
        
        # Apply time filters if specified
        if start_time is not None:
            time_filter = (home_period['Time'] >= start_time)
            home_period = home_period[time_filter]
            away_period = away_period[time_filter]
            pred_ball_period = pred_ball_period[time_filter]
            if actual_ball_period is not None:
                actual_ball_period = actual_ball_period[time_filter]
        
        if end_time is not None:
            time_filter = (home_period['Time'] <= end_time)
            home_period = home_period[time_filter]
            away_period = away_period[time_filter]
            pred_ball_period = pred_ball_period[time_filter]
            if actual_ball_period is not None:
                actual_ball_period = actual_ball_period[time_filter]
        
        # Check if we have data
        if len(home_period) == 0:
            print(f"No data found for period {period} in the specified time range")
            return None
        
        print(f"Creating animation for period {period} with {len(home_period)} frames")
        
        # Calculate total frames and sampling
        total_frames = fps * duration
        step = max(1, len(home_period) // total_frames)
        
        # Sample frames to achieve desired duration
        indices = list(range(0, len(home_period), step))[:total_frames]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Draw football pitch
        self._draw_pitch(ax)
        
        # Set axis limits based on pitch dimensions
        padding = 500  # Add some padding around the pitch
        ax.set_xlim(-self.pitch_length/2 - padding, self.pitch_length/2 + padding)
        ax.set_ylim(-self.pitch_width/2 - padding, self.pitch_width/2 + padding)
        
        # Initialize player and ball objects
        home_players = {}
        away_players = {}
        for player_id in self.home_player_ids:
            x_col = f"{player_id}_x"
            y_col = f"{player_id}_y"
            if x_col in home_period.columns and y_col in home_period.columns:
                # Player marker
                home_players[player_id] = ax.plot([], [], 'o', color=self.home_color, 
                                                 markersize=8, alpha=0.8)[0]
                # Player trail (if enabled)
                if show_trails:
                    home_players[f"{player_id}_trail"] = ax.plot([], [], '-', 
                                                           color=self.home_color, 
                                                           linewidth=1, alpha=0.3)[0]
        
        for player_id in self.away_player_ids:
            x_col = f"{player_id}_x"
            y_col = f"{player_id}_y"
            if x_col in away_period.columns and y_col in away_period.columns:
                # Player marker
                away_players[player_id] = ax.plot([], [], 'o', color=self.away_color, 
                                                 markersize=8, alpha=0.8)[0]
                # Player trail (if enabled)
                if show_trails:
                    away_players[f"{player_id}_trail"] = ax.plot([], [], '-', 
                                                           color=self.away_color, 
                                                           linewidth=1, alpha=0.3)[0]
        
        # Ball objects
        predicted_ball = ax.plot([], [], 'o', color=self.predicted_ball_color, 
                                markersize=10, label='Predicted Ball')[0]
        
        if show_trails:
            predicted_ball_trail = ax.plot([], [], '-', color=self.predicted_ball_color, 
                                          linewidth=2, alpha=0.5)[0]
        
        if actual_ball_period is not None:
            actual_ball = ax.plot([], [], 'o', color=self.actual_ball_color, 
                                 markersize=10, label='Actual Ball')[0]
            if show_trails:
                actual_ball_trail = ax.plot([], [], '-', color=self.actual_ball_color, 
                                           linewidth=2, alpha=0.5)[0]
        
        # Time display
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                           fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
        
        # Legend
        ax.legend(loc='upper right', fontsize=12)
        
        # Title
        if actual_ball_period is not None:
            title = f"Football Tracking with Predicted vs Actual Ball Positions - Period {period}"
        else:
            title = f"Football Tracking with Predicted Ball Positions - Period {period}"
        
        ax.set_title(title, fontsize=16)
        
        # Animation update function
        def update(frame_idx):
            if frame_idx >= len(indices):
                return
            
            i = indices[frame_idx]
            
            # Update home players
            for player_id in self.home_player_ids:
                x_col = f"{player_id}_x"
                y_col = f"{player_id}_y"
                
                if x_col in home_period.columns and y_col in home_period.columns:
                    x = home_period.iloc[i][x_col]
                    y = home_period.iloc[i][y_col]
                    
                    # Skip if position is missing
                    if pd.isna(x) or pd.isna(y):
                        continue
                    
                    # Update player position
                    home_players[player_id].set_data(x, y)
                    
                    # Update player trail
                    if show_trails and f"{player_id}_trail" in home_players:
                        start_idx = max(0, i - trail_length)
                        trail_x = home_period.iloc[start_idx:i+1][x_col].dropna().values
                        trail_y = home_period.iloc[start_idx:i+1][y_col].dropna().values
                        home_players[f"{player_id}_trail"].set_data(trail_x, trail_y)
            
            # Update away players
            for player_id in self.away_player_ids:
                x_col = f"{player_id}_x"
                y_col = f"{player_id}_y"
                
                if x_col in away_period.columns and y_col in away_period.columns:
                    x = away_period.iloc[i][x_col]
                    y = away_period.iloc[i][y_col]
                    
                    # Skip if position is missing
                    if pd.isna(x) or pd.isna(y):
                        continue
                    
                    # Update player position
                    away_players[player_id].set_data(x, y)
                    
                    # Update player trail
                    if show_trails and f"{player_id}_trail" in away_players:
                        start_idx = max(0, i - trail_length)
                        trail_x = away_period.iloc[start_idx:i+1][x_col].dropna().values
                        trail_y = away_period.iloc[start_idx:i+1][y_col].dropna().values
                        away_players[f"{player_id}_trail"].set_data(trail_x, trail_y)
            
            # Update predicted ball
            pred_x = pred_ball_period.iloc[i]['ball_x']
            pred_y = pred_ball_period.iloc[i]['ball_y']
            predicted_ball.set_data(pred_x, pred_y)
            
            if show_trails:
                start_idx = max(0, i - trail_length)
                pred_trail_x = pred_ball_period.iloc[start_idx:i+1]['ball_x'].values
                pred_trail_y = pred_ball_period.iloc[start_idx:i+1]['ball_y'].values
                predicted_ball_trail.set_data(pred_trail_x, pred_trail_y)
            
            # Update actual ball if available
            if actual_ball_period is not None:
                actual_x = actual_ball_period.iloc[i]['ball_x']
                actual_y = actual_ball_period.iloc[i]['ball_y']
                actual_ball.set_data(actual_x, actual_y)
                
                if show_trails:
                    start_idx = max(0, i - trail_length)
                    actual_trail_x = actual_ball_period.iloc[start_idx:i+1]['ball_x'].values
                    actual_trail_y = actual_ball_period.iloc[start_idx:i+1]['ball_y'].values
                    actual_ball_trail.set_data(actual_trail_x, actual_trail_y)
                
                # If we have both predictions, show error
                if not pd.isna(pred_x) and not pd.isna(pred_y) and not pd.isna(actual_x) and not pd.isna(actual_y):
                    error = np.sqrt((pred_x - actual_x)**2 + (pred_y - actual_y)**2)
                    time_text.set_text(f'Time: {home_period.iloc[i]["Time"]}, Error: {error:.2f}')
                else:
                    time_text.set_text(f'Time: {home_period.iloc[i]["Time"]}')
            else:
                time_text.set_text(f'Time: {home_period.iloc[i]["Time"]}')
        
        # Create animation
        print("Creating animation...")
        anim = animation.FuncAnimation(fig, update, frames=len(indices),
                                     interval=1000/fps, blit=False)
        
        # Save animation
        output_file = os.path.join(self.output_dir, f'football_animation_period_{period}.mp4')
        writer = animation.FFMpegWriter(fps=fps, bitrate=3600)
        anim.save(output_file, writer=writer)
        
        print(f"Animation saved to: {output_file}")
        plt.close()
        
        return output_file
    
    def _draw_pitch(self, ax):
        """Draw a football pitch on the given axes."""
        # Pitch dimensions
        pitch_length = self.pitch_length
        pitch_width = self.pitch_width
        
        # Draw pitch outline
        pitch = Rectangle((-pitch_length/2, -pitch_width/2), pitch_length, pitch_width,
                        facecolor='#3a813c', edgecolor='white', alpha=0.8)
        ax.add_patch(pitch)
        
        # Draw halfway line
        ax.plot([0, 0], [-pitch_width/2, pitch_width/2], 'white', linewidth=2)
        
        # Draw center circle
        center_circle = Circle((0, 0), 915, fill=False, color='white', linewidth=2)
        ax.add_patch(center_circle)
        
        # Draw center spot
        center_spot = Circle((0, 0), 20, color='white')
        ax.add_patch(center_spot)
        
        # Draw penalty areas
        penalty_area_length = 1650
        penalty_area_width = 4030
        
        # Left penalty area
        left_penalty_area = Rectangle(
            (-pitch_length/2, -penalty_area_width/2),
            penalty_area_length, penalty_area_width,
            fill=False, color='white', linewidth=2
        )
        ax.add_patch(left_penalty_area)
        
        # Right penalty area
        right_penalty_area = Rectangle(
            (pitch_length/2 - penalty_area_length, -penalty_area_width/2),
            penalty_area_length, penalty_area_width,
            fill=False, color='white', linewidth=2
        )
        ax.add_patch(right_penalty_area)
        
        # Draw goal areas
        goal_area_length = 550
        goal_area_width = 1830
        
        # Left goal area
        left_goal_area = Rectangle(
            (-pitch_length/2, -goal_area_width/2),
            goal_area_length, goal_area_width,
            fill=False, color='white', linewidth=2
        )
        ax.add_patch(left_goal_area)
        
        # Right goal area
        right_goal_area = Rectangle(
            (pitch_length/2 - goal_area_length, -goal_area_width/2),
            goal_area_length, goal_area_width,
            fill=False, color='white', linewidth=2
        )
        ax.add_patch(right_goal_area)
        
        # Draw penalty spots
        left_penalty_spot = Circle((-pitch_length/2 + 1100, 0), 20, color='white')
        ax.add_patch(left_penalty_spot)
        
        right_penalty_spot = Circle((pitch_length/2 - 1100, 0), 20, color='white')
        ax.add_patch(right_penalty_spot)
        
        # Draw corner arcs
        corner_radius = 100
        
        # Top-left corner
        top_left_corner = Wedge(
            (-pitch_length/2, pitch_width/2), corner_radius, 0, 90,
            width=2, color='white', fill=False
        )
        ax.add_patch(top_left_corner)
        
        # Bottom-left corner
        bottom_left_corner = Wedge(
            (-pitch_length/2, -pitch_width/2), corner_radius, 270, 360,
            width=2, color='white', fill=False
        )
        ax.add_patch(bottom_left_corner)
        
        # Top-right corner
        top_right_corner = Wedge(
            (pitch_length/2, pitch_width/2), corner_radius, 90, 180,
            width=2, color='white', fill=False
        )
        ax.add_patch(top_right_corner)
        
        # Bottom-right corner
        bottom_right_corner = Wedge(
            (pitch_length/2, -pitch_width/2), corner_radius, 180, 270,
            width=2, color='white', fill=False
        )
        ax.add_patch(bottom_right_corner)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add some basic labels
        ax.text(0, -pitch_width/2 - 200, 'Center Line', ha='center', color='white', fontsize=10)
        ax.text(-pitch_length/2 + 825, -pitch_width/2 - 200, 'Penalty Area', ha='center', color='white', fontsize=10)
        ax.text(pitch_length/2 - 825, -pitch_width/2 - 200, 'Penalty Area', ha='center', color='white', fontsize=10)
    
    def create_error_heatmap(self, period=1):
        """
        Create a heatmap showing prediction errors across the pitch.
        Requires actual ball data.
        
        Args:
            period: Match period to analyze
            
        Returns:
            Path to saved heatmap image
        """
        if self.actual_ball_data is None:
            print("Error: Actual ball data is required for error heatmap")
            return None
        
        # Filter data for the specified period
        period_filter = (self.home_data['IdPeriod'] == period)
        pred_ball_period = self.predicted_ball_data[period_filter].copy()
        actual_ball_period = self.actual_ball_data[period_filter].copy()
        
        # Calculate errors
        pred_ball_period['error_x'] = pred_ball_period['ball_x'] - actual_ball_period['ball_x']
        pred_ball_period['error_y'] = pred_ball_period['ball_y'] - actual_ball_period['ball_y']
        pred_ball_period['error_dist'] = np.sqrt(
            pred_ball_period['error_x']**2 + pred_ball_period['error_y']**2
        )
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Draw football pitch
        self._draw_pitch(ax)
        
        # Create heatmap based on ball positions and errors
        # Group the pitch into bins
        x_bins = np.linspace(-self.pitch_length/2, self.pitch_length/2, 30)
        y_bins = np.linspace(-self.pitch_width/2, self.pitch_width/2, 20)
        
        # Initialize error grid
        error_grid = np.zeros((len(y_bins)-1, len(x_bins)-1))
        count_grid = np.zeros((len(y_bins)-1, len(x_bins)-1))
        
        # Populate error grid
        for _, row in pred_ball_period.iterrows():
            x, y = row['ball_x'], row['ball_y']
            error = row['error_dist']
            
            # Skip if out of bounds or NaN
            if (pd.isna(x) or pd.isna(y) or pd.isna(error) or 
                x < -self.pitch_length/2 or x > self.pitch_length/2 or 
                y < -self.pitch_width/2 or y > self.pitch_width/2):
                continue
            
            # Find the bin
            x_idx = np.digitize(x, x_bins) - 1
            y_idx = np.digitize(y, y_bins) - 1
            
            # Ensure valid indices (should be unnecessary but just to be safe)
            if 0 <= x_idx < len(x_bins)-1 and 0 <= y_idx < len(y_bins)-1:
                error_grid[y_idx, x_idx] += error
                count_grid[y_idx, x_idx] += 1
        
        # Calculate average error per bin
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_error_grid = np.divide(error_grid, count_grid)
            avg_error_grid[np.isnan(avg_error_grid)] = 0
        
        # Plot heatmap
        cmap = plt.cm.get_cmap('hot_r')
        heatmap = ax.pcolormesh(x_bins, y_bins, avg_error_grid, cmap=cmap, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label('Average Prediction Error (units)', fontsize=12)
        
        # Set title
        ax.set_title(f'Ball Position Prediction Error Heatmap - Period {period}', fontsize=16)
        
        # Save figure
        output_file = os.path.join(self.output_dir, f'error_heatmap_period_{period}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Error heatmap saved to: {output_file}")
        return output_file
    
    def create_trajectory_comparison(self, start_time, end_time, period=1):
        """
        Create a static visualization comparing predicted and actual ball trajectories.
        
        Args:
            start_time: Start time for the visualization
            end_time: End time for the visualization
            period: Match period to visualize
            
        Returns:
            Path to saved image file
        """
        # Filter data for the specified period and time range
        period_filter = (self.home_data['IdPeriod'] == period)
        time_filter = (self.home_data['Time'] >= start_time) & (self.home_data['Time'] <= end_time)
        
        home_data_subset = self.home_data[period_filter & time_filter].copy()
        away_data_subset = self.away_data[period_filter & time_filter].copy()
        pred_ball_subset = self.predicted_ball_data[period_filter & time_filter].copy()
        
        if self.actual_ball_data is not None:
            actual_ball_subset = self.actual_ball_data[period_filter & time_filter].copy()
        else:
            actual_ball_subset = None
        
        # Check if we have data
        if len(home_data_subset) == 0:
            print(f"No data found for period {period} in the specified time range")
            return None
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Draw football pitch
        self._draw_pitch(ax)
        
        # Set axis limits based on pitch dimensions
        padding = 500
        ax.set_xlim(-self.pitch_length/2 - padding, self.pitch_length/2 + padding)
        ax.set_ylim(-self.pitch_width/2 - padding, self.pitch_width/2 + padding)
        
        # Get first and last positions for home players
        for player_id in self.home_player_ids:
            x_col = f"{player_id}_x"
            y_col = f"{player_id}_y"
            
            if x_col in home_data_subset.columns and y_col in home_data_subset.columns:
                # Get first and last valid positions
                positions = home_data_subset[[x_col, y_col]].dropna()
                
                if len(positions) > 0:
                    first_pos = positions.iloc[0]
                    last_pos = positions.iloc[-1]
                    
                    # Plot player positions
                    ax.plot(first_pos[x_col], first_pos[y_col], 'o', color=self.home_color, 
                           alpha=0.6, markersize=8)
                    ax.plot(last_pos[x_col], last_pos[y_col], 's', color=self.home_color, 
                           alpha=0.9, markersize=8)
                    
                    # Plot player trajectories
                    ax.plot(positions[x_col], positions[y_col], '-', color=self.home_color, 
                           alpha=0.3, linewidth=1)
        
        # Get first and last positions for away players
        for player_id in self.away_player_ids:
            x_col = f"{player_id}_x"
            y_col = f"{player_id}_y"
            
            if x_col in away_data_subset.columns and y_col in away_data_subset.columns:
                # Get first and last valid positions
                positions = away_data_subset[[x_col, y_col]].dropna()
                
                if len(positions) > 0:
                    first_pos = positions.iloc[0]
                    last_pos = positions.iloc[-1]
                    
                    # Plot player positions
                    ax.plot(first_pos[x_col], first_pos[y_col], 'o', color=self.away_color, 
                           alpha=0.6, markersize=8)
                    ax.plot(last_pos[x_col], last_pos[y_col], 's', color=self.away_color, 
                           alpha=0.9, markersize=8)
                    
                    # Plot player trajectories
                    ax.plot(positions[x_col], positions[y_col], '-', color=self.away_color, 
                           alpha=0.3, linewidth=1)
        
        # Plot predicted ball trajectory
        pred_x = pred_ball_subset['ball_x'].values
        pred_y = pred_ball_subset['ball_y'].values
        
        # Start and end markers for predicted ball
        ax.plot(pred_x[0], pred_y[0], 'o', color=self.predicted_ball_color, 
               markersize=10, label='Predicted Ball Start')
        ax.plot(pred_x[-1], pred_y[-1], 's', color=self.predicted_ball_color, 
               markersize=10, label='Predicted Ball End')
        
        # Plot predicted ball path
        ax.plot(pred_x, pred_y, '-', color=self.predicted_ball_color, 
               linewidth=2, alpha=0.8, label='Predicted Ball Path')
        
        # Plot actual ball trajectory if available
        if actual_ball_subset is not None:
            actual_x = actual_ball_subset['ball_x'].values
            actual_y = actual_ball_subset['ball_y'].values
            
            # Start and end markers for actual ball
            ax.plot(actual_x[0], actual_y[0], 'o', color=self.actual_ball_color, 
                   markersize=10, label='Actual Ball Start')
            ax.plot(actual_x[-1], actual_y[-1], 's', color=self.actual_ball_color, 
                   markersize=10, label='Actual Ball End')
            
            # Plot actual ball path
            ax.plot(actual_x, actual_y, '-', color=self.actual_ball_color, 
                   linewidth=2, alpha=0.8, label='Actual Ball Path')
            
            # Calculate and visualize errors at specific points
            error_step = max(1, len(pred_x) // 10)  # Show errors at about 10 points
            for i in range(0, len(pred_x), error_step):
                if i < len(actual_x):  # Ensure we have actual data at this point
                    # Draw line between predicted and actual positions
                    ax.plot([pred_x[i], actual_x[i]], [pred_y[i], actual_y[i]], 
                           '--', color='yellow', linewidth=1, alpha=0.6)
                    
                    # Calculate error
                    error = np.sqrt((pred_x[i] - actual_x[i])**2 + (pred_y[i] - actual_y[i])**2)
                    
                    # Add error text
                    if i % (error_step * 2) == 0:  # Show text for fewer points to avoid clutter
                        ax.text((pred_x[i] + actual_x[i])/2, (pred_y[i] + actual_y[i])/2, 
                               f'{error:.0f}', color='white', fontsize=8,
                               bbox=dict(facecolor='black', alpha=0.6))
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Set title
        if actual_ball_subset is not None:
            title = (f'Ball Trajectory Comparison - Period {period}\n'
                    f'Time: {start_time} to {end_time}')
        else:
            title = (f'Predicted Ball Trajectory - Period {period}\n'
                    f'Time: {start_time} to {end_time}')
        
        ax.set_title(title, fontsize=16)
        
        # Add markers legend
        circle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                           markersize=8, label='Start Position')
        square = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                           markersize=8, label='End Position')
        
        home = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.home_color, 
                         markersize=8, label='Home Team')
        away = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.away_color, 
                         markersize=8, label='Away Team')
        
        # Add a second legend for markers
        ax.add_artist(ax.legend(handles=[circle, square, home, away], 
                               loc='lower right', fontsize=10))
        
        # Save figure
        output_file = os.path.join(
            self.output_dir, 
            f'trajectory_comparison_period_{period}_{start_time}_to_{end_time}.png'
        )
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Trajectory comparison saved to: {output_file}")
        return output_file
    
    def create_error_analysis(self, period=1):
        """
        Create error analysis charts for predicted ball positions.
        Requires actual ball data.
        
        Args:
            period: Match period to analyze
            
        Returns:
            Path to saved analysis image
        """
        if self.actual_ball_data is None:
            print("Error: Actual ball data is required for error analysis")
            return None
        
        # Filter data for the specified period
        period_filter = (self.home_data['IdPeriod'] == period)
        pred_ball_period = self.predicted_ball_data[period_filter].copy()
        actual_ball_period = self.actual_ball_data[period_filter].copy()
        
        # Calculate errors
        pred_ball_period['error_x'] = pred_ball_period['ball_x'] - actual_ball_period['ball_x']
        pred_ball_period['error_y'] = pred_ball_period['ball_y'] - actual_ball_period['ball_y']
        pred_ball_period['error_dist'] = np.sqrt(
            pred_ball_period['error_x']**2 + pred_ball_period['error_y']**2
        )
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Ball Position Prediction Error Analysis - Period {period}', fontsize=18)
        
        # 1. Error over time
        ax1 = axs[0, 0]
        ax1.plot(pred_ball_period['Time'], pred_ball_period['error_dist'], 
                '-', color='red', alpha=0.7)
        ax1.set_title('Error Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Error Distance')
        ax1.grid(True, alpha=0.3)
        
        # Add moving average
        window_size = min(100, len(pred_ball_period) // 10)
        if window_size > 0:
            pred_ball_period['error_ma'] = pred_ball_period['error_dist'].rolling(
                window=window_size, center=True).mean()
            ax1.plot(pred_ball_period['Time'], pred_ball_period['error_ma'], 
                    '-', color='blue', linewidth=2, 
                    label=f'Moving Average (window={window_size})')
            ax1.legend()
        
        # 2. Error histogram
        ax2 = axs[0, 1]
        bins = np.linspace(0, pred_ball_period['error_dist'].quantile(0.99), 30)
        ax2.hist(pred_ball_period['error_dist'].dropna(), bins=bins, 
                alpha=0.7, color='blue')
        ax2.set_title('Error Distribution')
        ax2.set_xlabel('Error Distance')
        ax2.set_ylabel('Frequency')
        
        # Add mean and median lines
        mean_error = pred_ball_period['error_dist'].mean()
        median_error = pred_ball_period['error_dist'].median()
        ax2.axvline(mean_error, color='red', linestyle='--', 
                   label=f'Mean Error: {mean_error:.2f}')
        ax2.axvline(median_error, color='green', linestyle='--', 
                   label=f'Median Error: {median_error:.2f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. X-Y Error scatter plot
        ax3 = axs[1, 0]
        scatter = ax3.scatter(pred_ball_period['error_x'], pred_ball_period['error_y'], 
                            alpha=0.5, c=pred_ball_period['error_dist'], cmap='viridis')
        ax3.set_title('X-Y Error Distribution')
        ax3.set_xlabel('X Error')
        ax3.set_ylabel('Y Error')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax3.axvline(0, color='black', linestyle='-', alpha=0.3)
        
        # Add a circle to show the mean error distance
        circle = plt.Circle((0, 0), mean_error, fill=False, color='red', 
                           linestyle='--', label=f'Mean Error: {mean_error:.2f}')
        ax3.add_patch(circle)
        ax3.legend()
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax3)
        cbar.set_label('Error Distance')
        
        # 4. Error vs. Ball Speed
        ax4 = axs[1, 1]
        
        # Calculate ball speed
        actual_ball_period['ball_x_diff'] = actual_ball_period['ball_x'].diff()
        actual_ball_period['ball_y_diff'] = actual_ball_period['ball_y'].diff()
        actual_ball_period['ball_speed'] = np.sqrt(
            actual_ball_period['ball_x_diff']**2 + actual_ball_period['ball_y_diff']**2
        ) * 10  # Multiply by 10 to convert to units/second (assuming 0.1s intervals)
        
        # Join with error data
        speed_error_data = pd.concat([
            actual_ball_period['ball_speed'],
            pred_ball_period['error_dist']
        ], axis=1).dropna()
        
        # Create scatter plot
        scatter = ax4.scatter(speed_error_data['ball_speed'], speed_error_data['error_dist'], 
                            alpha=0.5, s=5)
        ax4.set_title('Error vs. Ball Speed')
        ax4.set_xlabel('Ball Speed (units/second)')
        ax4.set_ylabel('Error Distance')
        ax4.grid(True, alpha=0.3)
        
        # Add best fit line
        if len(speed_error_data) > 1:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                speed_error_data['ball_speed'], speed_error_data['error_dist']
            )
            x_line = np.array([speed_error_data['ball_speed'].min(), speed_error_data['ball_speed'].max()])
            y_line = slope * x_line + intercept
            ax4.plot(x_line, y_line, 'r', label=f'Fit: y={slope:.4f}x+{intercept:.2f}, R²={r_value**2:.2f}')
            ax4.legend()
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        output_file = os.path.join(self.output_dir, f'error_analysis_period_{period}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Error analysis saved to: {output_file}")
        return output_file


def main():
    """Main function to run the visualization."""
    parser = argparse.ArgumentParser(description='Football Tracking Visualization')
    parser.add_argument('--home', required=True, help='Path to home team data CSV')
    parser.add_argument('--away', required=True, help='Path to away team data CSV')
    parser.add_argument('--predicted-ball', required=True, help='Path to predicted ball positions CSV')
    parser.add_argument('--actual-ball', help='Path to actual ball positions CSV (if available)')
    parser.add_argument('--output-dir', default='animations', help='Directory to save output files')
    parser.add_argument('--period', type=int, default=1, help='Match period to visualize')
    parser.add_argument('--start-time', type=int, help='Start time for visualization')
    parser.add_argument('--end-time', type=int, help='End time for visualization')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for animation')
    parser.add_argument('--duration', type=int, default=20, help='Duration of animation in seconds')
    parser.add_argument('--no-trails', action='store_true', help='Disable movement trails')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = FootballAnimationVisualizer(
        home_data_path=args.home,
        away_data_path=args.away,
        predicted_ball_path=args.predicted_ball,
        actual_ball_path=args.actual_ball,
        output_dir=args.output_dir
    )
    
    # Create animation
    visualizer.create_animation(
        period=args.period,
        start_time=args.start_time,
        end_time=args.end_time,
        fps=args.fps,
        duration=args.duration,
        show_trails=not args.no_trails
    )
    
    # If actual ball data is available, create additional visualizations
    if args.actual_ball:
        # Create error heatmap
        visualizer.create_error_heatmap(period=args.period)
        
        # Create error analysis
        visualizer.create_error_analysis(period=args.period)
        
        # Create trajectory comparison if start and end times are specified
        if args.start_time is not None and args.end_time is not None:
            visualizer.create_trajectory_comparison(
                start_time=args.start_time,
                end_time=args.end_time,
                period=args.period
            )


if __name__ == "__main__":
    main()