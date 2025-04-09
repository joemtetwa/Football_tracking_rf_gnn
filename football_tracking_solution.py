"""
Football Tracking Data - Ball Position Prediction
------------------------------------------------
This solution predicts ball positions based on player tracking data.
It trains on matches with known ball coordinates and then
predicts ball positions for matches without ball data.
"""

import pandas as pd
import numpy as np
import re
import os
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class FootballBallPredictor:
    """
    Class for predicting ball positions from football tracking data.
    Trains on matches with known ball coordinates and predicts for new matches.
    """
    
    def __init__(self, velocity_window=3, smoothing_window=5):
        """Initialize the predictor with parameters for velocity calculation and smoothing."""
        self.velocity_window = velocity_window
        self.smoothing_window = smoothing_window
        self.x_model = None  # Model for predicting x-coordinates
        self.y_model = None  # Model for predicting y-coordinates
        self.scaler = None   # Feature scaler
    
    def train_and_predict(self, training_data_paths, test_data_paths, output_path=None):
        """
        Train on matches with ball data and predict for matches without ball data.
        
        Args:
            training_data_paths: List of tuples (home_path, away_path) for training matches
            test_data_paths: Tuple (home_path, away_path) for the test match
            output_path: Path to save the output predictions (optional)
            
        Returns:
            DataFrame with ball position predictions for the test match
        """
        print("Starting ball coordinate prediction pipeline...")
        
        # Process training data
        print("Processing training data...")
        training_features = []
        training_ball_x = []
        training_ball_y = []
        
        for i, (home_path, away_path) in enumerate(training_data_paths):
            print(f"Processing training match {i+1}/{len(training_data_paths)}")
            
            # Load match data - handle both csv and xlsx formats
            home_data = pd.read_csv(home_path) if home_path.lower().endswith('.csv') else pd.read_excel(home_path)
            away_data = pd.read_csv(away_path) if away_path.lower().endswith('.csv') else pd.read_excel(away_path)
            
            # Check if ball coordinates are present
            if 'ball_x' not in home_data.columns or 'ball_y' not in home_data.columns:
                print(f"Warning: Ball coordinates not found in training match {i+1}")
                continue
            
            # Extract features and targets
            match_features, match_ball_x, match_ball_y = self._process_training_match(home_data, away_data)
            
            training_features.extend(match_features)
            training_ball_x.extend(match_ball_x)
            training_ball_y.extend(match_ball_y)
        
        print(f"Collected {len(training_features)} training samples")
        
        # Train the model
        print("Training ball position prediction models...")
        self._train_models(training_features, training_ball_x, training_ball_y)
        
        # Process test data and predict
        print("Predicting ball positions for test match...")
        home_test = pd.read_csv(test_data_paths[0]) if test_data_paths[0].lower().endswith('.csv') else pd.read_excel(test_data_paths[0])
        away_test = pd.read_csv(test_data_paths[1]) if test_data_paths[1].lower().endswith('.csv') else pd.read_excel(test_data_paths[1])
        
        predictions = self._predict_test_match(home_test, away_test)
        
        # Apply temporal smoothing
        print("Applying temporal smoothing to ball trajectory...")
        smoothed_predictions = self._smooth_ball_trajectory(predictions)
        
        print(f"Ball position prediction complete. Generated {len(smoothed_predictions)} predictions")
        
        # Save to CSV if output path is provided
        if output_path:
            smoothed_predictions.to_csv(output_path, index=False)
            print(f"Ball position predictions written to {output_path}")
        
        return smoothed_predictions
    
    def _process_training_match(self, home_data, away_data):
        """Process a training match to extract features and ball positions."""
        # Extract player IDs
        home_player_ids, away_player_ids = self._extract_player_ids(home_data, away_data)
        print(f"Found {len(home_player_ids)} home players and {len(away_player_ids)} away players")
        
        # Calculate velocities
        home_velocities = self._calculate_velocities(home_data, home_player_ids)
        away_velocities = self._calculate_velocities(away_data, away_player_ids)
        
        # Extract features and targets for each frame
        features = []
        ball_x_values = []
        ball_y_values = []
        
        num_frames = min(len(home_data), len(away_data))
        sample_rate = max(1, num_frames // 10000)  # Sample at most 10,000 frames
        
        for i in range(0, num_frames, sample_rate):
            # Extract features for this frame
            frame_features = self._extract_frame_features(
                i, home_data, away_data, home_velocities, away_velocities, 
                home_player_ids, away_player_ids
            )
            
            # Get actual ball position
            ball_x = home_data.loc[i, 'ball_x']
            ball_y = home_data.loc[i, 'ball_y']
            
            # Only include frames with valid ball positions
            if not pd.isna(ball_x) and not pd.isna(ball_y):
                features.append(frame_features)
                ball_x_values.append(ball_x)
                ball_y_values.append(ball_y)
        
        return features, ball_x_values, ball_y_values
    
    def _extract_frame_features(self, time_index, home_data, away_data, 
                             home_velocities, away_velocities, 
                             home_player_ids, away_player_ids):
        """Extract features from a single frame for model training or prediction."""
        # Extract player data
        home_players = self._extract_player_data(home_data, home_velocities, time_index, home_player_ids)
        away_players = self._extract_player_data(away_data, away_velocities, time_index, away_player_ids)
        
        # Calculate team statistics
        home_stats = self._calculate_team_stats(home_players)
        away_stats = self._calculate_team_stats(away_players)
        
        # Create feature vector
        features = []
        
        # Add team centroid positions
        features.extend([
            home_stats['centroid']['x'], home_stats['centroid']['y'],
            away_stats['centroid']['x'], away_stats['centroid']['y']
        ])
        
        # Add team spread and speed
        features.extend([
            home_stats['spread'], home_stats['avg_speed'],
            away_stats['spread'], away_stats['avg_speed']
        ])
        
        # Add positions and velocities of fastest players (up to 3 from each team)
        home_sorted = sorted(home_players, key=lambda p: p['speed'], reverse=True)
        away_sorted = sorted(away_players, key=lambda p: p['speed'], reverse=True)
        
        # Add data for up to 3 fastest home players
        for i in range(min(3, len(home_sorted))):
            features.extend([
                home_sorted[i]['x'], home_sorted[i]['y'],
                home_sorted[i]['vx'], home_sorted[i]['vy'],
                home_sorted[i]['speed']
            ])
        
        # Pad if fewer than 3 players
        for i in range(3 - min(3, len(home_sorted))):
            features.extend([0, 0, 0, 0, 0])
        
        # Add data for up to 3 fastest away players
        for i in range(min(3, len(away_sorted))):
            features.extend([
                away_sorted[i]['x'], away_sorted[i]['y'],
                away_sorted[i]['vx'], away_sorted[i]['vy'],
                away_sorted[i]['speed']
            ])
        
        # Pad if fewer than 3 players
        for i in range(3 - min(3, len(away_sorted))):
            features.extend([0, 0, 0, 0, 0])
        
        # Add distance between closest opponents
        closest_dist = float('inf')
        
        for home_player in home_players:
            for away_player in away_players:
                dist = np.sqrt((home_player['x'] - away_player['x'])**2 + 
                              (home_player['y'] - away_player['y'])**2)
                if dist < closest_dist:
                    closest_dist = dist
        
        features.append(closest_dist if closest_dist != float('inf') else 1000)
        
        return features
    
    def _train_models(self, features, ball_x, ball_y):
        """Train machine learning models to predict ball x and y coordinates."""
        if len(features) == 0:
            raise ValueError("No training data available")
        
        # Convert to numpy arrays
        X = np.array(features)
        y_x = np.array(ball_x)
        y_y = np.array(ball_y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest models
        print("Training ball_x prediction model...")
        self.x_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.x_model.fit(X_scaled, y_x)
        
        print("Training ball_y prediction model...")
        self.y_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.y_model.fit(X_scaled, y_y)
        
        print("Model training complete")
    
    def _predict_test_match(self, home_data, away_data):
        """Predict ball positions for a test match using the trained models."""
        # Check if models are trained
        if self.x_model is None or self.y_model is None:
            raise ValueError("Models not trained. Call train_and_predict first.")
        
        # Extract player IDs
        home_player_ids, away_player_ids = self._extract_player_ids(home_data, away_data)
        print(f"Found {len(home_player_ids)} home players and {len(away_player_ids)} away players")
        
        # Calculate velocities
        home_velocities = self._calculate_velocities(home_data, home_player_ids)
        away_velocities = self._calculate_velocities(away_data, away_player_ids)
        
        # Generate predictions for each frame
        predictions = []
        num_frames = min(len(home_data), len(away_data))
        
        # Process in batches to show progress
        batch_size = 1000
        num_batches = (num_frames + batch_size - 1) // batch_size
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, num_frames)
            
            print(f"Processing batch {batch+1}/{num_batches} (frames {start_idx} to {end_idx-1})...")
            
            for i in range(start_idx, end_idx):
                # Extract features
                features = self._extract_frame_features(
                    i, home_data, away_data, home_velocities, away_velocities,
                    home_player_ids, away_player_ids
                )
                
                # Scale features
                scaled_features = self.scaler.transform([features])
                
                # Predict ball position
                ball_x = self.x_model.predict(scaled_features)[0]
                ball_y = self.y_model.predict(scaled_features)[0]
                
                predictions.append({
                    'MatchId': home_data.loc[i, 'MatchId'],
                    'IdPeriod': home_data.loc[i, 'IdPeriod'],
                    'Time': home_data.loc[i, 'Time'],
                    'ball_x': ball_x,
                    'ball_y': ball_y
                })
        
        return pd.DataFrame(predictions)
    
    def _extract_player_ids(self, home_data, away_data):
        """Extract player IDs from the column names in the tracking data."""
        # Extract player columns (exclude ball columns)
        home_player_cols = [col for col in home_data.columns 
                          if ('_x' in col or '_y' in col) and 'ball' not in col]
        away_player_cols = [col for col in away_data.columns 
                          if ('_x' in col or '_y' in col) and 'ball' not in col]
        
        # Extract unique player IDs
        home_player_ids = set()
        for col in home_player_cols:
            match = re.match(r'(home_\d+)_[xy]', col)
            if match:
                home_player_ids.add(match.group(1))
        
        away_player_ids = set()
        for col in away_player_cols:
            match = re.match(r'(away_\d+)_[xy]', col)
            if match:
                away_player_ids.add(match.group(1))
        
        return sorted(list(home_player_ids)), sorted(list(away_player_ids))
    
    def _calculate_velocities(self, data, player_ids):
        """Calculate player velocities based on position changes over time."""
        velocities = pd.DataFrame(index=data.index)
        velocities['MatchId'] = data['MatchId']
        velocities['IdPeriod'] = data['IdPeriod']
        velocities['Time'] = data['Time']
        
        window = self.velocity_window
        
        for player_id in player_ids:
            x_col = f"{player_id}_x"
            y_col = f"{player_id}_y"
            vx_col = f"{player_id}_vx"
            vy_col = f"{player_id}_vy"
            speed_col = f"{player_id}_speed"
            
            # Initialize velocity columns
            velocities[vx_col] = 0.0
            velocities[vy_col] = 0.0
            velocities[speed_col] = 0.0
            
            # Calculate velocities for each frame after the window
            for i in range(window, len(data)):
                current_x = data.loc[i, x_col]
                current_y = data.loc[i, y_col]
                past_x = data.loc[i - window, x_col]
                past_y = data.loc[i - window, y_col]
                
                # Skip if any position is null
                if pd.isna(current_x) or pd.isna(current_y) or pd.isna(past_x) or pd.isna(past_y):
                    continue
                
                # Time difference in seconds (assuming Time is in deciseconds)
                time_change = (data.loc[i, 'Time'] - data.loc[i - window, 'Time']) / 10
                
                # Handle possible time discontinuities
                if time_change <= 0:
                    continue
                
                vx = (current_x - past_x) / time_change
                vy = (current_y - past_y) / time_change
                speed = np.sqrt(vx**2 + vy**2)
                
                velocities.loc[i, vx_col] = vx
                velocities.loc[i, vy_col] = vy
                velocities.loc[i, speed_col] = speed
        
        return velocities
    
    def _extract_player_data(self, data, velocities, time_index, player_ids):
        """Extract valid player positions and attributes for a specific time frame."""
        players = []
        
        for player_id in player_ids:
            x_col = f"{player_id}_x"
            y_col = f"{player_id}_y"
            vx_col = f"{player_id}_vx"
            vy_col = f"{player_id}_vy"
            speed_col = f"{player_id}_speed"
            
            # Check if position data exists
            if pd.isna(data.loc[time_index, x_col]) or pd.isna(data.loc[time_index, y_col]):
                continue
            
            # Get position and velocity data
            x = data.loc[time_index, x_col]
            y = data.loc[time_index, y_col]
            vx = velocities.loc[time_index, vx_col]
            vy = velocities.loc[time_index, vy_col]
            speed = velocities.loc[time_index, speed_col]
            
            players.append({
                'id': player_id,
                'x': x,
                'y': y,
                'vx': vx,
                'vy': vy,
                'speed': speed
            })
        
        return players
    
    def _calculate_team_stats(self, players):
        """Calculate team formation statistics based on player positions."""
        if len(players) == 0:
            return {
                'centroid': {'x': 0, 'y': 0},
                'avg_speed': 0,
                'spread': 0,
                'players': []
            }
        
        # Calculate centroid
        sum_x = sum(p['x'] for p in players)
        sum_y = sum(p['y'] for p in players)
        centroid_x = sum_x / len(players)
        centroid_y = sum_y / len(players)
        
        # Calculate average speed
        avg_speed = sum(p['speed'] for p in players) / len(players)
        
        # Calculate team spread (average distance from centroid)
        spread = sum(np.sqrt((p['x'] - centroid_x)**2 + (p['y'] - centroid_y)**2) 
                    for p in players) / len(players)
        
        return {
            'centroid': {'x': centroid_x, 'y': centroid_y},
            'avg_speed': avg_speed,
            'spread': spread,
            'players': players
        }
    
    def _smooth_ball_trajectory(self, predictions):
        """Apply temporal smoothing to predicted ball positions."""
        # Create a copy of the predictions
        smoothed_predictions = predictions.copy()
        
        # Apply Gaussian smoothing to x and y coordinates
        sigma = self.smoothing_window / 3  # Convert window to sigma for Gaussian filter
        smoothed_predictions['ball_x'] = gaussian_filter1d(predictions['ball_x'].values, sigma)
        smoothed_predictions['ball_y'] = gaussian_filter1d(predictions['ball_y'].values, sigma)
        
        return smoothed_predictions


# Example usage script
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict ball positions from football tracking data')
    parser.add_argument('--train-home', nargs='+', required=True, 
                        help='Paths to home team tracking data CSVs for training')
    parser.add_argument('--train-away', nargs='+', required=True, 
                        help='Paths to away team tracking data CSVs for training')
    parser.add_argument('--test-home', required=True, help='Path to home team tracking data CSV for testing')
    parser.add_argument('--test-away', required=True, help='Path to away team tracking data CSV for testing')
    parser.add_argument('--output', required=True, help='Path to save output predictions CSV')
    parser.add_argument('--velocity-window', type=int, default=3, 
                        help='Window size for velocity calculation (default: 3)')
    parser.add_argument('--smoothing-window', type=int, default=5, 
                        help='Window size for trajectory smoothing (default: 5)')
    
    args = parser.parse_args()
    
    # Check if train home and away have the same number of files
    if len(args.train_home) != len(args.train_away):
        raise ValueError("Number of training home and away files must match")
    
    # Create training data pairs
    training_data = list(zip(args.train_home, args.train_away))
    test_data = (args.test_home, args.test_away)
    
    predictor = FootballBallPredictor(
        velocity_window=args.velocity_window,
        smoothing_window=args.smoothing_window
    )
    
    predictions = predictor.train_and_predict(
        training_data_paths=training_data,
        test_data_paths=test_data,
        output_path=args.output
    )
    
    print(f"Ball prediction complete. Sample predictions:")
    print(predictions.head())


if __name__ == "__main__":
    main()