"""
Football Tracking Data - Ball Position Prediction with Graph Neural Networks
---------------------------------------------------------------------------
This solution predicts ball positions based on player tracking data using GNNs.
Players are represented as nodes in a dynamic graph with spatial relationships.
"""

import pandas as pd
import numpy as np
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


class FootballGNN(nn.Module):
    """
    Graph Neural Network for predicting ball positions in football.
    """
    def __init__(self, node_features, hidden_channels=64):
        super(FootballGNN, self).__init__()
        # GNN layers
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Output regression layers for predicting ball position
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 2)  # 2 outputs: ball_x and ball_y
    
    def forward(self, data):
        # Node features
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolution layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Fully connected layers for prediction
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Output layer (ball coordinates)
        ball_pos = self.fc2(x)
        
        return ball_pos


class FootballBallPredictorGNN:
    """
    Class for predicting ball positions from football tracking data using GNNs.
    Trains on matches with known ball coordinates and predicts for new matches.
    """
    
    def __init__(self, velocity_window=3, smoothing_window=5, edge_threshold=2000, 
                 hidden_channels=64, node_features=7, learning_rate=0.001, epochs=50, 
                 batch_size=128):
        """
        Initialize the GNN-based predictor with parameters.
        
        Args:
            velocity_window: Window size for calculating player velocities
            smoothing_window: Window size for smoothing ball trajectories
            edge_threshold: Distance threshold for creating edges between players (spatial proximity)
            hidden_channels: Number of hidden channels in GNN
            node_features: Number of features per player node
            learning_rate: Learning rate for the optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.velocity_window = velocity_window
        self.smoothing_window = smoothing_window
        self.edge_threshold = edge_threshold  # Units for edge creation between players
        self.hidden_channels = hidden_channels
        self.node_features = node_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Model and scaler will be initialized during training
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
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
        print("Starting GNN ball coordinate prediction pipeline...")
        
        # Process training data to create graph datasets
        print("Processing training data and creating graph structures...")
        train_graphs = []
        ball_positions = []
        
        for i, (home_path, away_path) in enumerate(training_data_paths):
            print(f"Processing training match {i+1}/{len(training_data_paths)}")
            
            # Load match data - handle both csv and xlsx formats
            home_data = pd.read_csv(home_path) if home_path.lower().endswith('.csv') else pd.read_excel(home_path)
            away_data = pd.read_csv(away_path) if away_path.lower().endswith('.csv') else pd.read_excel(away_path)
            
            # Check if ball coordinates are present
            if 'ball_x' not in home_data.columns or 'ball_y' not in home_data.columns:
                print(f"Warning: Ball coordinates not found in training match {i+1}")
                continue
            
            # Extract features and create graphs
            match_graphs, match_ball_positions = self._process_training_match_to_graphs(
                home_data, away_data
            )
            
            train_graphs.extend(match_graphs)
            ball_positions.extend(match_ball_positions)
        
        print(f"Created {len(train_graphs)} training graphs")
        
        # Train the GNN model
        print("Training GNN model...")
        self._train_gnn_model(train_graphs, ball_positions)
        
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
    
    def _process_training_match_to_graphs(self, home_data, away_data):
        """Process a training match to create graph structures and extract ball positions."""
        # Extract player IDs
        home_player_ids, away_player_ids = self._extract_player_ids(home_data, away_data)
        print(f"Found {len(home_player_ids)} home players and {len(away_player_ids)} away players")
        
        # Calculate velocities
        home_velocities = self._calculate_velocities(home_data, home_player_ids)
        away_velocities = self._calculate_velocities(away_data, away_player_ids)
        
        # Create graphs and extract ball positions
        graphs = []
        ball_positions = []
        
        num_frames = min(len(home_data), len(away_data))
        sample_rate = max(1, num_frames // 10000)  # Sample at most 10,000 frames
        
        for i in range(0, num_frames, sample_rate):
            # Extract player data
            home_players = self._extract_player_data(home_data, home_velocities, i, home_player_ids)
            away_players = self._extract_player_data(away_data, away_velocities, i, away_player_ids)
            
            # Only proceed if we have enough players
            if len(home_players) + len(away_players) < 5:
                continue
            
            # Get actual ball position
            ball_x = home_data.loc[i, 'ball_x']
            ball_y = home_data.loc[i, 'ball_y']
            
            # Only include frames with valid ball positions
            if not pd.isna(ball_x) and not pd.isna(ball_y):
                # Create graph
                graph = self._create_player_graph(home_players, away_players)
                
                if graph is not None:
                    graphs.append(graph)
                    ball_positions.append([ball_x, ball_y])
        
        return graphs, ball_positions
    
    def _create_player_graph(self, home_players, away_players):
        """
        Create a graph from player positions with spatial relationships.
        
        Args:
            home_players: List of home team player data
            away_players: List of away team player data
            
        Returns:
            torch_geometric.data.Data: Graph representation
        """
        # Combine all players
        all_players = home_players + away_players
        
        if len(all_players) == 0:
            return None
        
        # Create node features
        node_features = []
        for player in all_players:
            # Features: [x, y, vx, vy, speed, is_home_team, distance_from_center]
            is_home = 1.0 if 'home' in player['id'] else 0.0
            
            # Distance from field center
            dist_from_center = np.sqrt(player['x']**2 + player['y']**2)
            
            # Create feature vector
            features = [
                player['x'], 
                player['y'], 
                player['vx'], 
                player['vy'], 
                player['speed'],
                is_home,
                dist_from_center
            ]
            
            node_features.append(features)
        
        # Create edges based on spatial proximity
        edges = []
        for i in range(len(all_players)):
            for j in range(i + 1, len(all_players)):
                player_i = all_players[i]
                player_j = all_players[j]
                
                # Calculate distance between players
                dist = np.sqrt((player_i['x'] - player_j['x'])**2 + 
                              (player_i['y'] - player_j['y'])**2)
                
                # Create edge if players are close enough
                if dist < self.edge_threshold:
                    # Add edges in both directions (undirected graph)
                    edges.append([i, j])
                    edges.append([j, i])
        
        # Convert to pytorch tensors
        if len(edges) == 0:
            # If no edges, create self-loops for all nodes
            edges = [[i, i] for i in range(len(all_players))]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create graph data object
        data = Data(x=x, edge_index=edge_index)
        
        return data
    
    def _train_gnn_model(self, train_graphs, ball_positions):
        """
        Train the GNN model on training graphs.
        
        Args:
            train_graphs: List of graph data objects
            ball_positions: List of [x, y] ball positions
        """
        # Initialize model
        self.model = FootballGNN(
            node_features=self.node_features,
            hidden_channels=self.hidden_channels
        ).to(self.device)
        
        # Convert ball positions to tensor
        ball_positions_tensor = torch.tensor(ball_positions, dtype=torch.float)
        
        # Scale ball positions
        ball_positions_np = ball_positions_tensor.numpy()
        self.scaler.fit(ball_positions_np)
        ball_positions_scaled = torch.tensor(
            self.scaler.transform(ball_positions_np), 
            dtype=torch.float
        )
        
        # Create dataset
        dataset = list(zip(train_graphs, ball_positions_scaled))
        
        # Set up optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            
            # Process in batches
            np.random.shuffle(dataset)
            num_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
            
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(dataset))
                batch_data = dataset[start_idx:end_idx]
                
                # Separate graphs and targets
                batch_graphs = [item[0] for item in batch_data]
                batch_targets = torch.stack([item[1] for item in batch_data]).to(self.device)
                
                # Create batch
                batch = Batch.from_data_list(batch_graphs).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                output = self.model(batch)
                
                # Compute loss
                loss = criterion(output, batch_targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(batch_data)
            
            avg_loss = total_loss / len(dataset)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        # Set to evaluation mode
        self.model.eval()
    
    def _predict_test_match(self, home_data, away_data):
        """
        Predict ball positions for a test match using the trained GNN model.
        
        Args:
            home_data: Home team tracking data
            away_data: Away team tracking data
            
        Returns:
            DataFrame with predicted ball positions
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train_and_predict first.")
        
        # Extract player IDs
        home_player_ids, away_player_ids = self._extract_player_ids(home_data, away_data)
        print(f"Found {len(home_player_ids)} home players and {len(away_player_ids)} away players")
        
        # Calculate velocities
        home_velocities = self._calculate_velocities(home_data, home_player_ids)
        away_velocities = self._calculate_velocities(away_data, away_player_ids)
        
        # Generate predictions
        predictions = []
        num_frames = min(len(home_data), len(away_data))
        
        # Process in batches
        batch_size = 1000
        num_batches = (num_frames + batch_size - 1) // batch_size
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, num_frames)
            
            print(f"Processing batch {batch+1}/{num_batches} (frames {start_idx} to {end_idx-1})...")
            
            batch_predictions = []
            batch_meta = []
            
            for i in range(start_idx, end_idx):
                # Extract player data
                home_players = self._extract_player_data(home_data, home_velocities, i, home_player_ids)
                away_players = self._extract_player_data(away_data, away_velocities, i, away_player_ids)
                
                # Create graph if we have enough players
                if len(home_players) + len(away_players) >= 5:
                    graph = self._create_player_graph(home_players, away_players)
                    
                    if graph is not None:
                        batch_predictions.append(graph)
                        batch_meta.append({
                            'MatchId': home_data.loc[i, 'MatchId'],
                            'IdPeriod': home_data.loc[i, 'IdPeriod'],
                            'Time': home_data.loc[i, 'Time'],
                            'index': i
                        })
                else:
                    # Not enough players, use a placeholder prediction
                    batch_meta.append({
                        'MatchId': home_data.loc[i, 'MatchId'],
                        'IdPeriod': home_data.loc[i, 'IdPeriod'],
                        'Time': home_data.loc[i, 'Time'],
                        'index': i,
                        'placeholder': True
                    })
            
            # Make predictions for the batch if we have valid graphs
            if batch_predictions:
                # Set model to evaluation mode
                self.model.eval()
                
                with torch.no_grad():
                    # Process in smaller sub-batches to avoid memory issues
                    sub_batch_size = 128
                    num_sub_batches = (len(batch_predictions) + sub_batch_size - 1) // sub_batch_size
                    
                    batch_outputs = []
                    
                    for j in range(num_sub_batches):
                        sub_start = j * sub_batch_size
                        sub_end = min(sub_start + sub_batch_size, len(batch_predictions))
                        sub_batch = batch_predictions[sub_start:sub_end]
                        
                        # Create batch
                        data_batch = Batch.from_data_list(sub_batch).to(self.device)
                        
                        # Forward pass
                        output = self.model(data_batch)
                        
                        # Convert to numpy and inverse transform
                        output_np = output.cpu().numpy()
                        output_unscaled = self.scaler.inverse_transform(output_np)
                        
                        batch_outputs.append(output_unscaled)
                    
                    # Combine all sub-batch outputs
                    all_outputs = np.vstack(batch_outputs)
            
            # Create prediction records
            pred_index = 0
            for meta in batch_meta:
                if 'placeholder' in meta:
                    # Use interpolation or a default position for frames with insufficient data
                    # Here we use a simplistic approach - using (0,0) as default
                    predictions.append({
                        'MatchId': meta['MatchId'],
                        'IdPeriod': meta['IdPeriod'],
                        'Time': meta['Time'],
                        'ball_x': 0.0,
                        'ball_y': 0.0
                    })
                else:
                    # Use predicted position
                    predictions.append({
                        'MatchId': meta['MatchId'],
                        'IdPeriod': meta['IdPeriod'],
                        'Time': meta['Time'],
                        'ball_x': all_outputs[pred_index, 0],
                        'ball_y': all_outputs[pred_index, 1]
                    })
                    pred_index += 1
        
        # Post-process to ensure predictions for all frames
        all_frames = pd.DataFrame({
            'MatchId': home_data['MatchId'],
            'IdPeriod': home_data['IdPeriod'],
            'Time': home_data['Time']
        })
        
        pred_df = pd.DataFrame(predictions)
        
        # Merge to ensure all frames are included
        final_predictions = pd.merge(
            all_frames, pred_df, on=['MatchId', 'IdPeriod', 'Time'], how='left'
        )
        
        # Fill missing values (if any) using forward/backward fill
        final_predictions['ball_x'] = final_predictions['ball_x'].fillna(method='ffill').fillna(method='bfill')
        final_predictions['ball_y'] = final_predictions['ball_y'].fillna(method='ffill').fillna(method='bfill')
        
        return final_predictions
    
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
    
    parser = argparse.ArgumentParser(description='Predict ball positions using GNN model')
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
    parser.add_argument('--edge-threshold', type=float, default=2000, 
                        help='Distance threshold for creating edges between players (default: 2000)')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--hidden-channels', type=int, default=64, 
                        help='Number of hidden channels in GNN (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.001, 
                        help='Learning rate for optimizer (default: 0.001)')
    
    args = parser.parse_args()
    
    # Check if train home and away have the same number of files
    if len(args.train_home) != len(args.train_away):
        raise ValueError("Number of training home and away files must match")
    
    # Create training data pairs
    training_data = list(zip(args.train_home, args.train_away))
    test_data = (args.test_home, args.test_away)
    
    predictor = FootballBallPredictorGNN(
        velocity_window=args.velocity_window,
        smoothing_window=args.smoothing_window,
        edge_threshold=args.edge_threshold,
        hidden_channels=args.hidden_channels,
        learning_rate=args.learning_rate,
        epochs=args.epochs
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