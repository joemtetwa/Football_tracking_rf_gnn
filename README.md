# Football Ball Tracking Experiment

This project provides tools for tracking and analyzing football (soccer) ball movement using player tracking data. It includes components for predicting ball positions, validating predictions, and visualizing results.

## Project Overview

The experiment workflow consists of three main steps:

1. **Ball Position Prediction**: Using player tracking data to predict the ball's position
2. **Prediction Validation**: Validating the quality of ball position predictions
3. **Result Visualization**: Creating animations and visualizations to evaluate prediction accuracy

## Requirements

- Python 3.8+
- Required Python packages:
  - matplotlib
  - pandas
  - numpy
  - tqdm
  - scipy (for some analysis functions)

You can install all required packages using:

```
pip install matplotlib pandas numpy tqdm scipy
```

## Data Format

The experiment requires the following data files:

- **Home Team Tracking Data** (CSV): Contains position data for home team players
- **Away Team Tracking Data** (CSV): Contains position data for away team players
- **Predicted Ball Positions** (CSV): Contains the predicted x,y coordinates of the ball
- **Actual Ball Positions** (CSV, optional): Contains the ground truth ball positions for validation

The tracking data should have the following structure:
- Columns for each player: `home_[player_id]_x`, `home_[player_id]_y`, `away_[player_id]_x`, `away_[player_id]_y`
- Time information: `MatchId`, `IdPeriod`, `Time`

## Running the Experiment

### 1. Predicting Ball Positions

To predict ball positions using the default model:

```
python predict_ball.py --home Data/match_X/Home.csv --away Data/match_X/Away.csv --output predicted_ball_positions.csv
```

To use the GNN (Graph Neural Network) model:

```
python predict_with_gnn.py --home Data/match_X/Home.csv --away Data/match_X/Away.csv --output gnn_predictions/gnn_ball_predictions.csv
```

Replace `match_X` with the specific match folder you want to use (e.g., `match_1`).

### 2. Validating Predictions

To validate the quality of the ball position predictions:

```
python validate_predictions.py --predictions predicted_ball_positions.csv --output-dir validation_results
```

This will generate:
- A validation report in the `validation_results` directory
- A summary plot with validation metrics
- CSV files with detailed validation data

### 3. Visualizing Results

To create visualizations of the tracking data and ball predictions:

```
python create_visualizations.py --home Data/match_X/Home.csv --away Data/match_X/Away.csv --predicted-ball predicted_ball_positions.csv --output-dir football_visualizations
```

If you have actual ball data for comparison:

```
python create_visualizations.py --home Data/match_X/Home.csv --away Data/match_X/Away.csv --predicted-ball predicted_ball_positions.csv --actual-ball actual_ball_positions.csv --output-dir football_visualizations
```

Additional visualization options:
- `--period`: Match period to visualize (default: 1)
- `--start-time`: Start time for visualization
- `--end-time`: End time for visualization
- `--fps`: Frames per second for animation (default: 10)
- `--duration`: Duration of animation in seconds (default: 20)
- `--no-trails`: Disable movement trails

### 4. All-in-One Visualization

For convenience, you can directly use the animation visualizer:

```
python football_animation_visualizer.py --home Data/match_X/Home.csv --away Data/match_X/Away.csv --predicted-ball predicted_ball_positions.csv --output-dir football_visualizations
```

## Understanding the Results

### Validation Metrics

The validation checks look for several types of issues:
- Missing values in the predicted data
- Out-of-bounds ball positions
- Unrealistically large position jumps
- Unrealistic ball speeds

### Visualization Types

1. **Animations**: Show player and ball movement across the pitch
2. **Error Heatmaps**: Visualize spatial distribution of prediction errors
3. **Trajectory Comparisons**: Compare predicted vs. actual ball paths
4. **Error Analysis**: Charts showing error distribution and correlations with ball speed

## Examples

### Basic Example

To run a complete example using match 1 data:

```
# Step 1: Predict ball positions
python predict_ball.py --home Data/match_1/Home.csv --away Data/match_1/Away.csv --output predicted_ball_positions.csv

# Step 2: Validate predictions
python validate_predictions.py --predictions predicted_ball_positions.csv

# Step 3: Visualize results
python football_animation_visualizer.py --home Data/match_1/Home.csv --away Data/match_1/Away.csv --predicted-ball predicted_ball_positions.csv
```

## File Structure

- `predict_ball.py`: Main script for ball position prediction
- `predict_with_gnn.py`: Alternative prediction using Graph Neural Networks
- `ball_predictions_validator.py`: Validation logic for predicted positions
- `validate_predictions.py`: Script to run validation
- `football_animation_visualizer.py`: Visualization tools
- `create_visualizations.py`: Script to generate all visualizations
- `Data/`: Directory containing match tracking data
- `football_visualizations/`: Output directory for visualizations
- `validation_results/`: Output directory for validation reports
