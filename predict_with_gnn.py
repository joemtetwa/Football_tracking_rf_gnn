"""
Usage script for the Graph Neural Network Ball Position Predictor
"""

from gnn_football_tracking import FootballBallPredictorGNN
import os

def predict_ball_positions_with_gnn():
    """
    Predict ball positions for football tracking data using Graph Neural Networks.
    """
    # Create output directory if it doesn't exist
    output_dir = 'gnn_predictions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define paths to data files
    # Training data (with ball coordinates)
    training_data = [
        ('Data\match_1\Home.csv', 'Data\match_1\Away.csv'), 
        ('Data\match_2\Home.csv', 'Data\match_2\Away.csv'),
        ('Data\match_3\Home.csv', 'Data\match_3\Away.csv')
    ]
    
    # Test data (without ball coordinates)
    test_data = ('Data\match_4\Home.csv', 'Data\match_4\Away.csv')
    
    # Output path for predictions
    output_path = os.path.join(output_dir, 'gnn_ball_predictions.csv')
    
    print(f"Starting GNN-based ball position prediction")
    print(f"Using {len(training_data)} training matches")
    print(f"Results will be saved to {output_path}")
    
    # Initialize the GNN predictor with custom parameters
    predictor = FootballBallPredictorGNN(
        velocity_window=3,         # Window size for velocity calculation
        smoothing_window=5,        # Window size for trajectory smoothing
        edge_threshold=2000,       # Distance threshold for edge creation (player connections)
        hidden_channels=64,        # Size of hidden layers in GNN
        node_features=7,           # Number of features per player node
        learning_rate=0.001,       # Learning rate for optimization
        epochs=30,                 # Number of training epochs
        batch_size=128             # Batch size for training
    )
    
    # Run training and prediction
    predictions = predictor.train_and_predict(
        training_data_paths=training_data,
        test_data_paths=test_data,
        output_path=output_path
    )
    
    print("\nPrediction complete!")
    print(f"Generated {len(predictions)} ball position predictions")
    print(f"Results saved to: {output_path}")
    print("\nSample predictions:")
    print(predictions.head(10))
    
    return predictions


if __name__ == "__main__":
    # Execute the prediction function
    predict_ball_positions_with_gnn()
    
    print("\nTo compare with the original Random Forest model:")
    print("python football_tracking_solution.py --train-home Home_match1.csv Home_match2.csv --train-away Away_match1.csv Away_match2.csv --test-home Home.csv --test-away Away.csv --output rf_ball_predictions.csv")