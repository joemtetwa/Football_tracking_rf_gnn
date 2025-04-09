"""
Usage script for the improved Football Ball Predictor with machine learning
"""

from football_tracking_solution import FootballBallPredictor

def predict_ball_positions():
    """
    Train on matches with ball data and predict ball positions for match 4.
    """
    # Initialize the predictor with custom parameters
    predictor = FootballBallPredictor(
        velocity_window=3,    # Window size for velocity calculation (frames)
        smoothing_window=5    # Window size for smoothing the trajectory
    )
    
    # Training data paths (matches with ball coordinates)
    # Assuming the first two matches have ball coordinates
    training_data = [
        ('Data\match_1\Home.csv', 'Data\match_1\Away.csv'),  # First match with ball coordinates
        ('Data\match_2\Home.csv', 'Data\match_2\Away.csv'),
        ('Data\match_3\Home.csv', 'Data\match_3\Away.csv')   # Second match with ball coordinates
    ]
    
    # Test data paths (match without ball coordinates - match 4)
    test_data = ('Data\match_4\Home.csv', 'Data\match_4\Away.csv')
    
    # Output path for predictions
    output_path = 'predicted_ball_positions.csv'
    
    # Run the training and prediction
    predictions = predictor.train_and_predict(
        training_data_paths=training_data,
        test_data_paths=test_data,
        output_path=output_path
    )
    
    print("\nPrediction complete!")
    print(f"Results saved to: {output_path}")
    print("\nSample predictions:")
    print(predictions.head(10))
    
    return predictions


if __name__ == "__main__":
    predict_ball_positions()