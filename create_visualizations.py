"""
Simple usage script for the Football Animation Visualizer
"""

from football_animation_visualizer import FootballAnimationVisualizer
import os

def create_visualizations():
    """
    Create visualizations for football tracking data with predicted ball positions.
    """
    # Create output directory if it doesn't exist
    output_dir = 'football_visualizations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define paths to data files
    home_data_path = 'Data\match_4\Home.csv'  # Home team tracking data
    away_data_path = 'Data\match_4\Away.csv'  # Away team tracking data
    predicted_ball_path = 'gnn_predictions\gnn_ball_predictions.csv'  # Predicted ball positions
    
    # If you have actual ball positions, uncomment this line:
    # actual_ball_path = 'actual_ball_positions.csv'
    actual_ball_path = None  # Set to None if you don't have actual ball data
    
    print(f"Creating football tracking visualizations...")
    print(f"Using home data: {home_data_path}")
    print(f"Using away data: {away_data_path}")
    print(f"Using predicted ball data: {predicted_ball_path}")
    if actual_ball_path:
        print(f"Using actual ball data: {actual_ball_path}")
    print(f"Output will be saved to: {output_dir}")
    
    # Initialize the visualizer
    visualizer = FootballAnimationVisualizer(
        home_data_path=home_data_path,
        away_data_path=away_data_path,
        predicted_ball_path=predicted_ball_path,
        actual_ball_path=actual_ball_path,
        output_dir=output_dir
    )
    
    # Create an animation for the first period, for the first 1000 time units 
    # (adjust based on your data)
    animation_file = visualizer.create_animation(
        period=1,              # Match period to visualize
        start_time=0,          # Starting time (set to None to start from beginning)
        end_time=2000,         # Ending time (set to None to go to the end)
        fps=15,                # Frames per second in the animation
        duration=20,           # Duration of the animation in seconds
        show_trails=True,      # Whether to show movement trails
        trail_length=15        # Length of movement trails in frames
    )
    
    print(f"Animation created: {animation_file}")
    
    # Create a static trajectory visualization for a specific play
    trajectory_file = visualizer.create_trajectory_comparison(
        start_time=500,        # Start time of the play
        end_time=2000,          # End time of the play
        period=1               # Match period
    )
    
    if trajectory_file:
        print(f"Trajectory visualization created: {trajectory_file}")
    
    # If you have actual ball data, create error analysis visualizations
    if actual_ball_path:
        # Create error heatmap
        heatmap_file = visualizer.create_error_heatmap(period=1)
        print(f"Error heatmap created: {heatmap_file}")
        
        # Create detailed error analysis
        analysis_file = visualizer.create_error_analysis(period=1)
        print(f"Error analysis created: {analysis_file}")
    
    print("\nAll visualizations complete!")
    print(f"Check the {output_dir} directory for all output files.")


if __name__ == "__main__":
    create_visualizations()