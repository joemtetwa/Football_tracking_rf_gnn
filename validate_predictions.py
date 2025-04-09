"""
Simple usage script for the Ball Prediction Validator
"""

from ball_predictions_validator import BallPredictionValidator

def validate_predictions():
    """
    Run validation on the predicted ball positions.
    """
    # Path to the predicted ball positions CSV
    predictions_path = 'predicted_ball_positions.csv'
    
    # Directory to save validation results
    output_dir = 'validation_results'
    
    print(f"Validating ball predictions from: {predictions_path}")
    print(f"Results will be saved to: {output_dir}")
    
    # Initialize the validator
    validator = BallPredictionValidator(
        predictions_path=predictions_path,
        output_dir=output_dir
    )
    
    # Run all validation checks
    validation_results = validator.run_all_validations(generate_report=True)
    
    print("\nValidation complete!")
    print(f"Check the validation report in {output_dir} for detailed results.")
    
    # Access some key validation metrics
    basic_checks = validation_results['basic_checks']
    movement_checks = validation_results['movement_checks']
    
    print("\nKey Validation Metrics:")
    print(f"- Total predictions: {basic_checks['total_predictions']}")
    print(f"- Missing values: {basic_checks['missing_values']['total_rows_with_missing']}")
    print(f"- Max ball speed: {movement_checks['speed_stats']['max']:.2f} units/second")
    print(f"- Mean ball speed: {movement_checks['speed_stats']['mean']:.2f} units/second")
    print(f"- Large position jumps: {movement_checks['large_jumps']['count']}")
    print(f"- Unrealistic speeds: {movement_checks['unrealistic_speeds']['count']}")
    
    # Validation summary
    total_issues = (
        basic_checks['missing_values']['total_rows_with_missing'] +
        movement_checks['large_jumps']['count'] +
        movement_checks['unrealistic_speeds']['count']
    )
    
    issue_percentage = (total_issues / basic_checks['total_predictions']) * 100
    
    if issue_percentage < 1:
        print("\n✅ Predictions look good! Very few issues detected.")
    elif issue_percentage < 5:
        print("\n⚠️ Predictions have some minor issues that may need attention.")
    else:
        print("\n❌ Predictions have significant issues that should be addressed.")


if __name__ == "__main__":
    validate_predictions()