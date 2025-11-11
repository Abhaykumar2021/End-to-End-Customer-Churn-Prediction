"""
Training script for Customer Churn Prediction
Run this script to train the model
"""
import sys
import argparse
from churn_prediction.logger import logger
from churn_prediction.exception import ChurnPredictionException
from churn_prediction.pipeline.training_pipeline import TrainingPipeline


def main():
    """Main function to run training pipeline"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Train Customer Churn Prediction Model')
        parser.add_argument('--data', type=str, required=False,
                          help='Path to the training data CSV file')
        args = parser.parse_args()
        
        # Create and run training pipeline
        logger.info("Initializing Training Pipeline")
        training_pipeline = TrainingPipeline()
        
        # Run the pipeline
        training_pipeline.run_pipeline(data_path=args.data)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise ChurnPredictionException(e, sys)


if __name__ == "__main__":
    main()
