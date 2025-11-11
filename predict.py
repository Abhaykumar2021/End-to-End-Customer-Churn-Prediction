"""
Prediction script for Customer Churn Prediction
Run this script to make predictions on new data
"""
import sys
import argparse
import pandas as pd
from churn_prediction.logger import logger
from churn_prediction.exception import ChurnPredictionException
from churn_prediction.pipeline.prediction_pipeline import PredictionPipeline


def main():
    """Main function to run prediction pipeline"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Predict Customer Churn')
        parser.add_argument('--input', type=str, required=True,
                          help='Path to the input CSV file with customer data')
        parser.add_argument('--output', type=str, required=False,
                          help='Path to save the predictions (optional)')
        args = parser.parse_args()
        
        # Read input data
        logger.info(f"Reading input data from {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Input data shape: {df.shape}")
        
        # Create prediction pipeline
        predict_pipeline = PredictionPipeline()
        
        # Make predictions
        logger.info("Making predictions...")
        predictions, prediction_probs = predict_pipeline.predict(df)
        
        # Add predictions to dataframe
        df['Predicted_Churn'] = predictions
        df['Churn_Probability'] = prediction_probs[:, 1]
        
        # Save results
        if args.output:
            df.to_csv(args.output, index=False)
            logger.info(f"Predictions saved to {args.output}")
        else:
            print("\nPredictions:")
            print(df[['Predicted_Churn', 'Churn_Probability']].head(10))
            logger.info("Predictions completed. Use --output to save results.")
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise ChurnPredictionException(e, sys)


if __name__ == "__main__":
    main()
