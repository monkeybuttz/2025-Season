# NFL Machine Learning Project

This project aims to predict the outcomes of NFL games using machine learning techniques. The project utilizes historical data, feature engineering, and various machine learning models to forecast game results.

## Project Structure

- `Data/`: Contains historical NFL data used for training and testing the models.
- `Predictions/`: Stores the predictions made by the models for upcoming games.
- `Models/`: Contains the trained machine learning models.
- `Main.py`: The main script that orchestrates the data processing, model training, and prediction generation.
- `Ensemble.py`: Implements an ensemble learning approach to improve prediction accuracy.
- `GBM.py`: Implements a Gradient Boosting Machine model for predicting game outcomes.
- `Regression.py`: Implements regression models for predicting scores.
- `RandomForest.py`: Implements a Random Forest model for predicting game outcomes.
- `ScrapeData.py`: Script for scraping NFL data from various sources.
- `Increment_Counter.py`: Script to increment the week counter for tracking the current NFL week.
- `README.md`: This file, providing an overview of the project.
- `Week_Counter.txt`: A text file that keeps track of the current NFL week for which predictions are being made.
- `requirements.txt`: Lists the Python dependencies required for the project.

## Getting Started

1. Clone the repository to your local machine.
2. Ensure you have Python and the required libraries installed (e.g., pandas, scikit-learn, xgboost). You can install the dependencies using:

   ```bash
   pip install -r requirements.txt
   ```

3. cd into the project directory and run the script to start the prediction process by executing:

   ```bash
   cd <path/to/project>
   python Main.py
   ```

   where `<path/to/project>` is the path to the directory where this README file is located.
