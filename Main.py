import os
import Increment_Counter
import ScrapeData
import Regression
import GBM
import RandomForest
import Ensemble

if __name__ == "__main__":
    # Step 1: Increment the counter to track the number of runs
    NFL_WEEK = Increment_Counter.main()
    NFL_WEEK_MINUS_1 = f"Week {int(NFL_WEEK.split()[1]) - 1}"
    print(f"Run ID: {NFL_WEEK}")
        
    # Step 2: Scrape the latest data
    ScrapeData.main(NFL_WEEK, NFL_WEEK_MINUS_1)
    print("Data Scraping Complete")
    
    # Step 3: Run Regression model
    Regression.main(NFL_WEEK, NFL_WEEK_MINUS_1)
    print("Regression Model Complete")
    
    # Step 4: Run GBM model
    GBM.main(NFL_WEEK, NFL_WEEK_MINUS_1)
    print("GBM Model Complete")
    
    # Step 5: Run Random Forest model
    RandomForest.main(NFL_WEEK, NFL_WEEK_MINUS_1)
    print("Random Forest Model Complete")
    
    # Step 6: Run Ensemble model
    Ensemble.main(NFL_WEEK, NFL_WEEK_MINUS_1)
    print("Ensemble Model Complete")
    
    print("All processes completed successfully.")
    os._exit(0)