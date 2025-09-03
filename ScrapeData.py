import os
import pandas as pd

# === NFL WEEK NUMBER ===
NFL_WEEK_TXT_FILE = "C:/Users/greg_yonan/Desktop/Tools/Programs/NFL/2025 Season/Week_Counter.txt"
with open(NFL_WEEK_TXT_FILE, 'r') as file:
    NFL_WEEK = file.read().strip()
    NFL_WEEK = int(NFL_WEEK.split()[1])
    NFL_WEEK = f"Week {NFL_WEEK}"
    NFL_WEEK_MINUS_1 = f"Week {int(NFL_WEEK.split()[1]) - 1}"
    
# === CONFIGURATION ===    
URL = "https://www.pro-football-reference.com/years/2025/"                                             # URL to scrape data from
FINAL_SAVE_DIR = "C:/Users/greg_yonan/Desktop/Tools/Programs/NFL/2025 Season/Data/" + NFL_WEEK_MINUS_1 # Final save directory
WAIT = 8                                                                                               # Time to wait after clicking download

# === USE PANDAS TO SCRAPE DATA ===
def scrape_data():
    # Read the data from the URL
    AFC = pd.read_html(URL, header=0)[0]
    NFC = pd.read_html(URL, header=0)[1]
    
    # Clean the data
    AFC = AFC.drop([0, 5, 10, 15])
    NFC = NFC.drop([0, 5, 10, 15])
    
    # combine the dataframes
    NFL = pd.concat([AFC, NFC], ignore_index=True)
    NFL = NFL.dropna()
    NFL = NFL.reset_index(drop=True)
    
    # if Team name has '*' or '+' remove it
    NFL['Tm'] = NFL['Tm'].str.replace('*', '', regex=False)
    NFL['Tm'] = NFL['Tm'].str.replace('+', '', regex=False)
    
    # Save the data to a CSV file to the final save directory
    NFL.to_csv(os.path.join(FINAL_SAVE_DIR, "NFL_2025.csv"), index=False)
    print("\nData scraped and saved to NFL_2025.csv")
    
        
# === MAIN FUNCTION ===
if __name__ == "__main__":
    # Call the scrape_data function
    scrape_data()
    # print a message to the user
    print("Exiting...")
    # Close the script
    os._exit(0)
# === END OF SCRIPT ===