import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# === NFL WEEK NUMBER ===
NFL_WEEK_TXT_FILE = "C:/Users/greg_yonan/Desktop/Tools/Programs/NFL/2025 Season/Week_Counter.txt"
with open(NFL_WEEK_TXT_FILE, 'r') as file:
    NFL_WEEK = file.read().strip()
    NFL_WEEK = int(NFL_WEEK.split()[1])
    NFL_WEEK = f"Week {NFL_WEEK}"
    NFL_WEEK_MINUS_1 = f"Week {int(NFL_WEEK.split()[1]) - 1}"

# File paths
DATA_FILENAME = "Data/" + NFL_WEEK_MINUS_1 + "/NFL_2025.csv"
PREDICTIONS_FILENAME = "Predictions/" + NFL_WEEK + '/' +  NFL_WEEK + ".xlsx"

# Below are column indicies
TEAM = 0 # Team Name
WINS = 1 # Wins
LOSSES = 2 # Losses
WIN_PERCENTAGE = 3 # Win Percentage
POINTS_FOR = 4 # Points For
POINTS_AGAINST = 5 # Points Against
POINTS_DIFFERENTIAL = 6 # Points Differential
MARGIN_OF_VICTORY = 7 # Margin of Victory
STRENGTH_OF_SCHEDULE = 8 # Strength of Schedule
SIMPLE_RATING_SYSTEM = 9 # Simple Rating System
OFFENSIVE_SRS = 10 # Offensive SRS
DEFENSIVE_SRS = 11 # Defensive SRS

# List of all teams
NFL_TEAMS = ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers', 'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins', 'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants', 'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers', 'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Commanders']

# Get team data     
def getTeamData(team_name):
    team_data = []
    with open(DATA_FILENAME, "r") as data_file:
        data = pd.read_csv(data_file)
        for index, row in data.iterrows():
            if row.iloc[TEAM] == team_name:
                team_data.append([row.iloc[WINS], row.iloc[LOSSES], row.iloc[WIN_PERCENTAGE], row.iloc[POINTS_FOR], row.iloc[POINTS_AGAINST], row.iloc[POINTS_DIFFERENTIAL], row.iloc[MARGIN_OF_VICTORY], row.iloc[STRENGTH_OF_SCHEDULE], row.iloc[SIMPLE_RATING_SYSTEM], row.iloc[OFFENSIVE_SRS], row.iloc[DEFENSIVE_SRS]])
            elif row.iloc[TEAM] == team_name:
                team_data.append([row.iloc[WINS], row.iloc[LOSSES], row.iloc[WIN_PERCENTAGE], row.iloc[POINTS_FOR], row.iloc[POINTS_AGAINST], row.iloc[POINTS_DIFFERENTIAL], row.iloc[MARGIN_OF_VICTORY], row.iloc[STRENGTH_OF_SCHEDULE], row.iloc[SIMPLE_RATING_SYSTEM], row.iloc[OFFENSIVE_SRS], row.iloc[DEFENSIVE_SRS]])
    team_data = pd.DataFrame(team_data, columns=['W', 'L', 'W-L%', 'PF', 'PA', 'PD', 'MoV', 'SoS', 'SRS', 'OSRS', 'DSRS'])
    return team_data
    

# Train the model and save it to a file
def Train():
    NFL_team_data = []
    for team in NFL_TEAMS:
        team_data = getTeamData(team)
        if team_data is not None:  # Ensure team_data is not None
            NFL_team_data.append(team_data)
            
    combined_data = pd.concat(NFL_team_data, ignore_index=True)
    
    # Convert columns to numeric types
    combined_data['W'] = pd.to_numeric(combined_data['W'], errors='coerce')
    combined_data['L'] = pd.to_numeric(combined_data['L'], errors='coerce')
    combined_data['W-L%'] = pd.to_numeric(combined_data['W-L%'], errors='coerce')
    combined_data['PF'] = pd.to_numeric(combined_data['PF'], errors='coerce')
    combined_data['PA'] = pd.to_numeric(combined_data['PA'], errors='coerce')
    combined_data['PD'] = pd.to_numeric(combined_data['PD'], errors='coerce')
    combined_data['MoV'] = pd.to_numeric(combined_data['MoV'], errors='coerce')
    combined_data['SoS'] = pd.to_numeric(combined_data['SoS'], errors='coerce')
    combined_data['SRS'] = pd.to_numeric(combined_data['SRS'], errors='coerce')
    combined_data['OSRS'] = pd.to_numeric(combined_data['OSRS'], errors='coerce')
    combined_data['DSRS'] = pd.to_numeric(combined_data['DSRS'], errors='coerce')
    
    # Drop rows with any NaN values
    combined_data.dropna(subset=['W', 'L', 'W-L%', 'PF', 'PA', 'PD', 'MoV', 'SoS', 'SRS', 'OSRS', 'DSRS'], inplace=True)
    
    # Create features and labels
    x = combined_data[['W', 'L', 'PF', 'PA', 'PD', 'MoV', 'SoS', 'SRS', 'OSRS', 'DSRS']]
    y = combined_data['W-L%']
    
    # Train the regression model
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    
    # Save the model to disk
    filename = 'Models/' + NFL_WEEK + '/Regression.sav'
    pickle.dump(model, open(filename, 'wb'))
    print('Model saved to disk')
    
    
# given a list of nfl games on an xlsx, predict the outcome of each game
def printOutcomes(model):
    # Read the Excel file    
    data = pd.read_excel(PREDICTIONS_FILENAME)
    data['Regression'] = data['Regression'].astype(str)
    
    for index, row in data.iterrows():
        team_a = row['Away Team']
        team_b = row['Home Team']
        team_a_data = getTeamData(team_a)
        team_b_data = getTeamData(team_b)
        team_a_features = pd.DataFrame([team_a_data[['W', 'L', 'PF', 'PA', 'PD', 'MoV', 'SoS', 'SRS', 'OSRS', 'DSRS']].mean().values], columns=['W', 'L', 'PF', 'PA', 'PD', 'MoV', 'SoS', 'SRS', 'OSRS', 'DSRS'])
        team_b_features = pd.DataFrame([team_b_data[['W', 'L', 'PF', 'PA', 'PD', 'MoV', 'SoS', 'SRS', 'OSRS', 'DSRS']].mean().values], columns=['W', 'L', 'PF', 'PA', 'PD', 'MoV', 'SoS', 'SRS', 'OSRS', 'DSRS'])
        team_a_win_prob = model.predict(team_a_features)[0]
        team_b_win_prob = model.predict(team_b_features)[0]
        
        if team_a_win_prob > team_b_win_prob:
            data.at[index, 'Regression'] = team_a
        else:
            data.at[index, 'Regression'] = team_b
    
    data.to_excel(PREDICTIONS_FILENAME, index=False)
    
# # Main function
if __name__ == '__main__':
    
    # create and train the model
    Train()
    
    # use model (Models\combined_model.sav) to predict the outcome of the game
    model = pickle.load(open('Models/' + NFL_WEEK + '/Regression.sav', 'rb'))
    
    printOutcomes(model)
    
    

