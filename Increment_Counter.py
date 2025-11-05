import os

def main():
    # open Week_ccounter.txt, read the week number, increment it by 1 
    with open('Week_Counter.txt', 'r') as file:
        week_line = file.read().strip()
        current_week = int(week_line.split()[1])
        new_week = current_week + 1
        
    # write the new week number back to the file
    with open('Week_Counter.txt', 'w') as file:
        file.write(f'Week {new_week}')
        file.flush()
        os.fsync(file.fileno())
        
    return str((f'Week {new_week}'))