#!/usr/bin/env python
# coding: utf-8

"""This file processes Saccade data. It uses three functions, each of which stores intermediary file (or final file for the last function). This helps with debugging.

1. FilteredRows
    - Drop participant data where ET is lower than 85
    - Drop participant data that were explicitly omitted/blacked out in the Crack Detection Sheet
    - Drop cracks that were only hits, no miss (to cope with class imbalance)
    - Drop participants where Gaze Calibration is poor
    - Drop rows that do not map into any AOI labels (irrelevant rows)
    - Filter outlier data for Saccade Amplitude and Peak Velocity. This didn't work for other measurements.
    
    Save the output in one file titled 'combined_no_outlier_saccade_table.csv' under saccade_processing folder.
    
2. ExtractAOIForTimeWindows 
    - Take combined_no_outlier_saccade_table.csv file. 
    -  Extract rows that fall within time window for each AOI. If they are fully out, these rows are dropped. If they are partially in, they are assigned a weight of 0.5. If they are fully in, their weight is 1. 
    
    Save the output in one file titled combined_filtered_SaccadeTable_{name of timecard} under saccade_processing folder.
    
3. AggrevateFinalData
    - It groups rows by AOI (defined by bridge, participant and crack name) and calculate median, weighted mean and weighted std for each AOI. It makes sure to drop NA value when calculating aggrevated stats because having them
    without any processing could make mean and std values NA, which is not desirable. 
    
    Save the output in one file titled 'saccade_final_{name of timecard} csv'.
"""

import pandas as pd
import numpy as np
import glob
import os

def filterRows(input_directory, output_path):
    """
    This function will filter saccade rows and save them in one single output file. It takes the following steps:
    1. Drop participants where ET is lower than 85
    2. Drop participants that were explicitly omitted (blacked out in Bridge Detection Sheet)
    3. Drop cracks where there are only hits, no miss
    4. Drop participants where Gaze Callibration is poor
    5. Drop rows that do not map to any AOI labels
    6. Finally, it filters outlier data (only for Saccade Amplitude and Peak Velocity). It will calcualte mean and std for each participant and will find 1 percent of data (about 3.5 standard deviation away) 
    and replace the data with "NA"
    
        
    Args:
        input_directory (pandas dataframe): directory that holds original saccade data (four files)

    Returns:
        _type_: pandas data frame
    """
    # Initialize an empty list to hold all rows for the combined DataFrame
    combined_rows = []

    # Iterate over all CSV files in the specified directory
    for file_path in glob.glob(os.path.join(input_directory, '*.csv')):
        
        # Load each CSV file into a DataFrame, skipping the first 6 rows
        df = pd.read_csv(file_path, skiprows=6, keep_default_na=False)
        
        #convert respondent name into int type
        df["Respondent Name"] = df["Respondent Name"].astype(int)

        # if ((df["Respondent Name"] == 20009) & df['Study Name'].str.contains("easy 1", na=False)).any():
        #     print("Respondent 20009 still present in 'easy 1' at the beginning", file_path)
            
        # Drop specific irrelevant columns
        columns_to_drop = ["Respondent Gender", "Respondent Age", "Respondent Group", "Stimulus Label"]
        df = df.drop([col for col in columns_to_drop if col in df.columns], axis='columns')
            
        #Drop participants where ET is lower than 85
        df.drop(df[(df['Study Name'].str.contains("easy 1", na=False)) & (df["Respondent Name"]==20020)].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("easy 2", na=False)) & (df["Respondent Name"]==20014)].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("easy 2", na=False)) & (df["Respondent Name"]==20020)].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("easy 2", na=False)) & (df["Respondent Name"]==20022)].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("easy 2", na=False)) & (df["Respondent Name"]==20028)].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("hard 1", na=False)) & (df["Respondent Name"]==20025)].index, inplace=True)
        
        #Drop rows where they are expliclty omitted
        df.drop(df[(df["Respondent Name"]==20019)].index, inplace=True)
        df.drop(df[(df["Respondent Name"]==20040)].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("easy 1", na=False)) & (df["Respondent Name"]==20012)].index, inplace=True)
        
        #Drop cracks that are all hit
        df.drop(df[(df['Study Name'].str.contains("easy 1", na=False)) & (df['AOI Label'].str.lower().str.contains("crack 3 hit", na=False))].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("easy 2", na=False)) & (df['AOI Label'].str.lower().str.contains("crack 3 hit", na=False))].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("easy 2", na=False)) & (df['AOI Label'].str.lower().str.contains("crack 10 hit", na=False))].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("easy 2", na=False)) & (df['AOI Label'].str.lower().str.contains("crack 14 hit", na=False))].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("easy 2", na=False)) & (df['AOI Label'].str.lower().str.contains("crack 15 hit", na=False))].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("hard 1", na=False)) & (df['AOI Label'].str.lower().str.contains("crack 4 hit", na=False))].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("hard 1", na=False)) & (df['AOI Label'].str.lower().str.contains("crack 5 hit", na=False))].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("hard 1", na=False)) & (df['AOI Label'].str.lower().str.contains("crack 17 hit", na=False))].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("hard 1", na=False)) & (df['AOI Label'].str.lower().str.contains("crack 20 hit", na=False))].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("hard 2", na=False)) & (df['AOI Label'].str.lower().str.contains("crack 4 hit", na=False))].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("hard 2", na=False)) & (df['AOI Label'].str.lower().str.contains("crack 15 hit", na=False))].index, inplace=True)
        df.drop(df[(df['Study Name'].str.contains("hard 2", na=False)) & (df['AOI Label'].str.lower().str.contains("crack 16 hit", na=False))].index, inplace=True)
        
        
        # if ((df["Respondent Name"] == 20009) & df['Study Name'].str.contains("easy 1", na=False)).any():
        #     print("Respondent 20009 still present in 'easy 1' after dropping some rows", file_path)
            
        # Drop rows where gaze is a poor calibration
        df = df[df["Gaze Calibration"] != "Poor"]
        
        #Extract rows that match AOI labels
        df = df[df["AOI Label"] != "NA"]

        # if ((df["Respondent Name"] == 20009) & df['Study Name'].str.contains("easy 1", na=False) ).any():
        #     print("Respondent 20009 still present in 'easy 1' after droppin NA and poor calibration", file_path)
        
        #Filtering outliers (only for saccade amplitude and saccade peak velocity)
        # Initialize a dictionary to collect participant-specific data for calculating statistics
        participants_collection = {}
        
        # Collecting data for each participant
        for index, row in df.iterrows():
            study = row["Study Name"]
            participant = row["Respondent Name"]
            saccade_amp = row["Saccade Amplitude"]
            saccade_peak_vel = row["Saccade Peak Velocity"]
            
            #Fix naming convention
            if "easy 1" in study:
                df.loc[index, "Study Name"] = "Bridge 1"
            elif "easy 2" in study:
                df.loc[index, "Study Name"] = "Bridge 2"
            elif "hard 1" in study:
                df.loc[index, "Study Name"] = "Bridge 3"
            elif "hard 2" in study:
                df.loc[index, "Study Name"] = "Bridge 4"

            if participant not in participants_collection:
                participants_collection[participant] = {
                    'Saccade Amplitude': [],
                    'Saccade Peak Velocity': []
                }
            
            participants_collection[participant]["Saccade Amplitude"].append(saccade_amp)
            participants_collection[participant]["Saccade Peak Velocity"].append(abs(saccade_peak_vel))
        
        # Calculate mean and std for each participant
        participants_stats = {}
        
        def get_stats(data):
            """
            The function calculates mean and standard deviation for data in each list (per participant)
            """
            cleaned_data = [float(x) for x in data if x != "NA"]
            mean_value = np.mean(cleaned_data)
            std_dev = np.std(cleaned_data)
            return [mean_value, std_dev]
        
        #Run the above function for all participants
        for participant, data in participants_collection.items():
            participants_stats[participant] = {
                "Saccade Amplitude": get_stats(data["Saccade Amplitude"]),
                "Saccade Peak Velocity": get_stats(data["Saccade Peak Velocity"])
            }
            
        #Now, filter outliers.
        def check_range(stats, data):
            """
            Return false if data falls outside the range (or NA), return true if data is within 3.5 std
            """
            if data == "NA":
                return False
            mean, std = stats
            
            # Define the range for 3.5 standard deviations
            lower_bound = mean - 3.5 * std
            upper_bound = mean + 3.5 * std
            
            return lower_bound <=abs(float(data)) <= upper_bound

        #keep outlier rows - this is only for records we don't really use it for anything
        outlier_row ={}
        outlier_row["Saccade Amplitude"] = []
        # filtered_row["Saccade Peak Acceleration"] =[]
        # filtered_row["Saccade Peak Deceleration"] =[]
        outlier_row["Saccade Peak Velocity"] =[]
        sacacade_amplitude_count = 0
        saccade_peak_velocity_count = 0

        #Any outlier data is replaced with "NA"
        for index, row in df.iterrows():
            participant= row["Respondent Name"]
            saccade_amp = row["Saccade Amplitude"]
            saccade_peak_vel = row["Saccade Peak Velocity"]
            # saccade_peak_ac = row["Saccade Peak Acceleration"]
            # sccade_peak_dec = row["Saccade Peak Deceleration"]
            
            participant_stat_amp = participants_stats[participant]["Saccade Amplitude"]
            participant_stat_vel = participants_stats[participant]["Saccade Peak Velocity"]
            # participant_stat_ac = participants_stats[participant]["Saccade Peak Acceleration"]
            # participant_stat_dec = participants_stats[participant]["Saccade Peak Deceleration"]
            
            if saccade_amp!="NA":
                sacacade_amplitude_count +=1
            if  saccade_peak_vel!="NA":
                saccade_peak_velocity_count +=1
            if saccade_amp !="NA" and not check_range(participant_stat_amp, saccade_amp):
                df.loc[index, "Saccade Amplitude"] = "NA"
                outlier_row['Saccade Amplitude'].append((index, participant))
            if saccade_peak_vel !="NA" and not check_range(participant_stat_vel, saccade_peak_vel):
                df.loc[index, "Saccade peak Velocity"] = "NA"
                outlier_row["Saccade Peak Velocity"].append((index, participant))
            # if saccade_amp !="NA" and not check_range(participant_stat_ac, saccade_peak_ac):
            #     row["Saccade Peak Acceleration"] = "NA"
            #     filtered_row["Saccade Peak Acceleration"].append(index)
            # if saccade_amp !="NA" and not check_range(participant_stat_dec, sccade_peak_dec):
            #     row["Saccade Peak Deceleration"] = "NA"
            #     filtered_row["Saccade Peak Deceleration"].append(index)
        
        #We add this four times since we are processing four files into one output file
        combined_rows.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(combined_rows, ignore_index=True)

    # Sort by "Study Name" first in the specified order, then by "Respondent" numerically
    study_order = ["Bridge 1", "Bridge 2", "Bridge 3", "Bridge 4"]
    combined_df['Study Name'] = pd.Categorical(combined_df['Study Name'], categories=study_order, ordered=True)
    combined_df = combined_df.sort_values(by=['Study Name', 'Respondent Name'])


    # Save the combined DataFrame to a single CSV file
    combined_df.to_csv(output_path, index=False)
    print(f"Combined data with outlier filtering saved to {output_path}")
    
    print(outlier_row)
    print("percentage of saccade ampitude filtered: ", len(outlier_row['Saccade Amplitude'])/sacacade_amplitude_count)
    print("percentage of peak velocity filtered: ", len(outlier_row['Saccade Peak Velocity'])/saccade_peak_velocity_count)
    
    return combined_df
        

def extractAOIForTimeWindows(combined_df,timecard_df, output_path):    
    """The function will select rows that fall within AOI time windows for each crack. It's based on time window rule set by timecard_df. 

    Args:
        combined_df (pandas dataframe): It keeps all relevant, filtered rows that fall within sort of AOI (but without timewindows)
        timecard_df (pandas dataframe): Timecard rule by each crack/participant/bridge

    Returns:
        _type_: pandas dataframe
    """
    # combined_df= pd.read_csv('data/saccade_processing/combined_filtered_SaccadeTable.csv', skiprows=0)
    
    #Need to keep track of weight. For row that falls fully within time window is assgined a weight of 1, the one that particially falls within is assigned a weight of 0.5
    combined_df['weight'] = 1

    #Show unique participants for each bridge
    unique_respondents_bridge_1 = combined_df[combined_df["Study Name"] == "Bridge 1"]["Respondent Name"].unique()
    print('Bridge 1: ', unique_respondents_bridge_1)

    unique_respondents_bridge_2 = combined_df[combined_df["Study Name"] == "Bridge 2"]["Respondent Name"].unique()
    print('Bridge 2: ', unique_respondents_bridge_2)

    unique_respondents_bridge_3 = combined_df[combined_df["Study Name"] == "Bridge 3"]["Respondent Name"].unique()
    print('Bridge 3: ', unique_respondents_bridge_3)
    
    unique_respondents_bridge_4 = combined_df[combined_df["Study Name"] == "Bridge 4"]["Respondent Name"].unique()
    print('Bridge 4: ', unique_respondents_bridge_4)

    #Keep tracks of rows to drop
    rows_to_drop =[]
    
    # Iterate through each row in the DataFrame
    for index, row in combined_df.iterrows():
        study = row["Study Name"]
        participant = int(row["Respondent Name"])
        aoiLabel = row["AOI Label"]
        saccade_start = row["Saccade Start"]
        saccade_end = row["Saccade End"]
        
        time_row = timecard_df[(timecard_df["Respondent"]==participant) & (timecard_df["Label"]==aoiLabel) & (timecard_df["Study Name"]==study)] #There should be a unique match (only one!)
        
        #Error checking
        if time_row.empty:
            print("Row empty: ", participant, aoiLabel, study) #This shouldn't happen
        if time_row.shape[0] > 1:
            print("Multiple rows: ", participant, aoiLabel, study) #This shouldn't happen
            print(time_row)
        
        start = time_row["Start"].values[0]
        end = time_row["End"].values[0]
        
        #drop the rows if saccade is outside the range completely
        if saccade_end < start or saccade_start > end:
            # if (participant == 20002) & (study == "Bridge 3") & ("Crack 6" in aoiLabel):
            #     print(index, start, end, saccade_start, saccade_end)
            rows_to_drop.append(index)
        
        #saccade is partially covered by time window
        elif saccade_start < start or saccade_end > end: 
            combined_df.loc[index, 'weight'] = 0.5
            

    print(len(rows_to_drop))


    # Drop the rows from the original DataFrame
    df_filtered = combined_df.drop(rows_to_drop)

    # Count the number of rows where 'weight' equals 0.5
    count = df_filtered[df_filtered['weight'] == 0.5].shape[0]
    print(f'Number of rows where weight is 0.5: {count}')

    # if ((df_filtered["Respondent Name"] == 20002) & df_filtered['Study Name'].str.contains("Bridge 3", na=False) & df_filtered["AOI Label"].str.contains("Crack 6", na=False)).any():
    #         print("Respondent 20003 still present in 'Bridge 3'")
            
    # Save the modified DataFrame to a new CSV file (optional)
    df_filtered.to_csv(output_path, index=False)
    print(f"Data after timecard matching saved to {output_path}")
    return df_filtered

def aggrevateFinalData(processed_combined_df, output_path):
    """Finally, this function will calculate median, weighted mean, and weighted std for each AOI. 
    Args:
        processed_combined_df (pandas dataframe): This file should store all relevant rows for final data aggrevation.

    Returns:
        _type_: pandas dataframe
    """
    # processed_combined_df = pd.read_csv('data/saccade_processing/combined_filtered_SaccadeTable_3seconds_middle.csv')

    #Drop columns
    columns_to_drop = ["Gaze Calibration", "Stimulus Start", "Stimulus Duration", "Saccade Start", "Saccade End", "Saccade Index", "Saccade Index by Stimulus", "Saccade Type", "Saccade Index by AOI", "Dwell Index (on AOI)",
                    "AOI Instance Duration", "AOI Instance Start", "AOI Type", "AOI Intersections", "Saccade Direction", "AOI Instance"]

    # Drop irrelevant columns only if they exist in the DataFrame
    processed_combined_df = processed_combined_df.drop([col for col in columns_to_drop if col in processed_combined_df.columns], axis='columns')

    #We check how much of data is NA
    row_count = 0
    amplitude_count = 0
    weight_count = 0
    for index, row in processed_combined_df.iterrows():
        row_count += 1
        if pd.isna(row["Saccade Amplitude"]):  # Check if the value is NaN
            amplitude_count += 1
        if pd.isna(row["weight"]):  # Check if the value is NaN
            weight_count += 1

    print(f"NaN Percentage: {amplitude_count / row_count * 100:.2f}%")
    print(f"NaN Percentage: {weight_count / row_count * 100:.2f}%") #Should be zero


    #Defining custom functions to calculate weighted_mean and weighted_std
    nan_count = {} #Keep track of NA value AFTER aggrevation (should be fairly small)
    def weighted_mean(values, weights, title = None):
        # Remove NaN values from both values and weights
        mask = ~pd.isna(values) & ~pd.isna(weights)
        valid_values = values[mask].astype(float)
        valid_weights = weights[mask].astype(float)
        
        if len(valid_values) > 0:
            return np.average(valid_values, weights=valid_weights)
        else:
            if title not in nan_count:
                nan_count[title] = 0
            nan_count[title] +=1
            return np.nan  # Return NaN if no valid data remains

    def weighted_std(values, weights):
        # Remove NaN values from both values and weights
        mask = ~pd.isna(values) & ~pd.isna(weights)
        valid_values = values[mask].astype(float)
        valid_weights = weights[mask].astype(float)
        
        if len(valid_values) > 0:
            average = weighted_mean(valid_values, valid_weights)
            variance = np.average((valid_values - average) ** 2, weights=valid_weights)
            return np.sqrt(variance)
        else:
            return np.nan  # Return NaN if no valid data remains

    def weighted_count(values):
        return np.sum(values)

    #Calculate statistics for each group (AOI label + participant)
    def custom_agg(group):
        return pd.Series({
            'Saccade Duration Mean': weighted_mean(group['Saccade Duration'], group['weight'], 'Saccade Duration'),
            'Saccade Duration Std': weighted_std(group['Saccade Duration'], group['weight']),
            'Saccade Duration Median': group['Saccade Duration'].median(),
            'Saccade Amplitude Mean': weighted_mean(group['Saccade Amplitude'], group['weight'], 'Saccade Amplitude'),
            'Saccade Amplitude Std': weighted_std(group['Saccade Amplitude'], group['weight']),
            'Saccade Amplitude Median': group['Saccade Amplitude'].median(),
            'Saccade Peak Velocity Mean': weighted_mean(group['Saccade Peak Velocity'], group['weight'], 'Saccade Peak Velocity'),
            'Saccade Peak Velocity Std': weighted_std(group['Saccade Peak Velocity'], group['weight']),
            'Saccade Peak Velocity Median': group['Saccade Peak Velocity'].median(),
            'Saccade Peak Acceleration Mean': weighted_mean(group['Saccade Peak Acceleration'], group['weight'], "Saccade Peak Acceleration"),
            'Saccade Peak Acceleration Std': weighted_std(group['Saccade Peak Acceleration'], group['weight']),
            'Saccade Peak Acceleration Median': group['Saccade Peak Acceleration'].median(),
            'Saccade Peak Deceleration Mean': weighted_mean(group['Saccade Peak Deceleration'], group['weight'], 'Saccade Peak Deceleration'),
            'Saccade Peak Deceleration Std': weighted_std(group['Saccade Peak Deceleration'], group['weight']),
            'Saccade Peak Deceleration Median': group['Saccade Peak Deceleration'].median(),
            "Saccade Count" : weighted_count(group['weight']),
        })

    
    #Calculate the final statistics per each AOI (defined by respondent name, AOI label for crack name and which bridge)
    grouped_processed_combined_df = processed_combined_df.groupby(["Respondent Name", "AOI Label", "Study Name"]).apply(custom_agg).reset_index()

    #Reorder the dataframe
    study_order = ["Bridge 1", "Bridge 2", "Bridge 3", "Bridge 4"]
    grouped_processed_combined_df['Study Name'] = pd.Categorical(grouped_processed_combined_df['Study Name'], categories=study_order, ordered=True)

    # if ((grouped_processed_combined_df["Respondent Name"] == 20002) & grouped_processed_combined_df['Study Name'].str.contains("Bridge 3", na=False) & grouped_processed_combined_df["AOI Label"].str.contains("Crack 3", na=False)).any():
    #         print("Respondent 20002 still present in 'Bridge 3'")
    
    #Save the dataframe 
    grouped_processed_combined_df.to_csv(output_path, index=False)
    print(f"Combined data with final data aggrevation saved to {output_path}")
    print(nan_count)
    print(len(grouped_processed_combined_df))
    return grouped_processed_combined_df


def main():
    # Necessary files
    input_directory = 'data/saccade/'  # All saccade data in this folder
    timecard_df= pd.read_csv('data/timecards/combined_timecard_3seconds.csv', skiprows=0) #Read timecard.csv
    output_path_1 = "data/saccade_processing/combined_no_outlier_saccade_table.csv"
    output_path_2 = 'data/saccade_processing/combined_filtered_SaccadeTable_3seconds_middle.csv'
    output_path_3 = "data/finalized_data/saccade_final_3seconds_middle.csv"
    combined_df = filterRows(input_directory, output_path_1)
    df_filtered = extractAOIForTimeWindows(combined_df, timecard_df, output_path_2)
    grouped_processed_combined_df = aggrevateFinalData(df_filtered, output_path_3)

main()
    

# #Verify
# processed_combined_df = pd.read_csv('data/finalized_data/saccade_final_3seconds_middle.csv')
# unique_respondents_bridge_1 = processed_combined_df[processed_combined_df["Study Name"] == "Bridge 1"]["Respondent Name"].unique()

# print('Bridge 1: ', unique_respondents_bridge_1)

# unique_respondents_bridge_2 = processed_combined_df[processed_combined_df["Study Name"] == "Bridge 2"]["Respondent Name"].unique()

# print('Bridge 2: ', unique_respondents_bridge_2)

# unique_respondents_bridge_3 = processed_combined_df[processed_combined_df["Study Name"] == "Bridge 3"]["Respondent Name"].unique()

# print('Bridge 3: ', unique_respondents_bridge_3)
# unique_respondents_bridge_4 = processed_combined_df[processed_combined_df["Study Name"] == "Bridge 4"]["Respondent Name"].unique()

# print('Bridge 4: ', unique_respondents_bridge_4)



