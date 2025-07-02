import os
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Constants and Configuration
@dataclass
class Config:
    OUTLIER_THRESHOLD: float = 3.5
    REQUIRED_COLUMNS: List[str]    = ('Study Name', 'Respondent Name', 'Fixation Duration', 'Fixation Dispersion')
    NUMERIC_COLUMNS: List[str]     = ('Fixation Duration', 'Fixation Dispersion')
    CATEGORICAL_COLUMNS: List[str] = ('Study Name', 'Respondent Name')

# Participant exclusion configuration
PARTICIPANT_FILTER: Dict[str, Set[int]] = {
    'bridge_1': {20003, 20006, 20007, 20010, 20012, 20018, 20019, 20020, 20023, 20033, 20040},
    'bridge_2': {20003, 20007, 20011, 20014, 20018, 20019, 20020, 20022, 20027, 20028, 20040, 20041, 20045},
    'bridge_3': {20010, 20011, 20013, 20014, 20018, 20019, 20021, 20025, 20028, 20029, 20031, 20033, 20040},
    'bridge_4': {20003, 20007, 20008, 20012, 20013, 20014, 20018, 20019, 20028, 20033, 20040, 20041}
}

# Updated bridge mapping to fix the bridge name mismatch issue
BRIDGE_MAPPING = {
    'bridge_1': 'Bridge 1',
    'bridge_2': 'Bridge 2', 
    'bridge_3': 'Bridge 3',
    'bridge_4': 'Bridge 4',
    'easy 1': 'Bridge 1',
    'easy 2': 'Bridge 2',
    'hard 1': 'Bridge 3',
    'hard 2': 'Bridge 4'
}

# Cracks to filter from each bridge
CRACKS_FILTER = {
    "Bridge 1": {'Crack 3'},
    "Bridge 2": {'Crack 3', 'Crack 10', 'Crack 14', 'Crack 15', 'Crack 19'},
    "Bridge 3": {'Crack 4', 'Crack 5', 'Crack 17', 'Crack 20'},
    "Bridge 4": {'Crack 4', 'Crack 15', 'Crack 16'}
}

PARTICIPANT_SPECIFIC_CRACKS = {
    "Bridge 2": {
        20008: {'Crack 11'}
    }
}

class DataValidator:
    @staticmethod
    def validate_data(df: pd.DataFrame) -> None:
        """Validate input data integrity."""
        missing_cols = set(Config.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        for col in Config.NUMERIC_COLUMNS:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must be numeric")

class DataOptimizer:
    @staticmethod
    def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage."""
        df = df.copy()
        for col in Config.CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype('category')
        return df

class CSVHandler:

    @staticmethod
    def safe_read_csv(file_path: str) -> pd.DataFrame:
        """Safely read CSV with error handling and proper data types."""
        try:
            # Define data types for numerical columns
            dtype_dict = {
                'Fixation Duration': 'float64',
                'Fixation Dispersion': 'float64',
                'Fixation X': 'float64',
                'Fixation Y': 'float64',
                'Fixation Start': 'float64',
                'Fixation End': 'float64',
                'Respondent Age': 'float64',
                'Stimulus Start': 'float64',
                'Stimulus Duration': 'float64',
                'AOI Instance Start': 'float64',
                'AOI Instance Duration': 'float64'
            }
            
            # Read CSV with specified data types and low_memory=False
            df = pd.read_csv(file_path, dtype=dtype_dict, low_memory=False)
            
            # Convert any remaining numeric columns that might have mixed types
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"Empty CSV file: {file_path}")
        except pd.errors.ParserError:
            raise ValueError(f"Malformed CSV file: {file_path}")
        except Exception as e:
            raise ValueError(f"Error reading {file_path}: {str(e)}")
    
    @staticmethod
    def filter_base_and_fa_labels(fixation_aoi: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out rows with 'Base' and 'FA' labels from the fixation AOI dataframe.
        
        Parameters:
        -----------
        fixation_aoi : pd.DataFrame
            The fixation AOI dataframe with a 'Label' column
        
        Returns:
        --------
        pd.DataFrame
            Filtered dataframe with only 'Crack' labels
        """
        # Check if Label column exists
        if 'Label' not in fixation_aoi.columns:
            print("Warning: Label column not found in dataframe")
            return fixation_aoi
        
        # Count rows before filtering
        total_rows = len(fixation_aoi)
        print(f"Total rows before filtering: {total_rows}")
        
        # Create a mask to keep only rows with 'Crack' in the Label
        crack_mask = fixation_aoi['Label'].str.contains('Crack', case=False, na=False)
        filtered_fixation_aoi = fixation_aoi[crack_mask].copy()
        
        # Count rows after filtering
        filtered_rows = len(filtered_fixation_aoi)
        removed_rows = total_rows - filtered_rows
        print(f"Rows with 'Crack' labels: {filtered_rows}")
        print(f"Rows with 'Base' or 'FA' labels removed: {removed_rows}")
        
        return filtered_fixation_aoi

    # Usage in your code:
    # fixation_aoi = filter_base_and_fa_labels(fixation_aoi)
    # fixation_aoi.to_csv('filtered_fixation_aoi.csv', index=False)

    @staticmethod
    def load_and_filter_timecard(timecard_path: str) -> pd.DataFrame:
        """Loads and filters timecard CSV, removing screen recordings and excluded cracks."""
        df = CSVHandler.safe_read_csv(timecard_path)
        
        # Check if 'Respondent' column exists and rename it
        if 'Respondent' in df.columns and 'Respondent Name' not in df.columns:
            df.rename(columns={'Respondent': 'Respondent Name'}, inplace=True)
        
        # Ensure Respondent Name is numeric
        df['Respondent Name'] = pd.to_numeric(df['Respondent Name'], errors='coerce')
        
        # First filter out screen recordings
        if 'Label' in df.columns:
            df = df[~df['Label'].str.contains('Screen recording', na=False)]
        
        # Then filter out excluded cracks for each bridge
        mask = pd.Series(True, index=df.index)
        
        # Apply bridge-level crack filtering
        for bridge, excluded_cracks in CRACKS_FILTER.items():
            bridge_mask = (df['Study Name'] == bridge)
            for crack in excluded_cracks:
                if 'Label' in df.columns:
                    mask &= ~(bridge_mask & df['Label'].str.contains(crack, na=False))
        
        # Apply participant-specific crack filtering
        for bridge, participants in PARTICIPANT_SPECIFIC_CRACKS.items():
            for participant, excluded_cracks in participants.items():
                participant_mask = (df['Study Name'] == bridge) & (df['Respondent Name'] == participant)
                for crack in excluded_cracks:
                    if 'Label' in df.columns:
                        mask &= ~(participant_mask & df['Label'].str.contains(crack, na=False))
        
        # Apply the mask
        filtered_df = df[mask]
        
        # Debug logging
        print(f"Timecard rows before crack filtering: {len(df)}")
        print(f"Timecard rows after crack filtering: {len(filtered_df)}")
        print("Participant-specific filtering applied for:")
        for bridge, participants in PARTICIPANT_SPECIFIC_CRACKS.items():
            for participant, cracks in participants.items():
                print(f"  {bridge}, Participant {participant}: {', '.join(cracks)}")
        
        return filtered_df
        
    @staticmethod
    def combine_csv_files(folder_path: str) -> pd.DataFrame:
        """Combines two CSV files in a folder and removes header artifacts."""
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if len(csv_files) != 2:
            raise ValueError(f"Expected 2 CSV files, found {len(csv_files)} in {folder_path}")

        dfs = []
        for csv_file in csv_files:
            df = CSVHandler.safe_read_csv(os.path.join(folder_path, csv_file))
            header = df.iloc[5]
            df = df.iloc[6:]
            df.columns = header
            dfs.append(df)

        combined_df = pd.concat(dfs, axis=0, ignore_index=True)
        return combined_df

class DataProcessor:
    @staticmethod
    def filter_participants(df: pd.DataFrame, bridge_num: str) -> pd.DataFrame:
        """Removes excluded participants for a given bridge."""
        try:
            # Get the exclusion set for this bridge
            filter_set = PARTICIPANT_FILTER.get(bridge_num, set())
            
            # Ensure Respondent Name is numeric
            df['Respondent Name'] = pd.to_numeric(df['Respondent Name'], errors='coerce')
            
            # Log before filtering
            before_count = len(df)
            
            # Filter out participants
            df = df[~df['Respondent Name'].isin(filter_set)]
            
            # Log after filtering
            after_count = len(df)
            filtered_count = before_count - after_count
            
            print(f"Participants filtered from {bridge_num}: {filtered_count} rows removed")
            
            return df
        except Exception as e:
            raise ValueError(f"Error filtering participants for {bridge_num}: {str(e)}")

    @staticmethod
    def rename_bridges(df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes bridge names using the global BRIDGE_MAPPING."""
        df = df.copy()
        
        # Create a new column with standardized bridge names
        if 'Study Name' in df.columns:
            # Apply mapping to standardize bridge names
            for old_name, new_name in BRIDGE_MAPPING.items():
                mask = df['Study Name'].str.contains(old_name, case=False, na=False)
                df.loc[mask, 'Study Name'] = new_name
            
            # Debug information
            unique_bridges = df['Study Name'].unique()
            print(f"Bridge names after standardization: {unique_bridges}")
            
        return df

    @staticmethod
    def filter_outliers_vectorized(df: pd.DataFrame) -> pd.DataFrame:
        """Filters outliers using vectorized operations."""
        df = df.copy()
        grouped = df.groupby(['Study Name', 'Respondent Name'])
        
        # Track total and outlier counts
        total_counts = {}
        outlier_counts = {}
        
        # Ensure columns are numeric before processing
        for metric in ['Fixation Duration', 'Fixation Dispersion']:
            if metric not in df.columns:
                print(f"Warning: Column {metric} not found in dataframe")
                continue
                
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
            total_counts[metric] = df[metric].notna().sum()
            outlier_counts[metric] = 0
            
            stats = grouped[metric].agg(['mean', 'std'])
            # Only process where we have valid statistics
            valid_stats = stats.dropna()
            
            lower_bound = valid_stats['mean'] - Config.OUTLIER_THRESHOLD * valid_stats['std']
            upper_bound = valid_stats['mean'] + Config.OUTLIER_THRESHOLD * valid_stats['std']
            
            for (bridge, participant), (lower, upper) in zip(valid_stats.index, zip(lower_bound, upper_bound)):
                mask = (
                    (df['Study Name'] == bridge) & 
                    (df['Respondent Name'] == participant) & 
                    ~df[metric].between(lower, upper)
                )
                outlier_counts[metric] += mask.sum()
                df.loc[mask, metric] = np.nan
        
        # Print outlier percentages
        print("\nOutlier Statistics:")
        for metric in outlier_counts:
            percentage = (outlier_counts[metric] / total_counts[metric] * 100) if total_counts[metric] > 0 else 0
            print(f"{metric}: {outlier_counts[metric]} outliers out of {total_counts[metric]} valid values ({percentage:.2f}%)")
                    
        return df
    
    @staticmethod
    def normalize_zscore(df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs z-score normalization on fixation metrics, grouped by participant and bridge.
        Ensures mean is exactly 0 and standard deviation is 1 for each group.
        """
        df = df.copy()
        
        # Metrics to normalize
        metrics = ['Fixation Duration', 'Fixation Dispersion']
        
        # Track normalization statistics
        for metric in metrics:
            if metric not in df.columns:
                print(f"Warning: Column {metric} not found for normalization")
                continue
                
            # Create new column for normalized values
            norm_col = f'{metric}_normalized'
            
            # Ensure column is numeric
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
            
            # Calculate z-scores within each group
            def normalize_group(group):
                # Skip groups with no variation
                if group.std() == 0:
                    return pd.Series(0, index=group.index)
                
                # Perform z-score normalization
                return (group - group.mean()) / group.std()
            
            # Apply normalization to each group
            df[norm_col] = df.groupby(['Study Name', 'Respondent Name'])[metric].transform(normalize_group)
            
            # Original range verification
            print(f"\nOriginal {metric} range: {df[metric].min():.2f} to {df[metric].max():.2f}")
            print(f"Normalized {metric} range: {df[norm_col].min():.2f} to {df[norm_col].max():.2f}")
        
        return df

class FixationAnalyzer:
    @staticmethod
    def calculate_overlap_vectorized(fixation_table: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
        """Calculate fixation overlaps using vectorized operations."""
        # Check if required columns exist
        required_columns = ['Study Name', 'Respondent Name', 'Start', 'End', 'Fixation Start', 'Fixation End']
        for df, df_name in [(fixation_table, 'fixation_table'), (intervals, 'intervals')]:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns in {df_name}: {missing_cols}")
        
        # Ensure numeric types for time columns
        numeric_columns = ['Start', 'End', 'Fixation Start', 'Fixation End']
        
        # Convert in fixation_table
        for col in numeric_columns[2:]:  # Fixation Start and End
            if col in fixation_table.columns:
                fixation_table[col] = pd.to_numeric(fixation_table[col], errors='coerce')
            
        # Convert in intervals
        for col in numeric_columns[:2]:  # Start and End
            if col in intervals.columns:
                intervals[col] = pd.to_numeric(intervals[col], errors='coerce')
        
        # Check for participant ID types
        print(f"Respondent Name types - fixation_table: {fixation_table['Respondent Name'].dtype}, intervals: {intervals['Respondent Name'].dtype}")
        
        # Ensure Respondent Name is numeric in both dataframes
        fixation_table['Respondent Name'] = pd.to_numeric(fixation_table['Respondent Name'], errors='coerce')
        intervals['Respondent Name'] = pd.to_numeric(intervals['Respondent Name'], errors='coerce')
        
        # Count merged records
        print(f"Fixation table rows: {len(fixation_table)}")
        print(f"Intervals rows: {len(intervals)}")
        
        # Create a merged dataframe with overlap conditions
        merged = pd.merge(
            intervals,
            fixation_table,
            on=['Study Name', 'Respondent Name'],
            suffixes=('_interval', '_fixation')
        )
        
        print(f"Merged rows before overlap filtering: {len(merged)}")
        
        # Calculate overlaps
        merged['overlap_start'] = np.maximum(merged['Start'], merged['Fixation Start'])
        merged['overlap_end'] = np.minimum(merged['End'], merged['Fixation End'])
        merged['overlap_duration'] = merged['overlap_end'] - merged['overlap_start']
        # merged = merged[merged['overlap_duration'] > 0]
        
        print(f"Merged rows after overlap filtering: {len(merged)}")
        
        # Calculate weights
        merged['fixation_duration'] = merged['Fixation End'] - merged['Fixation Start']
        merged['weight'] = np.where(
            (merged['fixation_duration'] > 0) & (merged['overlap_duration'] > 0),
            merged['overlap_duration'] / merged['fixation_duration'],
            0
        )
        
        return merged

    @staticmethod
    def calculate_fixation_statistics(df: pd.DataFrame, fixation_table: pd.DataFrame) -> pd.DataFrame:
        """Calculate fixation statistics using normalized data"""
        # Calculate overlaps using the original fixation table
        overlaps = FixationAnalyzer.calculate_overlap_vectorized(fixation_table, df)

        # Adjust weights: set to 0.5 for partial overlaps
        overlaps['adjusted_weight'] = np.where(
            (overlaps['weight'] > 0) & (overlaps['weight'] < 1),
            0.5,
            overlaps['weight']
        )
        
        # Use normalized columns if they exist
        duration_col = 'Fixation Duration_normalized' if 'Fixation Duration_normalized' in overlaps.columns else 'Fixation Duration'
        dispersion_col = 'Fixation Dispersion_normalized' if 'Fixation Dispersion_normalized' in overlaps.columns else 'Fixation Dispersion'
        
        # Check group count
        group_count = overlaps.groupby(['Study Name', 'Respondent Name', 'Start', 'End']).ngroups
        print(f"Number of groups for statistics calculation: {group_count}")
        
        stats_list = []
        
        # Process each group and store results in a dictionary with simple column names
        for (study_name, respondent_name, start, end), group in overlaps.groupby(['Study Name', 'Respondent Name', 'Start', 'End']):
            stats = {
                'Study Name': study_name,
                'Respondent Name': respondent_name,
                'Start': start,
                'End': end
            }
            
            # Fixation Counts
            fixation_count = group['weight'].dropna().sum()
            stats['Fixation Counts'] = fixation_count if not pd.isna(fixation_count) else 0
            
            # Duration statistics
            valid_durations = group[duration_col].dropna()
            valid_weights = group['weight'][valid_durations.index]
            
            if len(valid_durations) > 0 and valid_weights.sum() > 0:
                stats['Average Fixation Duration'] = (valid_durations * valid_weights).sum() / valid_weights.sum()
                stats['Median Fixation Duration'] = valid_durations.median()
                stats['STD Fixation Duration'] = valid_durations.std() if len(valid_durations) > 1 else np.nan
            else:
                stats['Average Fixation Duration'] = np.nan
                stats['Median Fixation Duration'] = np.nan
                stats['STD Fixation Duration'] = np.nan
            
            # Dispersion statistics
            valid_dispersions = group[dispersion_col].dropna()
            valid_weights_disp = group['weight'][valid_dispersions.index]
            
            if len(valid_dispersions) > 0 and valid_weights_disp.sum() > 0:
                stats['Average Fixation Dispersion'] = (valid_dispersions * valid_weights_disp).sum() / valid_weights_disp.sum()
                stats['Median Fixation Dispersion'] = valid_dispersions.median()
                stats['STD Fixation Dispersion'] = valid_dispersions.std() if len(valid_dispersions) > 1 else np.nan
            else:
                stats['Average Fixation Dispersion'] = np.nan
                stats['Median Fixation Dispersion'] = np.nan
                stats['STD Fixation Dispersion'] = np.nan
                
            stats_list.append(stats)
        
        # Convert list of dictionaries to DataFrame
        stats_df = pd.DataFrame(stats_list)
        
        # Ensure consistent column order
        column_order = [
            'Study Name', 'Respondent Name', 'Start', 'End',
            'Fixation Counts', 
            'Average Fixation Duration', 'Median Fixation Duration', 'STD Fixation Duration',
            'Average Fixation Dispersion', 'Median Fixation Dispersion', 'STD Fixation Dispersion'
        ]
        # Use only columns that exist
        available_columns = [col for col in column_order if col in stats_df.columns]
        stats_df = stats_df[available_columns]
        
        return stats_df

class Pipeline:
    @staticmethod
    def run_pipeline(root_folder: str, timecard_path: str, perform_normalization: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Runs the entire preprocessing pipeline with progress tracking."""

        # Extract timecard name from path
        timecard_name = os.path.splitext(os.path.basename(timecard_path))[0]

        print("Loading and filtering timecard data...")
        fixation_aoi = CSVHandler.load_and_filter_timecard(timecard_path)
        
        # Store the original timecard DataFrame to preserve Label column
        original_fixation_aoi = fixation_aoi.copy()
        
        print("Processing bridge data...")
        bridge_data = {}
        for bridge in tqdm(['bridge_1', 'bridge_2', 'bridge_3', 'bridge_4'], desc="Processing bridges"):
            folder_path = os.path.join(root_folder, bridge)
            print(f"Processing {bridge}...")
            
            try:
                # Read and combine CSV files
                bridge_df = CSVHandler.combine_csv_files(folder_path)
                
                # Log participant counts before filtering
                print(f"Before participant filtering: {len(bridge_df)} rows")
                
                # Apply participant filtering
                bridge_df = DataProcessor.filter_participants(bridge_df, bridge)
                
                # Log after filtering
                print(f"After participant filtering: {len(bridge_df)} rows")
                
                # Rename bridges to standardized names
                bridge_df = DataProcessor.rename_bridges(bridge_df)
                
                # Memory optimization
                bridge_df = DataOptimizer.optimize_memory(bridge_df)
                
                # Store processed data
                bridge_data[bridge] = bridge_df
                
            except Exception as e:
                print(f"Error processing {bridge}: {str(e)}")
                continue
        
        if not bridge_data:
            raise ValueError("No bridge data was successfully processed")
            
        print("Combining all bridge data...")
        fixation_table = pd.concat(bridge_data.values(), ignore_index=True)
        
        # Log unique bridge names and participant counts
        print("Bridge name counts in fixation table:")
        print(fixation_table['Study Name'].value_counts())
        
        print("Participant counts per bridge:")
        for bridge in fixation_table['Study Name'].unique():
            count = fixation_table[fixation_table['Study Name'] == bridge]['Respondent Name'].nunique()
            print(f"{bridge}: {count} participants")

        # Optional normalization step
        if perform_normalization:
            print("Performing z-score normalization...")
            fixation_table = DataProcessor.normalize_zscore(fixation_table)
        
        print("Filtering outliers...")
        fixation_table = DataProcessor.filter_outliers_vectorized(fixation_table)

        # Plot filtered distributions
        try:
            print("Plotting filtered distributions...")
            Visualizer.plot_filtered_distributions(fixation_table, timecard_name)
        except Exception as e:
            print(f"Warning: Could not create visualizations: {str(e)}")
        
        print("Calculating fixation statistics...")
        fixation_aoi = FixationAnalyzer.calculate_fixation_statistics(fixation_aoi, fixation_table)
        
        
        # Merge Label column from original timecard data
        if 'Label' in original_fixation_aoi.columns:
            print("Merging Label column back to results")
            fixation_aoi = fixation_aoi.merge(
                original_fixation_aoi[['Study Name', 'Respondent Name', 'Start', 'End', 'Label']], 
                on=['Study Name', 'Respondent Name', 'Start', 'End'], 
                how='left'
            )
        
        fixation_aoi = CSVHandler.filter_base_and_fa_labels(fixation_aoi)
        
        return fixation_table, fixation_aoi

def main():
    """Main entry point for the pipeline."""
    try:
        root_folder = "/Users/bryce2hua/Desktop/HAL - Research/Data Preprocessing/data"
        timecard_folder = "/Users/bryce2hua/Desktop/HAL - Research/Data Preprocessing/timecards"
        
        # Create output directories if they don't exist
        os.makedirs("processed_fixation_tables", exist_ok=True)
        os.makedirs("processed_fixation_aoi", exist_ok=True)
        
        # Get all timecard files in the folder
        timecard_files = [f for f in os.listdir(timecard_folder) if f.endswith('.csv')]
        
        for timecard_file in timecard_files:
            print(f"\n{'='*50}")
            print(f"Processing timecard: {timecard_file}")
            print(f"{'='*50}")
            
            timecard_path = os.path.join(timecard_folder, timecard_file)
            
            # Get the base name of the timecard file without extension
            timecard_name = os.path.splitext(timecard_file)[0]
            
            # Run pipeline for this timecard
            fixation_table, fixation_aoi = Pipeline.run_pipeline(root_folder, timecard_path, perform_normalization=True)
            
            # Print summary statistics
            print("\nData Summary:")
            print(f"Total fixations processed: {len(fixation_table)}")
            print(f"Total intervals analyzed: {len(fixation_aoi)}")
            
            # Create output filenames based on timecard name
            table_output = f"processed_fixation_tables/processed_fixation_table_{timecard_name}.csv"
            aoi_output = f"processed_fixation_aoi/processed_fixation_aoi_{timecard_name}.csv"
            
            # Save results
            fixation_table.to_csv(table_output, index=False)
            fixation_aoi.to_csv(aoi_output, index=False)
            print(f"Saved results as {table_output} and {aoi_output}")
            
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()


class Visualizer:
   
   @staticmethod
   def setup_visualization_folders(timecard_name: str):
        """Create folders for different visualization stages and timecards"""
        base_path = Path("/Users/bryce2hua/Desktop/HAL - Research/Data Preprocessing/visualizations")
        folders = {
            'raw': base_path / 'raw_distributions' / timecard_name,
            'filtered': base_path / 'filtered_distributions' / timecard_name, 
            'final': base_path / 'final_distributions' / timecard_name
        }
        
        # Create folders if they don't exist
        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        return folders

   @staticmethod
   def plot_per_participant_distributions(data: pd.DataFrame, timecard_name: str, stage: str):
    """Plot distributions for each participant within each bridge"""
    folders = Visualizer.setup_visualization_folders(timecard_name)
    
    # Metrics to plot
    metrics = ['Fixation Duration', 'Fixation Dispersion'] if stage in ['raw', 'filtered'] else \
            ['Fixation Counts', 'Average Fixation Duration', 'Average Fixation Dispersion']
    
    # Iterate through bridges
    for bridge in data['Study Name'].unique():
        bridge_data = data[data['Study Name'] == bridge]
        
        # Iterate through participants in this bridge
        for participant in bridge_data['Respondent Name'].unique():
            participant_data = bridge_data[bridge_data['Respondent Name'] == participant]
            
            # Create figure for this participant
            plt.figure(figsize=(15, 5))
            
            # Create subplots for each metric
            for idx, metric in enumerate(metrics, 1):
                plt.subplot(1, len(metrics), idx)
                
                # Use appropriate plot type based on stage
                if stage in ['raw', 'filtered']:
                    # Check for normalized column
                    norm_col = f'{metric}_normalized'
                    if norm_col in participant_data.columns:
                        sns.histplot(data=participant_data, x=norm_col, kde=True)
                        plot_metric = norm_col
                    else:
                        sns.histplot(data=participant_data, x=metric, kde=True)
                        plot_metric = metric
                else:
                    plt.bar([0], [participant_data[metric].mean()])
                    plot_metric = metric
                
                plt.title(f'{plot_metric}\n{bridge} - Participant {participant}')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save to a path that includes bridge and participant
            output_path = folders[stage] / f'{bridge}_participant_{participant}_distribution.png'
            plt.savefig(output_path)
            plt.close()

   @staticmethod
   def plot_raw_distributions(data: pd.DataFrame, timecard_name: str):
        """Plot raw distributions per participant per bridge"""
        Visualizer.plot_per_participant_distributions(data, timecard_name, 'raw')

   @staticmethod
   def plot_filtered_distributions(data: pd.DataFrame, timecard_name: str):
        """Plot filtered distributions per participant per bridge"""
        Visualizer.plot_per_participant_distributions(data, timecard_name, 'filtered')

   @staticmethod
   def plot_final_distributions(data: pd.DataFrame, timecard_name: str):
        """Plot final distributions per participant per bridge"""
        Visualizer.plot_per_participant_distributions(data, timecard_name, 'final')

if __name__ == "__main__":
    main()