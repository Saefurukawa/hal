import os
import pandas as pd
import re
from typing import Dict, Set, Union, List, Tuple
from dataclasses import dataclass, field

@dataclass
class Config:
    BRIDGES: list = field(default_factory=lambda: ["Bridge 1", "Bridge 2", "Bridge 3", "Bridge 4"])
    DISTANCE_FILE: str = "combined_distance_table.csv"
    # CHANGE: the name of your result folder (containing all final tables of all timecard versions)
    RESULT_FOLDERS: str = "processed_fixation_aoi"

class CrackDataProcessor:
    def __init__(self, result_file: str):
        self.config = Config()  # Create an instance of Config
        self.config.RESULT_FILE = result_file
    
    @staticmethod
    def extract_number(aoi_label: str) -> Union[int, None]:
        """Extract number from crack label."""
        match = re.search(r'\d+', aoi_label)
        return int(match.group()) if match else None

    def initialize_bridge_dict(self) -> Dict[str, dict]:
        """Initialize dictionary for bridges."""
        return {bridge: {} for bridge in self.config.BRIDGES}

    def process_distance_data(self) -> Dict[str, Dict[int, Set[int]]]:
        """Process distance data to get expected crack information."""
        data_df = pd.read_csv(self.config.DISTANCE_FILE)
        bridge_participant_cracks = self.initialize_bridge_dict()

        # Filter rows that contain "crack" in the Label column (case insensitive)
        crack_rows = data_df[data_df["Label"].str.contains("crack", case=False, na=False)]
        
        # Extract crack numbers
        crack_rows["Crack Number"] = crack_rows["Label"].apply(self.extract_number)

        # Group by bridge and participant, ensuring 0 is not ignored
        for (bridge, participant), group in crack_rows.groupby(["Study Name", "Respondent Name"]):
            valid_cracks = group["Crack Number"].dropna().astype(int)  # Keep all valid numbers
            if not valid_cracks.empty:
                bridge_participant_cracks[bridge][int(participant)] = set(valid_cracks)

        return bridge_participant_cracks

    def process_result_data(self) -> Dict[str, Dict[int, Set[int]]]:
        """Process result data to get existing crack information."""
        data_df = pd.read_csv(self.config.RESULT_FILE)
        existing_cracks = self.initialize_bridge_dict()

        for _, row in data_df.iterrows():
            try:
                participant = int(row["Respondent Name"])
                bridge = row["Study Name"]
                
                # Check for both Label and AOI Label columns
                if "Label" in data_df.columns:
                    aoi_label = row["Label"]
                elif "AOI Label" in data_df.columns:
                    aoi_label = row["AOI Label"]
                else:
                    continue  # Skip if neither label column exists
                
                if "base" in str(aoi_label).lower() or "crack" not in str(aoi_label).lower():
                    continue

                crack_number = self.extract_number(str(aoi_label))
                if crack_number:
                    if participant not in existing_cracks[bridge]:
                        existing_cracks[bridge][participant] = set()
                    existing_cracks[bridge][participant].add(crack_number)
            except (ValueError, TypeError):
                # Skip rows with invalid data
                continue

        return existing_cracks

    def find_missing_participants(self, expected: Dict, existing: Dict) -> Dict[str, Set[int]]:
        """Find participants that are in expected data but not in existing data."""
        missing_participants = self.initialize_bridge_dict()
        for bridge in expected:
            missing_participants[bridge] = set(expected[bridge].keys()) - set(existing[bridge].keys())
        return missing_participants
    
    def find_extra_participants(self, expected: Dict, existing: Dict) -> Dict[str, Set[int]]:
        """Find participants that are in existing data but not in expected data."""
        extra_participants = self.initialize_bridge_dict()
        for bridge in existing:
            extra_participants[bridge] = set(existing[bridge].keys()) - set(expected[bridge].keys())
        return extra_participants

    def calculate_coverage_stats(self, expected: Dict, existing: Dict) -> Tuple[int, int, float, Dict[str, float]]:
        """Calculate coverage statistics."""
        expected_count = 0
        existing_count = 0
        bridge_coverage = {}
        
        for bridge in expected:
            bridge_expected = 0
            bridge_existing = 0
            
            for participant in expected[bridge]:
                expected_entries = expected[bridge][participant]
                bridge_expected += len(expected_entries)
                
                if participant in existing[bridge]:
                    existing_entries = existing[bridge][participant]
                    bridge_existing += len(existing_entries)
            
            expected_count += bridge_expected
            existing_count += bridge_existing
            
            if bridge_expected > 0:
                bridge_coverage[bridge] = bridge_existing / bridge_expected
            else:
                bridge_coverage[bridge] = 0.0
                
        overall_coverage = existing_count / expected_count if expected_count > 0 else 0.0
        
        return expected_count, existing_count, overall_coverage, bridge_coverage

    def analyze_differences(self, expected: Dict, existing: Dict) -> None:
        """Analyze differences between expected and existing data."""
        print(f"\n--- Analysis for {os.path.basename(self.config.RESULT_FILE)} ---")
        
        # Find missing and extra participants
        missing_participants = self.find_missing_participants(expected, existing)
        extra_participants = self.find_extra_participants(expected, existing)
        
        # Check for missing participants
        print("\nChecking missing participants:")
        for bridge in missing_participants:
            if missing_participants[bridge]:
                print(f"\n{bridge} missing participants: {sorted(missing_participants[bridge])}")
            else:
                print(f"\n{bridge}: No missing participants")
        
        # Check for extra participants
        print("\nChecking extra participants:")
        for bridge in extra_participants:
            if extra_participants[bridge]:
                print(f"\n{bridge} extra participants: {sorted(extra_participants[bridge])}")
            else:
                print(f"\n{bridge}: No extra participants")

        # Check for missing cracks
        print("\nChecking missing cracks per participant:")
        missing_cracks = self.initialize_bridge_dict()
        extra_cracks = self.initialize_bridge_dict()

        for bridge in expected:
            print(f"\n{bridge}:")
            for participant in expected[bridge]:
                if participant not in existing[bridge]:
                    print(f"Participant {participant}: ALL CRACKS MISSING - {expected[bridge][participant]}")
                    continue
                    
                expected_entries = expected[bridge][participant]
                existing_entries = existing[bridge][participant]
                
                missing_cracks[bridge][participant] = expected_entries - existing_entries
                extra_cracks[bridge][participant] = existing_entries - expected_entries
                
                if missing_cracks[bridge][participant]:
                    print(f"Participant {participant}: Missing cracks {sorted(missing_cracks[bridge][participant])}")
                
        # Check for extra cracks
        print("\nChecking extra cracks per participant:")
        for bridge in extra_cracks:
            print(f"\n{bridge}:")
            for participant in extra_cracks[bridge]:
                if extra_cracks[bridge][participant]:
                    print(f"Participant {participant}: Extra cracks {sorted(extra_cracks[bridge][participant])}")
        
        # Calculate coverage statistics
        expected_count, existing_count, overall_coverage, bridge_coverage = self.calculate_coverage_stats(expected, existing)
        
        print(f"\nCoverage Statistics:")
        print(f"Total expected cracks: {expected_count}")
        print(f"Total existing cracks: {existing_count}")
        print(f"Overall coverage ratio: {overall_coverage:.2%}")
        
        print("\nCoverage by bridge:")
        for bridge, coverage in bridge_coverage.items():
            print(f"{bridge}: {coverage:.2%}")
    
    def check_nan_percentages(self) -> pd.DataFrame:
        """
        Check the percentage of NaN values for each column in the result file.
        
        Returns:
        pd.DataFrame: A DataFrame with columns and their NaN percentages
        """
        # Read the result file
        data_df = pd.read_csv(self.config.RESULT_FILE)
        
        # Calculate NaN percentages
        nan_percentages = {}
        for column in data_df.columns:
            # Calculate percentage of NaN values
            nan_percentage = data_df[column].isna().mean() * 100
            nan_percentages[column] = nan_percentage
        
        # Convert to DataFrame for better visualization
        nan_summary = pd.DataFrame.from_dict(
            nan_percentages, 
            orient='index', 
            columns=['NaN Percentage']
        )
        nan_summary.index.name = 'Column'
        nan_summary.sort_values('NaN Percentage', ascending=False, inplace=True)
        
        # Print the results
        print(f"\nNaN Percentages in {os.path.basename(self.config.RESULT_FILE)}:")
        print(nan_summary)
        print("Total number of rows:", len(data_df))
        
        return nan_summary

def main():
    # Get the result folder path
    result_folder = Config().RESULT_FOLDERS
    
    # Find all CSV files in the result folder
    result_files = [
        os.path.join(result_folder, f) 
        for f in os.listdir(result_folder) 
        if f.endswith('.csv')
    ]
    
    # Process each result file
    for result_file in result_files:
        processor = CrackDataProcessor(result_file)
        
        # Process data
        expected_cracks = processor.process_distance_data()
        existing_cracks = processor.process_result_data()
        
        # Analyze differences
        processor.analyze_differences(expected_cracks, existing_cracks)

        # Check NaN percentages
        processor.check_nan_percentages()

if __name__ == "__main__":
    main()
