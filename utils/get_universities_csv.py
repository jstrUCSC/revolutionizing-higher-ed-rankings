import pandas as pd
import os
import random

# Path to the directory containing the CSV files
input_directory = "./data"  # Replace with your folder path
output_file = "/Users/mainoahmuna/Downloads/projects/revolutionizing-higher-ed-rankings/public/university_rankings.csv"  # Output CSV file

# Initialize an empty DataFrame for the combined data
combined_data = pd.DataFrame()

# Loop through all files in the directory
for file in os.listdir(input_directory):
    if file.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(input_directory, file)
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, usecols=["name", "affiliation", "homepage", "scholarid"])
            # Append to the combined DataFrame
            combined_data = pd.concat([combined_data, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading {file}: {e}")

# Drop duplicates based on the "affiliation" column (university name)
combined_data.drop_duplicates(subset=["affiliation"], inplace=True)

# Add the new columns to match the desired format
columns = [
    "Index","University","Artificial Intelligence & Machine Learning",
    "Data Science & Data Mining","Computer Vision & Image Processing",
    "Natural Language Processing","Systems and Networking","Databases","Security and Privacy",
    "Human Computer Interaction","Theoretical Computer Science","Software Engineering",
    "Computer Graphics & Virtual Reality","Quantum Computing","Interdisciplinary Fields"
]

# Create an empty DataFrame with the final structure
final_data = pd.DataFrame(columns=columns)

# Populate the "University" column and other columns
final_data["University"] = combined_data["affiliation"]
final_data["Index"] = range(1, len(final_data) + 1)  # Assign ranks automatically (1, 2, 3...)

# Fill all research-related fields with 0
for col in columns[2:]:  
    final_data[col] = [0 for _ in range(len(final_data))]

# Save the resulting DataFrame to a new CSV
final_data.to_csv(output_file, index=False)

print(f"Ranked universities CSV saved as {output_file}")