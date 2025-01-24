import pandas as pd
import os

# Path to the directory containing the CSV files
input_directory = "utils/data"  # Replace with your folder path
output_file = "universities_ranked.csv"  # Output CSV file

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

# Drop duplicates
combined_data.drop_duplicates(inplace=True)

# Add the new columns to match the desired format
columns = [
    "Rank", "University", "Artificial intelligence", "Computer systems and networks", "Cybersecurity",
    "Databases and data mining", "Digital Libraries", "Human computer interaction", "Machine Learning",
    "Medical Image Computing", "Natural Language Processing", "Parallel Computing", "Program Analysis",
    "Programming Languages", "Programming languages and verification", "Vision and graphics"
]

# Create an empty DataFrame with the final structure
final_data = pd.DataFrame(columns=columns)

# Populate the "University" column and other columns
final_data["University"] = combined_data["affiliation"]
final_data["Rank"] = range(1, len(final_data) + 1)  # Assign ranks automatically (1, 2, 3...)

# Fill all research-related fields with 0
for col in columns[2:]:
    final_data[col] = 0

# Save the resulting DataFrame to a new CSV
final_data.to_csv(output_file, index=False)

print(f"Ranked universities CSV saved as {output_file}")