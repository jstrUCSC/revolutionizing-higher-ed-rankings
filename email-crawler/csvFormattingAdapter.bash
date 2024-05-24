#!/bin/bash

# Define the input CSV file
input_csv="u_scores.csv"
# Define the output CSV file
output_csv="u_scores_filled.csv"

# Read the header line to determine the number of columns
header=$(head -n 1 "$input_csv")

# Use awk to process the file and fill empty fields with 0
awk -F, -v OFS=, '
BEGIN {
    # Read the header and determine the number of columns
    header = getline h < "'"$input_csv"'"
    split(h, headers, FS)
    num_columns = length(headers)
    print h > "'"$output_csv"'"
}
NR > 1 {
    # For each row, check each field and fill empty fields with 0
    for (i = 1; i <= num_columns; i++) {
        if ($i == "") {
            $i = 0
        }
    }
    print $0
}' "$input_csv" >> "$output_csv"

echo "Processed CSV file has been saved to $output_csv"

