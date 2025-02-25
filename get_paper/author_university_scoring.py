import pandas as pd

def rank_universities_by_publications():
    #Load the author-university CSV file
    input_file = "author_universities_output.csv"
    df = pd.read_csv(input_file)

    #Count occurrences of each university (number of publications)
    university_counts = df["University"].value_counts().reset_index()
    university_counts.columns = ["University", "Publications"]

    #Save the ranked universities to an output CSV file
    output_file = "university_publications_ranking.csv"
    university_counts.to_csv(output_file, index=False)

    print(f"University publication rankings saved to {output_file}")

    return university_counts

#Run the function
university_ranking_df = rank_universities_by_publications()
