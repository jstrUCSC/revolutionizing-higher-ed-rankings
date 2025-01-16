# Use llama3-8b
import os
import re
import csv
import torch
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
from rankers.pairwise import PairwiseRanker


# Load model directly
def load_llama_model(model_name="meta-llama/Meta-Llama-3-8B", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def get_paper_title(content):
    lines = content.splitlines()
    title = lines[0].strip()
    return title


# def split_text(text, chunk_size=1024):
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# # Summarize all chunks
# def sum_paper_chunk(model, tokenizer, paper_content, max_new_tokens=200, device="cuda"):
#     chunks = split_text(paper_content, chunk_size=512)
#     summary_parts = []

#     for chunk in chunks:
#         prompt = f"Summarize the following chunk of a research paper in a concise manner:\n\n{chunk}\n\nSummary:"
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True)
        
#         outputs = model.generate(
#             inputs["input_ids"].to(device),
#             attention_mask=inputs["attention_mask"].to(device),
#             max_new_tokens=max_new_tokens,
#             temperature=0.7,
#             top_p=0.8,
#             repetition_penalty=1.2,         # decrease repeat
#             pad_token_id=tokenizer.pad_token_id
#         )
#         summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         summary_parts.append(summary)
    
#     return " ".join(summary_parts)


# # Step 1: Generate individual summaries for each paper using chunk summaries
# def generate_paper_sum(model, tokenizer, papers, titles, max_new_tokens=300, device="cuda"):
#     summaries = []

#     for i, paper_content in enumerate(papers):
#         chunk_summary = sum_paper_chunk(model, tokenizer, paper_content, max_new_tokens=200, device=device)

#         prompt = f"Based on the following summaries of the research paper, provide a concise overall summary:\n\n{chunk_summary}\n\nOverall Summary:"
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, padding=True)
        
#         outputs = model.generate(
#             inputs["input_ids"].to(device),
#             attention_mask=inputs["attention_mask"].to(device),
#             max_new_tokens=max_new_tokens,
#             temperature=0.7,
#             top_p=0.8,
#             repetition_penalty=1.2,
#             pad_token_id=tokenizer.pad_token_id
#         )
#         overall_summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#         summaries.append((titles[i], overall_summary))

#     return summaries


# Only extract abstract and conclusion sections
def extract_abstract_and_conclusion(content):
    abstract, conclusion = "", ""

    lines = content.splitlines()
    abstract_start = None
    conclusion_start = None

    # Find the positions of "abstract" and "conclusion"
    for i, line in enumerate(lines):
        if 'abstract' in line.lower() and abstract_start is None:
            abstract_start = i
        if 'conclusion' in line.lower() and conclusion_start is None:
            conclusion_start = i

    # Extract abstract
    if abstract_start is not None:
        abstract_lines = []
        for i in range(abstract_start, len(lines)):
            if lines[i].strip() == "" or len(abstract_lines) > 50:
                break
            abstract_lines.append(lines[i])
        abstract = " ".join(abstract_lines)

    # Extract conclusion
    if conclusion_start is not None:
        conclusion_lines = []
        for i in range(conclusion_start, len(lines)):
            if len(conclusion_lines) > 50:
                break
            conclusion_lines.append(lines[i])
        conclusion = " ".join(conclusion_lines)

    return abstract, conclusion



# Step 2: ranking
# def rank_papers(model, tokenizer, paper_details, max_new_tokens=500, device="cuda"):
#     prompt = ("Compare the following research papers based on their abstracts and conclusions. "
#               "Rank the papers from best to worst and provide a brief justification for your ranking:\n\n")

#     for i, (title, details) in enumerate(paper_details):
#         prompt += f"Paper {i+1}: {title}\n{details}\n\n"


#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096, padding=True)
#     outputs = model.generate(
#         inputs["input_ids"].to(device),
#         attention_mask=inputs["attention_mask"].to(device),
#         max_new_tokens=max_new_tokens,
#         temperature=0.7,
#         top_p=0.8,
#         repetition_penalty=1.2,
#         pad_token_id=tokenizer.pad_token_id
#     )

#     rank_result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#     print("Ranking Result:", rank_result)

#     rankings = []
#     seen_titles = set()

#     for line in rank_result.splitlines():
#         line = line.strip()
#         match = re.match(r"(\d+)\.\s*Paper\s*(\d+)", line)
#         if match:
#             rank = int(match.group(1))
#             index = int(match.group(2))
#             if index <= len(paper_details):
#                 title = paper_details[index - 1][0]
#                 if title not in seen_titles:
#                     seen_titles.add(title)
#                     rankings.append((rank, title))

#     if len(rankings) == 0:
#         rankings = [(i+1, paper_details[i][0]) for i in range(len(paper_details))]

#     return sorted(rankings, key=lambda x: x[0])

def rank_papers_pairwise(paper_details, pairwise_ranker: PairwiseRanker):
    n = len(paper_details)
    scores = [0] * n

    # Perform pairwise comparison for all pairs using PairwiseRanker
    for i in range(n):
        for j in range(i + 1, n):
            result = pairwise_ranker.compare(paper_details[i][1], paper_details[j][1])
            if result == "A":
                scores[i] += 1
            else:
                scores[j] += 1

    # Create rankings based on scores
    rankings = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranked_papers = [(rank + 1, paper_details[i][0]) for rank, (i, _) in enumerate(rankings)]
    return ranked_papers

# Save rankings to CSV
def save_to_csv(rankings, output_file="rankings.csv"):
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Rank", "Title"])
        for rank, title in rankings:
            writer.writerow([rank, title])


def main():
    pdf_paths = ["308.pdf", "tot.pdf"]  # Test PDF papers
    print("Reading PDF...")

    # Loading model
    print("Loading LLama3-8b model...")
    tokenizer, model = load_llama_model(model_name="meta-llama/Meta-Llama-3-8B")

    papers = []
    titles = []

    for pdf_path in pdf_paths:
        content = read_pdf(pdf_path)
        title = get_paper_title(content)
        papers.append(content)
        titles.append(title)

    # Extract summaries from abstracts and conclusions
    print("Extracting abstracts and conclusions for each paper...")
    paper_details = []

    for i, paper_content in enumerate(papers):
        abstract, conclusion = extract_abstract_and_conclusion(paper_content)
        combined_summary = f"Abstract: {abstract}\nConclusion: {conclusion}"
        paper_details.append((titles[i], combined_summary))

    print("Details extracted:\n\n", paper_details)

    # Initialize PairwiseRanker from pairwise.py
    print("Initializing PairwiseRanker...")
    pairwise_ranker = PairwiseRanker(model, tokenizer)

    # Rank papers using pairwise comparison
    print("Ranking papers using pairwise comparison...")
    rankings = rank_papers_pairwise(paper_details, pairwise_ranker)

    # Save rankings to CSV
    save_to_csv(rankings, output_file="paper_rankings.csv")
    print("Ranking completed. Results saved to 'paper_rankings.csv'.")


if __name__ == "__main__":
    main()

