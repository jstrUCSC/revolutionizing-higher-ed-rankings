# Use llama3-70b
import os
import re
import csv
import torch
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate


# Load model directly
def load_llama_model(model_name="meta-llama/Meta-Llama-3-70B", device="cuda"):        
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


def split_text(text, chunk_size=1024):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# def split_long_text(text, tokenizer, chunk_size=1024):
#     tokens = tokenizer.encode(text)
#     return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]


def get_paper_title(content):
    lines = content.splitlines()
    title = lines[0].strip()
    return title

# Summarize all chunks
def sum_paper_chunk(model, tokenizer, paper_content, max_new_tokens=1000, device="cuda"):
    chunks = split_text(paper_content)
    summary_parts = []

    for chunk in chunks:
        prompt = f"Summarize the following chunk of a research paper:\n\n{chunk}\n\nSummary:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096, padding=True)
        
        outputs = model.generate(
            inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary_parts.append(summary)
    
    return " ".join(summary_parts)


# Step 1: Generate individual summaries for each paper
def generate_paper_sum(model, tokenizer, papers, titles, max_new_tokens=1000, device="cuda"):
    summaries = []

    for i, paper_content in enumerate(papers):
        prompt = f"Read the following research paper and provide a concise summary:\n\n{paper_content}\n\nSummary:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096, padding=True)
        
        outputs = model.generate(
            inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append((titles[i], summary))
    
    return summaries


# Step 2: ranking
def rank_papers(model, tokenizer, summaries, max_new_tokens=1000, device="cuda"):
    # combined_summaries = []
    prompt = "Compare the following research papers based on their summaries and rank them from best to worst. Provide a brief justification:\n\n"
    
    # for paper_content in papers:
    #     combined_summary = sum_paper_chunk(model, tokenizer, paper_content, max_new_tokens, device)
    #     combined_summaries.append(combined_summary)

    for i, (title, summary) in enumerate(summaries):
        prompt += f"Paper {i+1}: {title}\nSummary: {summary}\n\n"


    prompt += "Based on the above summaries, rank the papers from best to worst in the following format:\n\n" \
              "1. Paper X: <reason>\n" \
              "2. Paper Y: <reason>\n" \
              "3. Paper Z: <reason>\n"

    # for i, summary in enumerate(combined_summaries):
    #     prompt += f"Paper {i+1} Summary:\n{summary}\n\n"
    # prompt += "Please rank these papers and explain your reasoning:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096, padding=True)
    outputs = model.generate(
        inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
    rank_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Ranking Result:", rank_result)

    rankings = []
    seen_titles = set()

    for line in rank_result.splitlines():
        line = line.strip()
        # if re.match(r"Paper \d+", line):
        #     # Extract paper index and corresponding title
        #     match = re.search(r"Paper (\d+)", line)
        match = re.match(r"(\d+)\.\s*Paper\s*(\d+)", line)
        if match:
            rank = int(match.group(1))
            index = int(match.group(2))
            if index <= len(summaries):
                title = summaries[index - 1][0]
                if title not in seen_titles:
                    seen_titles.add(title)
                    rankings.append((rank, title))

    # if len(rankings) < len(titles):
    #     missing_titles = [title for title in titles if title not in seen_titles]
    #     for index, title in enumerate(missing_titles, start=len(rankings) + 1):
    #         rankings.append((index, title))
                    
    # If no ranking could be determined, default to the order of summaries
    if len(rankings) == 0:
        rankings = [(i+1, summaries[i][0]) for i in range(len(summaries))]

    return sorted(rankings, key=lambda x: x[0])


def save_to_csv(rankings, output_file="rankings.csv"):
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Title"])
        for rank, (index, title) in enumerate(rankings, start=1):
            writer.writerow([rank, title])


# # Example: show text
# def analyze_text(model, tokenizer, text, device="cuda", max_length=4096):
#     inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
#     outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Jist test
def main():
    pdf_paths = ["212.pdf", "117.pdf", "308.pdf", "tot.pdf"]       # test pdf paper
    print("Reading PDF...")
    # pdf_text = read_pdf(pdf_path)

    # Loading model
    print("Loading LLama3-70b model...")
    tokenizer, model = load_llama_model(model_name="meta-llama/Meta-Llama-3-70B")

    papers = []
    titles = []

    for pdf_path in pdf_paths:
        content = read_pdf(pdf_path)
        title = get_paper_title(content)
        print(f"Content from {pdf_path}:\n", content[:500])  
        papers.append(content)
        titles.append(title)

    rankings = rank_papers(model, tokenizer, papers, titles)
    save_to_csv(rankings, output_file="paper_rankings.csv")
    print("Ranking completed. Results saved to 'paper_rankings.csv'.")

    # # Analyzing
    # print("Analyzing text...")
    # summarized_text = analyze_text(model, tokenizer, pdf_text)
    # print("\nSummarized Content:\n", summarized_text)

    # pass

if __name__ == "__main__":
    main()

