import os
import re
import csv
import torch
import argparse
import shutil  # Added for file moving
import json
import requests
from typing import List, Tuple
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model

def select_and_rank_references(
    model, 
    tokenizer, 
    main_content: str, 
    references: List[str], 
    device="cuda"
) -> List[Tuple[int, str]]:

    references_text = "\n".join(
        [info["text"] for _, info in sorted(references.items(), key=lambda x: int(x[0]))]
    )

    rich_prompt = f"""
        ########
        # CONTEXT #
        You are an expert literature analysis AI trained to interpret how research papers
        build upon prior work. You will receive an abstract and a list of references from
        a scientific publication. Each reference has a unique number in brackets [X].
        ########

        # OBJECTIVE #
        Your goal is to identify which prior works (references) are most *important* to the paper.
        "Important" means that the paper heavily depends on, builds upon, or directly extends that work.
        Conversely, references cited only for background information, general comparison,
        or minor support are *less important*.

        ########
        # INPUT #
        Paper Abstract:
        \"\"\"{main_content}\"\"\"

        References:
        {references_text}
        ########

        # TASK DESCRIPTION #
        1. Read the abstract carefully and infer which references are most central to the paper’s ideas.
        2. Consider how the paper might:
        - Reuse or extend a method or dataset from a cited work.
        - Build on a theoretical framework or model from a cited work.
        - Directly compare against or improve upon a cited baseline.
        3. Rank the references in order of importance to the paper.

        ########
        # RESPONSE RULES #
        - You must output short one-sentence explanations for the **five most important references**.
        - For each selected reference, include its number in brackets [X] and explain *why* it is important.
        - After all explanations, output a final ranked list in the **exact format below**:

        Rank 1. [X]
        Rank 2. [Y]
        Rank 3. [Z]
        Rank 4. [A]
        Rank 5. [B]

        - Do not include any other text or commentary after the ranked summary.
        - If the paper appears to depend on a particular model, dataset, or theoretical paper, that reference should rank higher.

        ########
        # OUTPUT FORMAT #
        1. [X] One-sentence explanation.
        2. [Y] One-sentence explanation.
        3. [Z] One-sentence explanation.
        4. [A] One-sentence explanation.
        5. [B] One-sentence explanation.

        Then:
        Rank 1. [X]
        Rank 2. [Y]
        Rank 3. [Z]
        Rank 4. [A]
        Rank 5. [B]
        ########
        """
    
    prompt = f"""
        ########
        # CONTEXT #
        You are an expert literature analysis AI trained to identify which prior works
        are most central to a research paper's ideas and contributions.
        ########

        # OBJECTIVE #
        Given an abstract and a list of numbered references, determine which cited works
        the paper most heavily depends on, builds upon, or extends. 
        Less important references are those cited only for general background or minor support.

        ########
        # INPUT #
        Paper Abstract:
        \"\"\"{main_content}\"\"\"

        References:
        {references_text}
        ########

        # RESPONSE RULES #
        - Output short, one-sentence explanations for the **five most important references**.
        - Each explanation must begin with the reference number in brackets [X].
        - After the five explanations, output a ranked summary in the **exact format** below:
        Rank 1. [X]
        Rank 2. [Y]
        Rank 3. [Z]
        Rank 4. [A]
        Rank 5. [B]
        - Do not include any other commentary, notes, or extra text.

        ########
        # OUTPUT FORMAT #
        1. [X] One-sentence explanation.
        2. [Y] One-sentence explanation.
        3. [Z] One-sentence explanation.
        4. [A] One-sentence explanation.
        5. [B] One-sentence explanation.

        Then:
        Rank 1. [X]
        Rank 2. [Y]
        Rank 3. [Z]
        Rank 4. [A]
        Rank 5. [B]
        ########
        """
    
    simple_prompt = f"""
        You are an AI that ranks references by importance to a paper.

        Important = the paper builds on, extends, or depends on that work.
        Less important = background or minor citations.

        Abstract:
        {main_content}

        References:
        {references_text}

        Write one short sentence for the 5 most important references.
        Start each with its number in brackets [X].
        Then list the final ranking in this exact format:
        Rank 1. [X]
        Rank 2. [Y]
        Rank 3. [Z]
        Rank 4. [A]
        Rank 5. [B]
        """
    
    inputs = tokenizer(
        rich_prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048, 
        padding=True
    ).to(device)
    
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=1500, 
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
        do_sample=False        
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    rank_pattern = r"(?:Rank|Top|#)\s*(\d+)[:\.]?\s*\[(\d+)\]"
    rank_matches = re.findall(rank_pattern, response)
    
    selected_refs = []

    if rank_matches:
        rank_refs = sorted([(int(rank), int(idx)) for rank, idx in rank_matches])
        for _, ref_idx in rank_refs:
            ref_text = references.get(str(ref_idx), {}).get("text", "")
            if ref_text:
                selected_refs.append((ref_idx, ref_text))

    if len(selected_refs) < 5:
        list_pattern = r"(?:^|\n)\s*(\d+)\.?\s*\[(\d+)\]"
        list_matches = re.findall(list_pattern, response)
        for _, ref_idx_str in list_matches:
            ref_idx = int(ref_idx_str)
            if any(idx == ref_idx for idx, _ in selected_refs):
                continue
            ref_text = references.get(str(ref_idx), {}).get("text", "")
            if ref_text:
                selected_refs.append((ref_idx, ref_text))
            if len(selected_refs) >= 5:
                break

    if len(selected_refs) < 5:
        ref_pattern = r"\[(\d+)\]"
        ref_matches = re.findall(ref_pattern, response)
        for ref_idx_str in ref_matches:
            ref_idx = int(ref_idx_str)
            if any(idx == ref_idx for idx, _ in selected_refs):
                continue
            ref_text = references.get(str(ref_idx), {}).get("text", "")
            if ref_text:
                selected_refs.append((ref_idx, ref_text))
            if len(selected_refs) >= 5:
                break

    if len(selected_refs) < 5:
        # Fallback: fill with remaining refs in order
        for ref_idx_str, ref_info in sorted(references.items(), key=lambda x: int(x[0])):
            ref_idx = int(ref_idx_str)
            if any(idx == ref_idx for idx, _ in selected_refs):
                continue
            ref_text = ref_info.get("text", "")
            if ref_text:
                selected_refs.append((ref_idx, ref_text))
            if len(selected_refs) >= 5:
                break
    
    return selected_refs[:5] 

def save_to_csv(references, selected_refs, output_file="selected_references.csv"):
    with open(output_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Rank", "Reference", "Author Count"])
        for rank, (index, ref) in enumerate(selected_refs, start=1):
            ref_single_line = references[str(index)].get("text", "ERROR text not found")
            author_count = references[str(index)].get("author_count", 0)
            writer.writerow([rank, ref_single_line, author_count])

def ref_to_string(ref):
    title = ref.get("title", "").strip()
    authors = ref.get("authors", [])
    date = ref.get("imprint", {}).get("date", "").strip()

    # Format authors as "A, B, and C" or empty if missing
    authors_str = ", ".join(authors) if authors else "Unknown authors"
    date_str = f"({date})" if date else "(n.d.)"  # n.d. = no date

    # Compose concise formatted reference
    return f"{authors_str}. {title} {date_str}"

def process_json_paper(paper: dict, model, tokenizer, device, output_csv):
    """Process a paper represented as a JSON dict instead of a PDF."""
    # Extract basic metadata
    title = paper.get("paper_title") or paper.get("extracted_paper_title") or "Unknown Title"
    filename = os.path.basename(paper.get("source_pdf", "unknown.pdf"))
    print(f"\nProcessing JSON: {filename}")
    print(f"Paper Title: {title}")

    # Extract references list
    references_data = paper.get("references", [])
    if not references_data:
        print("No references found. Skipping.")
        return

    # Convert to simple text strings for ranking
    references = {}
    
    for ref in references_data:
        ref_string = ref_to_string(ref)
        number = str(int(ref.get("reference_id", "-1").replace("b", "")) + 1)
        formatted_ref = f"[{number}] {ref_string}"
        references[number] = {
            "text": formatted_ref,
            "author_count": len(ref.get("authors", []))
        }
 

    print(f"Total {len(references)} references found.")

    if len(references.keys()) < 5:
        print("References are fewer than 5. Skipping ranking.")
        return

    # Optionally extract "main content" if available (citation contexts, abstract, etc.)
    main_content = paper.get("abstract", title)

    # Rank and select top references

    selected_refs = select_and_rank_references(model, tokenizer, main_content, references, device=device)

    if len(selected_refs) < 5:
        print("Failed to select five references.")
    else:
        save_to_csv(references, selected_refs, output_file=output_csv)
        print(f"Top 5 references saved to {output_csv}")
        print("Top References:")
        for rank, (index, ref) in enumerate(selected_refs, start=1):
            print(f"{rank}. {ref[:100]}...")

def main():
    json_folder = "sample_json"
    output_csv = "sample_json/references.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_papers = []

    # model, tokenizer = load_model(device=device)
    model, tokenizer = None, None

    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            with open(os.path.join(json_folder, filename), "r", encoding="utf-8") as f:
                all_papers.append(json.load(f))

    for paper in all_papers:
        process_json_paper(paper, model, tokenizer, device, output_csv)

if __name__ == "__main__":
    main()