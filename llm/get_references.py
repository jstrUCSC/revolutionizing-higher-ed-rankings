import os
import re
import csv
import json
import torch
import argparse
from typing import List, Tuple
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model

def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def get_paper_title(content: str) -> str:
    lines = content.splitlines()
    for line in lines:
        if line.strip():
            return line.strip()
    return "Untitled"

#############################
# Format detection and segmentation

def detect_reference_format(references_text: str) -> str:
    """
    bracketed: [1] ...
    parenthesized: (1) ...
    numberdot: 1. ... or 1) ...
    no index
    """
    lines = references_text.splitlines()
    
    bracketed_count = 0
    parenthesized_count = 0
    numberdot_count = 0
    
    for line in lines:
        line_stripped = line.strip()
        # [1]
        if re.match(r'^\[\d+\]\s', line_stripped):
            bracketed_count += 1
        # (1)
        elif re.match(r'^\(\d+\)\s', line_stripped):
            parenthesized_count += 1
        # 1. or 1)
        elif re.match(r'^\d+(\.|\))\s', line_stripped):
            numberdot_count += 1

    if bracketed_count >= 3:
        return "bracketed"
    elif parenthesized_count >= 3:
        return "parenthesized"
    elif numberdot_count >= 3:
        return "numberdot"
    else:
        return "none"

def parse_bracketed_references(references_text: str) -> list:
    """
    [1], [2]
    """
    raw_refs = re.split(r'(?=^\[\d+\])', references_text, flags=re.MULTILINE)
    references = []
    for ref_chunk in raw_refs:
        ref_chunk = ref_chunk.strip()
        if ref_chunk:
            references.append(ref_chunk)
    return references

def parse_parenthesized_references(references_text: str) -> list:
    """
    (1), (2) 
    """
    raw_refs = re.split(r'(?=^\(\d+\))', references_text, flags=re.MULTILINE)
    references = []
    for ref_chunk in raw_refs:
        ref_chunk = ref_chunk.strip()
        if ref_chunk:
            references.append(ref_chunk)
    return references

def parse_numberdot_references(references_text: str) -> list:
    """
    1. or 1)
    """
    raw_refs = re.split(r'(?=^\d+(\.|\))\s)', references_text, flags=re.MULTILINE)
    references = []
    for ref_chunk in raw_refs:
        ref_chunk = ref_chunk.strip()
        if ref_chunk:
            references.append(ref_chunk)
    return references

def heuristic_line_split_references(references_text: str) -> list:
    lines = references_text.strip().splitlines()
    references = []
    current_ref_lines = []

    def is_new_reference_line(line: str) -> bool:
        if not line:
            return False
        return line[0].isupper() or line[0].isdigit()

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue  # Skip blank line

        if current_ref_lines and is_new_reference_line(line_stripped):
            full_ref = " ".join(current_ref_lines).strip()
            if len(full_ref) > 5:
                references.append(full_ref)
            current_ref_lines = [line_stripped]
        else:
            current_ref_lines.append(line_stripped)

    if current_ref_lines:
        full_ref = " ".join(current_ref_lines).strip()
        if len(full_ref) > 5:
            references.append(full_ref)
    return references

def parse_noindex_references(references_text: str) -> list:
    """
    no index
    """
    pattern = r"\.\s*\n\s*\n+(?=[A-Z][a-zA-Z-]+ [A-Z]|[A-Z]\. [A-Z][a-zA-Z-]+)"

    # raw_refs = re.split(r'\.\s*\n\s*\n+', references_text.strip())
    # raw_refs = [ref.strip() + '.' for ref in raw_refs if ref.strip()]
    raw_refs = re.split(pattern, references_text.strip())
    raw_refs = [ref.strip() for ref in raw_refs if ref.strip()]

    references = []
    for ref_chunk in raw_refs:
        ref_chunk = ref_chunk.strip()
        if len(ref_chunk) > 5:
            lines_in_chunk = ref_chunk.splitlines()
            one_ref = " ".join(l.strip() for l in lines_in_chunk if l.strip())
            references.append(one_ref)

    if len(references) <= 1:
        # fallback
        references = heuristic_line_split_references(references_text)
    return references

def parse_references(references_text: str) -> list:
    # remove "References"
    lines = references_text.strip().split("\n", 1)
    if len(lines) > 1:
        if re.match(r'(?i)^\s*references\s*$', lines[0]):
            references_text = lines[1]

    fmt = detect_reference_format(references_text)
    if fmt == "bracketed":
        references = parse_bracketed_references(references_text)
    elif fmt == "parenthesized":
        references = parse_parenthesized_references(references_text)
    elif fmt == "numberdot":
        references = parse_numberdot_references(references_text)
    else:
        references = parse_noindex_references(references_text)

    clean_refs = []
    for ref in references:
        r = ref.strip()
        # remove "[1]", "(1)", "1.", "1)" ...
        r = re.sub(r'^(?:\[\d+\]|\(\d+\)|\d+\.|\d+\))\s*', '', r)
        clean_refs.append(r)

    final_refs = [r for r in clean_refs if len(r) > 5]
    return final_refs

####################

def extract_references(content: str) -> str:
    references = ""
    lines = content.splitlines()
    ref_start = None
    keywords = [r"References"]
    for i, line in enumerate(lines):
        if any(re.search(kw, line, re.IGNORECASE) for kw in keywords):
            ref_start = i
            break

    if ref_start is None:
        print("No References section found in the text.")
        return ""

    possible_stops = [r'\bAppendix\b', 
                      r'\bAPPENDIX\b', 
                      r'\bNotation\b', 
                      r'\bNOTATION\b', 
                      r'\bSupplementary\b', 
                      r'\bSUPPLEMENTARY\b',
                      r'\bAttention Visualizations\b',
                      r'\bA NOTATIONS\b']
                    #   r'^[A-Z]\.\s', 
                    #   r'^[A-Z]\s']
    
    ref_end = len(lines)
    for j in range(ref_start+1, len(lines)):
        for stop_word in possible_stops:
            if re.search(stop_word, lines[j], re.IGNORECASE):
                ref_end = j
                break
        if ref_end != len(lines):
            break
    
    references_lines = lines[ref_start:ref_end]
    references = "\n".join(references_lines)

    return references

def extract_main_content(content: str) -> str:
    main_content = ""
    lines = content.splitlines()
    ref_start = None
    # keywords = [r"Abstract", r"Introduction"]
    keywords = [r"References"]
    for i, line in enumerate(lines):
        if any(re.search(kw, line, re.IGNORECASE) for kw in keywords):
            ref_start = i
            break
    if ref_start is not None:
        main_content = "\n".join(lines[:ref_start])
    else:
        main_content = content
    return main_content

def summarize_content(model, tokenizer, content: str, device="cuda", max_new_tokens=300) -> str:

    system_prompt = (
        "You are a helpful assistant for academic summarization. "
        "Do not restate the entire text; provide a concise summary."
    )
    user_prompt = (
        "Please write a thorough, multi-paragraph summary covering" 
        "the main contributions, methodologies, experimental results," 
        "and conclusions. The summary should be around 1000 words.\n\n"
        f"{content}\n\n"
        "Your summary should be relatively short and must not copy the text verbatim."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True
    ).to(device)

    # if DataParallel: use model.module to generate
    gen_model = model.module if hasattr(model, 'module') else model

    outputs = gen_model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id
    )

    generated = outputs[0][input_ids.shape[1]:]
    summary = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return summary

#####################3
# top 5
def parse_selected_references(response_text):

    # Look for the section after "Selected References:"
    if "Selected References:" in response_text:
        selected_section = response_text.split("Selected References:")[1].strip()
        # Parse the references with their rank preserved
        selected_refs = []
        for line in selected_section.splitlines():
            if line.strip():
                # Look for reference number in different formats
                match = re.search(r'(?:\[(\d+)\]|\((\d+)\)|\b(\d+)\. |\b(\d+)\))', line)
                if match:
                    for group in match.groups():
                        if group and group.isdigit():
                            index = int(group)
                            if index > 0:  # Valid reference index
                                selected_refs.append(index)
                                break
        return selected_refs  # Return selected refs
    return []

def pick_top_5_references(model, tokenizer, summary: str, references: list, device="cuda"):

    refs_joined = "\n\n".join([f"[{idx+1}]: {ref}" for idx, ref in enumerate(references)])
    print("JOINED REFERENCES:", refs_joined)
    
    system_prompt = (
        "You are an academic assistant. You have the paper's summary and the reference list. "
        "Please identify which 5 references appear to have the greatest contribution "
        "to the paper based on the summary, main arguments, or any context you can infer."
    )

#######################3 one reference a time -> five times

    user_prompt = (
        f"Paper Summary:\n{summary}\n\n"
        f"Reference List:\n{refs_joined}\n\n"
        "Please analyze ALL references carefully. Select the FIVE most IMPORTANT different references "
        "based on their contribution to the paper's core ideas, not just the first ones in the list. "
        "Consider which references inform the key methodologies, concepts, and findings in the paper. "
        "Output the selected references strictly in the following format with no additional text:\n\n"
        "Selected References:\n"
        "[X] Reference text\n"
        "[Y] Reference text\n"
        "...\n"
        "Where X, Y, etc. are the original reference numbers."
    )
        # "Please analyze the importance of each reference based on the summary, "
        # "select the five most important different references. "
        # "Output the selected references strictly in the following format with no additional text:\n\n"
        # "Selected References:\n"
        # "[1] Reference 1\n"
        # "[2] Reference 2\n"
        # "...\n"
    # )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True
    ).to(device)

    gen_model = model.module if hasattr(model, 'module') else model

    outputs = gen_model.generate(
        input_ids,
        max_new_tokens=1000,
        do_sample=False,
        # temperature=0.3,
        # top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id
    )

    # generated = outputs[0][input_ids.shape[1]:]
    # top5_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
    # return top5_output

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    selected_indices = parse_selected_references(response)

    seen = set()
    unique_indices = []
    for idx in selected_indices:
        if idx not in seen and idx <= len(references):
            seen.add(idx)
            unique_indices.append(idx)
            if len(unique_indices) >= 5:
                break

    # If we don't have 5 references yet, try a fallback regex approach
    if len(unique_indices) < 5:
        print("Warning: Couldn't find enough references in the structured format, using fallback extraction...")
        # Regex pattern to capture numbers in [num], (num), num., num)
        pattern = r'(?:\[(\d+)\]|\((\d+)\)|\b(\d+)\. |\b(\d+)\))'
        matches = re.findall(pattern, response)
        for match in matches:
            for group in match:
                if group and group.isdigit():
                    index = int(group)
                    if 1 <= index <= len(references) and index not in seen:
                        seen.add(index)
                        unique_indices.append(index)
                        if len(unique_indices) >= 5:
                            break
            if len(unique_indices) >= 5:
                break

    if len(unique_indices) < 5:
        print("Warning: Still not enough references found, selecting first available references...")
        for i in range(1, min(len(references) + 1, 6)):
            if i not in seen:
                unique_indices.append(i)
                if len(unique_indices) >= 5:
                    break
    
    # Map indices to reference texts
    selected_refs = []
    for idx in unique_indices[:5]:  # Limit to first 5
        selected_refs.append((idx, references[idx-1]))

    # # Extract all possible reference indices from the response
    # selected_indices = []
    # # Regex pattern to capture numbers in [num], (num), num., num)
    # pattern = r'(?:\[(\d+)\]|\((\d+)\)|\b(\d+)\. |\b(\d+)\))'
    # matches = re.findall(pattern, response)
    # for match in matches:
    #     for group in match:
    #         if group.isdigit():
    #             index = int(group)
    #             if 1 <= index <= len(references):
    #                 selected_indices.append(index)
    
    # # Deduplicate while preserving order
    # seen = set()
    # selected_indices = [idx for idx in selected_indices if not (idx in seen or seen.add(idx))]
    
    # # Take the first 5 unique indices
    # top5_indices = selected_indices[:5]
    
    # # Map indices to reference texts
    # selected_refs = []
    # for idx in top5_indices:
    #     selected_refs.append((idx, references[idx-1]))
    
    return selected_refs

def save_top5_to_csv(selected_refs: List[Tuple[int, str]], csv_path="references_top5.csv"):
    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Rank", "Original Index", "Reference"])
        for rank, (index, ref) in enumerate(selected_refs, start=1):
            writer.writerow([rank, index, ref])




def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # pdf_path = "2312.02126v3.pdf"
    pdf_path = "Attention_is_all_you_need.pdf"
    # pdf_path = "308.pdf"
    tokenizer, model = load_model(device=device)

    content = read_pdf(pdf_path)

    title = get_paper_title(content)

    main_content = extract_main_content(content)

    references_text = extract_references(content)

    # summary = summarize_content(model, tokenizer, main_content, device=device, max_new_tokens=300)
    chunk_size = 2000

    paragraphs = []
    for i in range(0, len(main_content), chunk_size):
        paragraphs.append(main_content[i:i+chunk_size])

    # summarize_content => partial_summaries
    partial_summaries = []
    for chunk in paragraphs:
        partial_summaries.append(summarize_content(model, tokenizer, chunk))

    # partial_summaries => summarize_content 
    joined_partial = "\n".join(partial_summaries)
    final_summary = summarize_content(model, tokenizer, joined_partial, max_new_tokens=2000)
    print("\nFinal summary:", final_summary)

    references = parse_references(references_text)
    # print("\nParsed References (total {}):".format(len(references)))
    # for ref in references:
    #     print("-", ref)

    # print("\n\n\ncontent:\n", content)

    print("\n\n\ntitle\n", title)

    # print("\n\n\nmain_content\n", main_content)

    # print(f"\n\n\nGot summary\n: {summary[:300]}...\n")

    print("\n\n\nreferences_text\n", references_text)

    print("\n\n\nreferences\n", references)

    # if len(references) > 5:
    #     raw_top5_str = pick_top_5_references(model, tokenizer, final_summary, references, device=device)
    #     print("\nTop 5 references (raw LLM output):\n", raw_top5_str)

    #     lines = raw_top5_str.strip().splitlines()
    #     top5_list = []

    #     top5_list = top5_list[:5]
    #     save_top5_to_csv(top5_list, csv_path="references_top5.csv")
    #     print("\nTop 5 references saved to references_top5.csv.")
    # else:
    #     print("\nNot enough references to pick top 5.")
    selected_refs = pick_top_5_references(model, tokenizer, final_summary, references, device=device)
    if len(selected_refs) < 5:
        print("Failed to select five references.")
    else:
        print("Successfully selected five references.")
        save_top5_to_csv(selected_refs, csv_path="top5_references.csv")
        print(f"Results saved to 'selected_references'.\n")
        print("Top 5 References:")
        for rank, (index, ref) in enumerate(selected_refs, start=1):
            print(f"{rank}. [Index={index}] {ref}")


if __name__ == "__main__":
    main()


