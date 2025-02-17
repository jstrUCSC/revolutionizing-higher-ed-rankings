import os
import re
import csv
import torch
import torch.distributed as dist
import argparse
from typing import List, Tuple
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP

# def setup_distributed():
#     dist.init_process_group("nccl")
#     torch.cuda.set_device(dist.get_rank())

# def load_llama_model(model_name="meta-llama/Meta-Llama-3-70B", device="cuda"):
# def load_llama_model(model_name="meta-llama/Meta-Llama-3-8B", device="cuda"):
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
def load_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", device="cuda"):
    # setup_distributed()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # **这里改成 DataParallel**
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)  

    model.to(device)  

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


def extract_references(content: str) -> str:
    references = ""
    lines = content.splitlines()
    ref_start = None

    keywords = [r"References"]
    for i, line in enumerate(lines):
        if any(re.search(kw, line, re.IGNORECASE) for kw in keywords):
            ref_start = i
            break

    if ref_start is not None:
        references = "\n".join(lines[ref_start:])
    else:
        print("No References section found in the text.")
    
    return references


def extract_main_content(content: str) -> str:
    main_content = ""
    lines = content.splitlines()
    ref_start = None

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


def parse_references(references_text: str) -> List[str]:
    # if references labeled like [1], then
    pattern = r'\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)'
    
    # but how about no label?
    matches = re.findall(pattern, references_text, re.DOTALL)
    refs = [re.sub(r'\s+', ' ', ref[1]).strip() for ref in matches]
    return refs


def summarize_content(
    model, 
    tokenizer, 
    content: str, 
    max_new_tokens=500, 
    device="cuda"
) -> str:
    # prompt = (
    #     "Please provide a concise summary of the following paper text for subsequent analysis:\n\n"
    #     f"{content}\n\nSummary:"
    # )
    messages = [{
        "role": "user",
        "content": f"Please provide a concise summary of the following paper text for subsequent analysis:\n\n{content}\n\nSummary:"
    }]

    # inputs = tokenizer(
        # prompt, 
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048
        #panding=true
    ).to(device)
    
    outputs = model.module.generate(
        # inputs["input_ids"],
        # attention_mask=inputs["attention_mask"],
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id
    )

    summary = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # summary = re.sub(r'^(Summary[:：]?)', '', summary, flags=re.IGNORECASE).strip()
    # return summary
    return summary.strip()


def select_and_rank_references(
    model, 
    tokenizer, 
    main_content: str, 
    references: List[str], 
    device="cuda"
) -> List[Tuple[int, str]]:
    max_main_content_length = 2000
    if len(main_content) > max_main_content_length:
        main_content = summarize_content(model, tokenizer, main_content, device=device)
        print(f"Content is too long, using summarized main content:\n{main_content[:500]}...\n")

    references_text = "\n".join([f"[{i+1}] {ref}" for i, ref in enumerate(references)])
    
    # prompt = (
    #     "The following is the main content of a research paper and its list of references. "
    #     "Please analyze the importance of each reference based on the main content, "
    #     "select the five most important references, and rank them in order of importance. "
    #     "Output the selected references strictly in the following format with no additional text:\n\n"
    #     "Main content:\n"
    #     f"{main_content}\n\n"
    #     "References list:\n"
    #     f"{references_text}\n\n"
    #     "Rank:\n"
    # )
    messages = [{
        "role": "user",
        "content": (
            "The following is the main content of a research paper and its list of references. "
            "Please analyze the importance of each reference based on the main content, "
            "select the five most important references, and rank them in order of importance. "
            "Output the selected references strictly in the following format with no additional text:\n\n"
            f"Main content:\n{main_content}\n\n"
            f"References list:\n{references_text}"
        )
    }]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(device)
    
    outputs = model.module.generate(
        input_ids,
        max_new_tokens=500,
        temperature=0.3,     
        top_p=0.9,           
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    print("LLM Raw Response:\n", response, "\n")

    selected_refs = []
    for line in response.splitlines():
        line = line.strip()
        match = re.match(r"^\s*[\[\(]?(\d+)[\]\)]?[\.\s]*(.*)", line)
        if match and len(selected_refs) < 5:
            index = int(match.group(1))
            if 1 <= index <= len(references):
                selected_refs.append((index, references[index-1]))
    
    return selected_refs[:5]


def save_to_csv(selected_refs: List[Tuple[int, str]], output_file="selected_references.csv"):
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Rank", "Reference"])
        for rank, (index, ref) in enumerate(selected_refs, start=1):
            writer.writerow([rank, ref])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pdf_path = "2212.06872v5.pdf"
    
    print(f"Reading PDF: {pdf_path}")
    content = read_pdf(pdf_path)
    if not content.strip():
        print("PDF content is empty.")
        return

    title = get_paper_title(content)
    print(f"Paper Title: {title}")

    main_content = extract_main_content(content)
    references_text = extract_references(content)
    
    if not references_text.strip():
        print("No valid references found.")
        return

    references = parse_references(references_text)
    print(f"Total {len(references)} references found.")
    
    if len(references) < 5:
        print("Insufficient references for ranking.")
        return

    print("Loading DeepSeek-R1-Distill-Qwen-14B model...")
    tokenizer, model = load_model(device=device)
    
    print("Ranking references...")
    selected_refs = select_and_rank_references(model, tokenizer, main_content, references, device)
    
    if len(selected_refs) == 5:
        save_to_csv(selected_refs)
        print("\nTop 5 References:")
        for rank, (idx, ref) in enumerate(selected_refs, 1):
            print(f"{rank}. [Ref{idx}] {ref[:150]}...")
    else:
        print("Failed to select 5 references")

if __name__ == "__main__":
    # """
    # example:
    # python get_references.py --/nfs/stak/users/wangl9/hpc-share/revolutionizing-higher-ed-rankings/llm 2212.06872v5.pdf --/nfs/stak/users/wangl9/hpc-share/revolutionizing-higher-ed-rankings/llm selected_references.csv
    # """
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--pdf_path", type=str, required=False, default="2212.06872v5.pdf", 
    #                     help="Path to the PDF file.")
    # parser.add_argument("--output_file", type=str, required=False, default="selected_references.csv", 
    #                     help="Name of the output CSV file.")
    # args = parser.parse_args()
    
    # main(args.pdf_path, args.output_file)
    main()


