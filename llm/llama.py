# Use llama3-8b
from PyPDF2 import PdfReader            # pdf to text
from transformers import AutoTokenizer, AutoModelForCausalLM
# from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch


# Load model directly
def load_llama_model(model_name="meta-llama/Meta-Llama-3-8B", device="cuda"):        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    return tokenizer, model


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def split_text(text, chunk_size=1024):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# Example: show text
def analyze_text(model, tokenizer, text, device="cuda", max_length=4096):
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example: summarize
def summarize_chunk(model, tokenizer, chunk, device="cuda"):
    prompt = f"Please summarize this paper:"
    inputs = tokenizer.encode(chunk, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model.generate(inputs, max_length=1000, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Jist test
def main():
    pdf_path = "117.pdf"       # test pdf paper
    print("Reading PDF...")
    pdf_text = read_pdf(pdf_path)

    # Loading model
    print("Loading LLama3-8b model...")
    tokenizer, model = load_llama_model(model_name="meta-llama/Meta-Llama-3-8B")

    print("Splitting text into chunks...")
    text_chunks = split_text(pdf_text, chunk_size=1024)

    # Analyzing
    print("Analyzing text...")
    summarized_text = analyze_text(model, tokenizer, pdf_text)
    print("\nSummarized Content:\n", summarized_text)

    # Summarizing
    print("Summarizing text...")
    summarized_chunks = []
    for i, chunk in enumerate(text_chunks):
        # print(f"Summarizing chunk {i + 1}/{len(text_chunks)}...")
        summary = summarize_chunk(model, tokenizer, chunk)
        summarized_chunks.append(summary)

    final_summary = "\n".join(summarized_chunks)
    print("\nFinal Summary:\n", final_summary)

if __name__ == "__main__":
    main()