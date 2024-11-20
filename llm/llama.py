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


# Example
def analyze_text(model, tokenizer, text, device="cuda", max_length=4096):
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



# Jist test
def main():
    pdf_path = "117.pdf"       # test pdf paper
    print("Reading PDF...")
    pdf_text = read_pdf(pdf_path)

    # Loading model
    print("Loading LLama3-8b model...")
    tokenizer, model = load_llama_model(model_name="meta-llama/Meta-Llama-3-8B")

    # Analyzing
    print("Analyzing text...")
    summarized_text = analyze_text(model, tokenizer, pdf_text)
    print("\nSummarized Content:\n", summarized_text)

if __name__ == "__main__":
    main()