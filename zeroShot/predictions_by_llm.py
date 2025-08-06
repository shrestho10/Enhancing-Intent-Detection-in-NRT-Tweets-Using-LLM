import pandas as pd
import torch
import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load dataset
dataset = pd.read_csv("./nrt_test_with_all.csv").dropna().reset_index(drop=True)
print("Printing one prompt:")
print(dataset['text'][40])


# Constants
BATCH_SIZE = 2000  # Number of rows per file
OUTPUT_DIR = "batches"  # Directory to store batch files
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

# Token authentication
token=''
from huggingface_hub import login
login(token=token)

# Load model and tokenizer configurations
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
model_name = "meta-llama/Meta-Llama-3-8B"

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    use_auth_token=token
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Define prediction function
def get_prediction(prompt, max_new_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Process dataset in batches
def process_batches(dataset, batch_size=BATCH_SIZE):
    total_batches = (len(dataset) + batch_size - 1) // batch_size  # Calculate total number of batches
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(dataset))
        batch = dataset.iloc[start_idx:end_idx].copy()
        
        tqdm.pandas(desc=f"Processing batch {batch_num + 1}/{total_batches}")
        batch['prediction'] = batch['text'].progress_apply(get_prediction)

        # Save each batch as a separate CSV file
        batch_file = os.path.join(OUTPUT_DIR, f'test_predictions_with_descriptions_{batch_num + 1}.csv')
        batch.to_csv(batch_file, index=False)
        print(f"Saved batch {batch_num + 1} to {batch_file}")

# Execute the batch processing
process_batches(dataset)
