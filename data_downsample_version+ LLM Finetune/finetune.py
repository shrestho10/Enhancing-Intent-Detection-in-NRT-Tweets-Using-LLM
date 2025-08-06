import pandas as pd
from datasets import DatasetDict, Dataset
import torch
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
token=''
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import balanced_accuracy_score, classification_report

from trl import SFTTrainer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import TrainingArguments
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import EarlyStoppingCallback

import random
seed = 42  # Choose any fixed number
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# If you are using CUDA (GPU), set the seed for GPU as well
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # In case of multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable optimization to ensure reproducibility

# Login with API token
from huggingface_hub import login
login(token=token)


train_df=pd.read_csv("./nrt_training_with_some_nores_v2.csv")
train_df=train_df[['Tweet','intent']]
train_df= train_df.rename(columns={'Tweet':'text','intent':'label'})

valid_df=pd.read_csv("./nrt_validation_with_some_nores_v2.csv")
valid_df=valid_df[['Tweet','intent']]
valid_df= valid_df.rename(columns={'Tweet':'text','intent':'label'})

test_df=pd.read_csv("./nrt_test_with_some_nores_v2.csv")
test_df=test_df[['Tweet','intent']]
test_df= test_df.rename(columns={'Tweet':'text','intent':'label'})

all_labels = pd.concat([train_df['label'], valid_df['label'], test_df['label']]).unique()

# Create a mapping dictionary
label_mapping = {label: idx for idx, label in enumerate(all_labels)}

# Apply the mapping to each DataFrame
train_df['label'] = train_df['label'].map(label_mapping)
valid_df['label'] = valid_df['label'].map(label_mapping)
test_df['label'] = test_df['label'].map(label_mapping)

# Converting pandas DataFrames into Hugging Face Dataset objects:
dataset_train = Dataset.from_pandas(train_df)
dataset_val = Dataset.from_pandas(valid_df)
dataset_test = Dataset.from_pandas(test_df)

# Combine them into a single DatasetDict
dataset = DatasetDict({
    'train': dataset_train,
    'val': dataset_val,
    'test': dataset_test
})

class_weights=(1/train_df.label.value_counts(normalize=True).sort_index()).tolist()
class_weights=torch.tensor(class_weights)
class_weights=class_weights/class_weights.sum()

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, 
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_use_double_quant = True, 
    bnb_4bit_compute_dtype = torch.bfloat16 
)

model_name = "meta-llama/Meta-Llama-3-8B"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=25,
    device_map='auto'
)


lora_config = LoraConfig(
    r = 64, 
    lora_alpha = 16,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, 
    bias = 'none',
    task_type = 'SEQ_CLS'
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

def data_preprocesing(row):
    return tokenizer(row['text'], truncation=True, max_length=512)

tokenized_data = dataset.map(data_preprocesing, batched=True, 
remove_columns=['text'])
tokenized_data.set_format("torch")

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)
    return {'balanced_accuracy' : balanced_accuracy_score(predictions, labels),
    'accuracy':accuracy_score(predictions,labels)}

class CustomTrainer(SFTTrainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            print("Using class weights")
            self.class_weights = torch.tensor(class_weights, 
                                              dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        if labels is None:
            raise ValueError("Missing 'labels' in input dictionary")
        
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels.long(), weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels.long())

        return (loss, outputs) if return_outputs else loss



training_args = TrainingArguments(
    output_dir = 'Llama-3-some_noresponse-v2',
    learning_rate = 1e-4,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = 5,
    logging_steps=500,
    logging_dir="./logs",
    weight_decay = 0.01,
    eval_strategy = 'steps',
    save_strategy = 'steps',
    eval_steps=1000,
    save_steps=1000,
    load_best_model_at_end = True,
    greater_is_better=True,
    metric_for_best_model='balanced_accuracy'  
)

trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_data['train'],
    eval_dataset = tokenized_data['val'],
    tokenizer = tokenizer,
    data_collator = collate_fn,
    compute_metrics = compute_metrics,
    class_weights=class_weights,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=7)]
)

train_result = trainer.train()


def generate_predictions(model,df_test):
    sentences = df_test.text.tolist()
    batch_size = 32  
    all_outputs = []

    for i in range(0, len(sentences), batch_size):

        batch_sentences = sentences[i:i + batch_size]

        inputs = tokenizer(batch_sentences, return_tensors="pt", 
        padding=True, truncation=True, max_length=512)

        inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') 
        for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            all_outputs.append(outputs['logits'])
        
    final_outputs = torch.cat(all_outputs, dim=0)
    df_test['predictions']=final_outputs.argmax(axis=1).cpu().numpy()

generate_predictions(model,test_df)
def get_metrics_result(test_df):
    y_test = test_df.label
    y_pred = test_df.predictions

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

get_metrics_result(test_df)


# Create reverse label mapping
reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}

# Convert numeric predictions to actual label names
test_df['predictions_names'] = test_df['predictions'].map(reverse_label_mapping)
test_df['label_names'] = test_df['label'].map(reverse_label_mapping)

def get_metrics_result_with_name(test_df):
    y_test = test_df.label_names
    y_pred = test_df.predictions_names

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

get_metrics_result_with_name(test_df)


# Save the LoRA-adapted model and tokenizer
model.save_pretrained("./my_saved_model_some_nores-v2")
tokenizer.save_pretrained("./my_saved_model_some_nores-v2")
