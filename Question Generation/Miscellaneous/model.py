from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from evaluate import load

from pynvml import *
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

torch.cuda.empty_cache()

dataset = load_dataset('eduvedras/Image_Description',trust_remote_code=True)

#dataset = dataset["train"].train_test_split(test_size=0.1)
train_ds = dataset["train"]
#test_ds = dataset["test"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_id = "microsoft/git-large-r-textcaps"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

def transforms(example_batch):
    images = [x for x in example_batch["Chart"]]
    captions = [x for x in example_batch["Description"]]
    inputs = processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs

train_ds.set_transform(transforms)
#test_ds.set_transform(transforms)


rouge = load("rouge")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    rouge_score = rouge.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"rouge_score": rouge_score}

from transformers import TrainingArguments, Trainer

model_name = model_id.split("/")[1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-description-large-textcaps",
    learning_rate=5e-5,
    num_train_epochs=100,
    fp16=True,
    per_device_train_batch_size=4,
    #per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=False,
    optim="adafactor",
    #evaluation_strategy="steps",
    #eval_steps=50,
    save_strategy="steps",
    save_total_limit=3,
    save_steps=0.25,
    logging_steps=0.1,
    remove_unused_columns=False,
    push_to_hub=True,
    label_names=["labels"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    #eval_dataset=test_ds,
    #compute_metrics=compute_metrics,
)

result = trainer.train()

print_summary(result)