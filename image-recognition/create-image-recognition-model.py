# from datasets import Dataset
# dataset = Dataset.from_file("./v1/train/data-00000-of-00001.arrow")

from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor

# We'll use torchvision transformations
# To prepare our images for training and validation
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

from datasets import load_metric, load_from_disk

import numpy as np

import torch

metric = load_metric("accuracy")

# pre-trained model from which to fine-tune
model_checkpoint = "microsoft/swin-tiny-patch4-window7-224"
#model_checkpoint = "google/vit-base-patch16-224"
batch_size = 32 # batch size for training and evaluation

#dataset = load_dataset("arrow", data_files={'train': './build/sample-dataset/train/data-00000-of-00001.arrow'})
dataset = load_from_disk("./build/tarkov-items-image-dataset")

# Load all albels from the dataset
labels = dataset["train"].features["label"].names

# Create two dictionaries to map labels to integers and vice versa
label2id, id2label = dict(), dict()

# Go through all labels and assign them an integer
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# For debugging
# img = dataset['train'][4]['image']
# plt.imshow(img)

# Reference for the code below: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb#scrollTo=4O_p3WrpRyej

# Preprocessing images typically comes down to (1) resizing them to a particular size (2) normalizing the color channels (R,G,B) using a mean and standard deviation. 
# These are referred to as image transformations.
# In addition, one typically performs what is called data augmentation during training (like random cropping and flipping) to make the model more robust and achieve higher accuracy. 
# Data augmentation is also a great technique to increase the size of the training data.

# We can create a image processor based on another pre-trained model
# In this case the "microsoft/swin-tiny-patch4-window7-224"
image_processor  = AutoImageProcessor.from_pretrained(model_checkpoint)

# Our normalize function that uses the previous image_processor as base
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

# We have two ways of defining the size, crop_size and max_size, based on the image_processor.size
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

# The transformations that will be used for training
train_transforms = Compose(
        [
            #RandomResizedCrop(crop_size),
            #RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

# The transformations that will be used for validation
val_transforms = Compose(
        [
            #Resize(size),
            #CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

# Function to apply the transformations to the training data
def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    
    example_batch["pixel_values"] = [
        # For each iamge inside of our batch, we convert it to RGB and then apply the training transforms
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]

    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""

    example_batch["pixel_values"] = [
        # For each image inside of our batch, we convert it to RGB and then apply the validation transforms
        val_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    
    return example_batch

# split up training into training + validation (10% of the data)
splits = dataset["train"].train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

# Next, we can preprocess our dataset by applying these functions. 
# We will use the set_transform functionality, 
# which allows to apply the functions above on-the-fly 
# (meaning that they will only be applied when the images are loaded in RAM).

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# Training the model, but still not exactly specialized
model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

# The warning is telling us we are throwing away some weights (the weights and bias of the classifier layer) and randomly initializing some other (the weights and bias of a new classifier layer).
# This is expected in this case, because we are adding a new head for which we don't have pretrained weights, so the library warns us we should fine-tune this model before using it for inference, which is exactly what we are going to do.

checkpoint_name = model_checkpoint.split("/")[-1]
model_name = f"{checkpoint_name}-finetuned-tarkov"

# Arguments for the training

# Most of the training arguments are pretty self-explanatory, but one that is quite important here is remove_unused_columns=False.
# This one will drop any features not used by the model's call function. By default it's True because usually it's ideal to drop unused feature columns, making it easier to unpack inputs into the model's call function.
# But, in our case, we need the unused features ('image' in particular) in order to create 'pixel_values'.
args = TrainingArguments(
    f"./build/{model_name}",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    # push_to_hub=True,
    hub_model_id=f"uriannrima/{model_name}",
)


# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
# Train the model
train_results = trainer.train()

# rest is optional but nice to have
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate()
# some nice to haves:
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# model.save_pretrained(f"./build")