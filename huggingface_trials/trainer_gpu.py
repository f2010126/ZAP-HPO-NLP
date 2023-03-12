import argparse
from tqdm import tqdm
import evaluate
import wandb
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed,AutoConfig

from accelerate import Accelerator, DistributedType

########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################

MAX_GPU_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16


def get_dataloaders(accelerator: Accelerator, batch_size: int = 8):
    """
    Creates a set of `DataLoader`s for the `amazon` dataset,
    using "bert-base-cased" as the tokenizer.
    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("amazon_reviews_multi", 'en',)

    # Combine review title and body. Update stars to go from 1-5 to 0-4
    def prep_data(example):
        example['review_body'] = ["{} {}".format(title, review) for title, review in zip(example['review_title'], example['review_body'])]
        example['stars'] = [int(star) - 1 for star in example['stars']]
        return example
    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["review_body"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        datasets = datasets.map(prep_data, batched=True, num_proc=4, remove_columns=['review_title'])
        # We use the `map` method to apply the function to all the examples in the dataset
        tokenized_datasets = datasets.map(
            tokenize_function,
            num_proc=4,
            batched=True,
            remove_columns=["review_id", 'language', 'reviewer_id', 'product_id',
                            'review_body','product_category'],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("stars", "labels")
    print('dataset loaded')

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        max_length = 128 if accelerator.distributed_type == DistributedType.TPU else None
        # When using mixed precision we want round multiples of 8/16
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
    )

    return train_dataloader, eval_dataloader


def training_function(config, args):
    # Initialize accelerator
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision,log_with="wandb")
    accelerator.init_trackers(
        "TestNLPWandb",
        config=config,
        init_kwargs={
            "wandb": {
                "notes": "testing accelerate pipeline",
                "tags": ["tag_a", "tag_b"],
                "entity": "gladiator",
            }
        },
    )

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    experiment_name = config["exp_name"]

    # use the same metric as GLUE
    metric = evaluate.load("glue", "mrpc")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    print('Setting up')
    # Helper function for reproducible behavior to set the seed in random, numpy, torch and/or tf
    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model_config = AutoConfig.from_pretrained(
        model_name, num_labels=5)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)
    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    print('Preparing')
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )


    # Now we train the model
    print('Training')
    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=100, bar_format="{l_bar}{bar} [ time left: {remaining}, time spent: {elapsed}]") as pbar:
            for step, batch in tqdm(enumerate(train_dataloader)):
                # We could avoid this line since we set the accelerator with `device_placement=True`.
                batch.to(accelerator.device)
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                accelerator.log({"train_loss": loss}, step=step)
                accelerator.backward(loss)
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()


        model.eval()
        for step, batch in tqdm(enumerate(eval_dataloader)):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        accelerator.log({"eval_metric": eval_metric}, step=epoch)
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)

    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
             "and an Nvidia Ampere GPU.",
    )
    # store_true sets default value of False is used to test the script on CPU
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument("--model_name", type=str, help="Name of the model to train.", default="bert-base-cased")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to train.", default="amazon_reviews_multi")
    parser.add_argument("--exp_name", type=str, help="Name of WANDDB experiment.", default="test-pipeline")
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 1, "seed": 42, "batch_size": 8,
              "model_name": args.model_name, "dataset_name": args.dataset_name, "exp_name": args.exp_name}
    training_function(config, args)


if __name__ == "__main__":
    main()
