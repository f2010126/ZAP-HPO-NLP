import argparse
from tqdm import tqdm
import evaluate
import wandb
import os
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, \
    AutoConfig

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


def get_dataloaders(accelerator: Accelerator, batch_size: int = 8, model_name = 'bert-base-german-cased',
                    dataset="amazon_reviews_multi", split="de"):
    """
    Creates a set of `DataLoader`s for the `amazon` dataset,
    using "bert-base-cased" as the tokenizer.
    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
        model_name (`str`, *optional*):
            The name of the model to use for tokenization.
        dataset_name (`str`, *optional*):
            The name of the dataset to use.
        split (`str`, *optional*):
            The name of the split to use. Usually is the language,
            but can be a subset of the dataset. It's not the same as test/train/val.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datasets = load_dataset(dataset, split, )

    # Combine review title and body. Update stars to go from 1-5 to 0-4
    def prep_data(example):
        example['review_body'] = ["{} {}".format(title, review) for title, review in
                                  zip(example['review_title'], example['review_body'])]
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
                            'review_body', 'product_category'],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("stars", "labels")
    accelerator.print('dataset loaded')

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

    test_dataloader = DataLoader(
        tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
    )

    return train_dataloader, eval_dataloader, test_dataloader

def set_up_checkpoint():
    try:
        os.makedirs("models")
    except FileExistsError:
        # directory already exists
        pass

def training_function(config, args):
    # Initialize accelerator
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision, log_with="wandb")
    if accelerator.is_main_process:
        set_up_checkpoint()
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    experiment_name = config["exp_name"]
    group = config["wandb_group"]
    job_type = config["wandb_job"]
    # Helper function for reproducible behavior to set the seed in random, numpy, torch and/or tf
    set_seed(seed)
    checkpoint_path = f'./models/{model_name}_checkpoint.pt'

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    config['grad_acc'] = gradient_accumulation_steps
    # Init the logging
    accelerator.init_trackers(
            experiment_name,
            config=args,
            init_kwargs={
                "wandb": {
                    "notes": "testing accelerate pipeline",
                    "group": group,
                    "job_type": job_type,
                    "entity": "insane_gupta",  # wandb username
                }
            },
        )

    accelerator.print('Setting up')
    train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(accelerator, batch_size, model_name=model_name, dataset=dataset_name, split=args.split_lang)
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model_config = AutoConfig.from_pretrained(
        model_name, num_labels=5)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)
    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )
    # which step and epoch to start from
    ep_step = 0
    epochs_run = 0

    # Restore model, ep_step, epoch, optimiser, scheduler  from Checkpoint
    if os.path.exists(checkpoint_path):
        accelerator.print("Loading snapshot")
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        epochs_run = ckpt['epoch']
        ep_step=ckpt['ep_step']
    # Prepare everything
    accelerator.print('Preparing with Accelerate')
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader,test_dataloader, lr_scheduler
    )
    accelerator.register_for_checkpointing(lr_scheduler)
    accelerator.register_for_checkpointing(optimizer)

    # Instantiate training metrics
    metric = evaluate.load("f1")
    train_acc = evaluate.load("accuracy")
    train_f1 = evaluate.load("f1")
    # train the model
    accelerator.print('Training')
    for epoch in range(epochs_run, num_epochs):
        # Training
        accelerator.print(f'Epoch {epoch} started')
        model.train()
        with tqdm(total=100, bar_format="{l_bar}{bar} [ time left: {remaining}, time spent: {elapsed}]") as pbar:

            for step, batch in tqdm(enumerate(train_dataloader)):
                # restarting from checkpoint
                if step < ep_step:
                    continue

                # We could avoid this line since we set the accelerator with `device_placement=True`.
                batch.to(accelerator.device)
                outputs = model(**batch)
                # Calculate the loss, accuracy and f1 score
                predictions = outputs.logits.argmax(dim=-1)
                # gather accross all processes
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                train_acc.add_batch(predictions=predictions, references=references)
                train_f1.add_batch(predictions=predictions, references=references)

                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                # Log the loss and metrics
                accelerator.log({"train_loss": loss}, step=step)
                accelerator.log({"train_acc": train_acc.compute()['accuracy'],
                                     "train_f1": train_f1.compute(average="weighted")['f1']}, step=step)
                accelerator.backward(loss)


                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if step % 50 == 0 and (accelerator.is_main_process):
                    # checkpointing
                    save_model = accelerator.unwrap_model(model)
                    torch.save({
                                'ep_step': step,
                                'epoch': epoch,
                                'model_state_dict': save_model.state_dict(),
                                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                            }, checkpoint_path)

                pbar.update(1)

        # Validation
        model.eval()
        accelerator.print('Validation after epoch')
        for step, batch in tqdm(enumerate(eval_dataloader)):
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = round(metric.compute(average="weighted")['f1'],5)
        accelerator.log({"eval_metric": eval_metric}, step=epoch)
        accelerator.print(f"epoch eval {epoch}:", eval_metric)

    accelerator.print('Testing')
    metric = evaluate.load("f1")
    model.eval()
    for step, batch in tqdm(enumerate(test_dataloader)):
        batch.to(accelerator.device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    eval_metric = round(metric.compute(average="weighted")['f1'], 5)
    accelerator.log({"test_metric": eval_metric}, step=epoch)
    # Use accelerator.print to print only on the main process.
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    # Distrubuted training parameters
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
    # Training Hyperparameters
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train. Default is 3.", default=3)
    parser.add_argument("--batch_size", type=int, help="Batch size to train. Default is 32.", default=32)
    parser.add_argument("--lr", type=float, help="Learning rate to train. Default is 2e-5.", default=2e-5)
    parser.add_argument("--seed", type=int, help="Seed to use. Default is 42.", default=42)
    # Model and dataset parameters
    parser.add_argument("--model_name", type=str, help="Name of the model to train. Default is bert-base-german-cased", default="bert-base-german-cased")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to train. Default is amazon_reviews_multi.",
                        default="amazon_reviews_multi")
    parser.add_argument("--split-lang",type=str,default='de',help='Language split of data to use.Deafult is German ')
    # Logging parameters
    parser.add_argument("--exp_name", type=str, help="Name of WANDDB experiment.", default="test-pipeline")
    parser.add_argument("--group_name", type=str, help="Name of WANDDB group.", default="BERT")
    parser.add_argument("--job_name", type=str, help="Name of WANDDB job.", default="8GPU")
    args = parser.parse_args()
    config = {"lr": args.lr, "num_epochs": args.num_epochs, "seed": args.seed, "batch_size": args.batch_size,
              "model_name": args.model_name,
              "dataset_name": args.dataset_name,
              "exp_name": args.exp_name,
              'wandb_group': args.group_name,
              'wandb_job': args.job_name, }
    training_function(config, args)


if __name__ == "__main__":
    main()
