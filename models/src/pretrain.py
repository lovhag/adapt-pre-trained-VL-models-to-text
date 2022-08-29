import argparse
import deepspeed
import torch
import transformers
import logging
import shutil
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from training_utils import get_all_checkpoints
from datasets import get_text_image_pretraining_dataset
from clip_bert.modeling_bert import BertImageForMaskedLM
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer, DataCollatorForLanguageModeling, BertConfig


def get_args():
    parser = argparse.ArgumentParser(description="Pretraines a bert-base-uncased model on either a pure text or text+image dataset")

    group = parser.add_argument_group("Data", "Data configuration")
    group.add_argument("--vlp-dataset", nargs=3, help="train, val and image_feature files respectively")
    group.add_argument("--text-dataset", nargs=2, help="train and val files respectively")

    group = parser.add_argument_group("Training", "Training configuration")
    group.add_argument("--local_rank", type=int, required=True, help="Which local gpu to use")
    group.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory to load and save checkpoints to")
    group.add_argument("--checkpoint-every", type=int, default=1000, help="Checkpoint every X step")
    group.add_argument("--checkpoint-max", type=int, default=5, help="Max checkpoints to keep")
    group.add_argument("--resume-checkpoint", default=None, type=Path, help="Path to checkpoint to resume")
    group.add_argument("--evaluate-every", default=None, type=int, help="How often to evaluate")
    group.add_argument("--no-pretrained-weights", action="store_true", default=False, help="Activate this flag if you do not wish for the model to be initialized from pretrained weights")
    group.add_argument("--bert-model", default="bert-base-uncased", help="What BERT model version from Huggingface to use")

    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


def main(args):
    if args.local_rank is not None:
        device = torch.device(f"cuda:{args.local_rank}")

    assert (args.vlp_dataset is None) != (args.text_dataset is None), "Either --vlp-dataset or --text-dataset must be set"

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    if args.no_pretrained_weights:
        config = BertConfig.from_pretrained(args.bert_model,
                                           output_attentions=False,
                                           output_hidden_states=False)
        model = BertImageForMaskedLM(config)
    else:
        model = BertImageForMaskedLM.from_pretrained(args.bert_model,
                                                    output_attentions=False,
                                                    output_hidden_states=False)

    model.train()

    model_engine, _, _, _ = deepspeed.initialize(args=args,
                                                 model=model,
                                                 model_parameters=model.parameters())

    if args.vlp_dataset is not None:    
        train_path, val_path, image_features_path = args.vlp_dataset
        train_ds, val_ds = get_text_image_pretraining_dataset(train_path, val_path, tokenizer, image_features_path, use_visual_prediction=False)
    else:
        train_path, val_path = args.text_dataset
        train_ds, val_ds = get_text_image_pretraining_dataset(train_path, val_path, tokenizer, image_features_path=None, use_visual_prediction=False)


    if args.evaluate_every is not None:
        assert args.evaluate_every % model_engine.gradient_accumulation_steps() == 0, \
            "evaluate_every needs to be divisible with gradient_accumulation_steps"

    if model_engine.global_rank == 0:
        transformers.logging.set_verbosity_info()

    
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)

    train_dataloader = DataLoader(train_ds, 
                                  batch_size=model_engine.train_micro_batch_size_per_gpu(), 
                                  collate_fn=collator, 
                                  num_workers=8,
                                  sampler=DistributedSampler(train_ds, 
                                                             num_replicas=model_engine.world_size,
                                                             rank=model_engine.global_rank))
    val_dataloader = DataLoader(val_ds, 
                                batch_size=model_engine.train_micro_batch_size_per_gpu(), 
                                collate_fn=collator,
                                num_workers=8,
                                sampler=DistributedSampler(val_ds, 
                                                           num_replicas=model_engine.world_size,
                                                           rank=model_engine.global_rank))

    # Optionally load checkpoint
    resume_step = 0
    if args.checkpoint_dir is not None:
        load_path, _ = model_engine.load_checkpoint(str(args.resume_checkpoint.parent) if args.resume_checkpoint is not None else args.checkpoint_dir, 
                                                    tag=str(args.resume_checkpoint.name) if args.resume_checkpoint is not None else None,
                                                    load_module_strict=False)
        if load_path is not None:
            logging.info(f"Loading existing checkpoint: {load_path}")

            # Resume steps if resuming checkpoint from checkpoint_dir
            if args.resume_checkpoint is not None and args.resume_checkpoint.parent == args.checkpoint_dir:
                resume_step = model_engine.global_steps

    while True:
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            step_output = model_engine(**batch)
            model_engine.backward(step_output.loss)
            model_engine.step()

            if model_engine.global_rank == 0:
                logging.info(f"step={model_engine.global_steps}\tloss={step_output.loss.item()}")

            # Evaluate
            if args.evaluate_every is not None and \
                model_engine.global_steps % args.evaluate_every == 0 and \
                model_engine.is_gradient_accumulation_boundary():
                model_engine.module.eval()
                evaluate(model_engine, device, val_dataloader)
                model_engine.module.train()

            # Checkpoint
            if args.checkpoint_dir is not None and \
                model_engine.global_steps % args.checkpoint_every == 0 and \
                model_engine.global_steps > resume_step:
                checkpoints = get_all_checkpoints(args.checkpoint_dir)
                if len(checkpoints) >= args.checkpoint_max and \
                    model_engine.global_rank == 0 and \
                    model_engine.is_gradient_accumulation_boundary():
                    logging.info(f"Removing old checkpoint: {checkpoints[-1]}")
                    shutil.rmtree(str(args.checkpoint_dir / checkpoints[-1]))
                args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model_engine.save_checkpoint(str(args.checkpoint_dir))


@torch.no_grad()
def evaluate(model_engine, device, test_dataloader):
    losses = []  # List of scalar tensors
    test_dl_iter = tqdm if model_engine.global_rank == 0 else iter
    for batch in test_dl_iter(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        step_output = model_engine(**batch)
        losses.append(step_output.loss)
    
    stacked_losses = torch.stack(losses)  # (num_batches, )
    all_losses = [torch.zeros_like(stacked_losses) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(all_losses, stacked_losses)
    if torch.distributed.get_rank() == 0:
        total_avg_loss = torch.cat(all_losses).mean()  # (num test examples, ) -> scalar
        print("Average test loss: " + str(total_avg_loss.item()))
        if model_engine.tensorboard_enabled():
            model_engine.summary_writer.add_scalar("Test/loss", total_avg_loss.item(), model_engine.global_steps)
            model_engine.summary_writer.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    main(args)
