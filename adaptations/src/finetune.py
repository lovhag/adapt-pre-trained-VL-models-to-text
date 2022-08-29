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
from transformers import BertTokenizer, DataCollatorForLanguageModeling, BertConfig, VisualBertForPreTraining, LxmertForPreTraining
from lxmert.alterations import LxmertLanguageOnlyXLayer
import copy

LXMERT_FEATURES_SHAPE = (1, 2048)
LXMERT_NORMALIZED_BOXES_SHAPE = (1, 4)

def get_args():
    parser = argparse.ArgumentParser(description="Pretraines a bert-base-uncased model on either a pure text or text+image dataset")

    group = parser.add_argument_group("Data", "Data configuration")
    group.add_argument("--text-dataset", nargs=2, help="train and val files respectively")

    group = parser.add_argument_group("Training", "Training configuration")
    group.add_argument("--local_rank", type=int, required=True, help="Which local gpu to use")
    group.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory to load and save checkpoints to")
    group.add_argument("--checkpoint-every", type=int, default=1000, help="Checkpoint every X step")
    group.add_argument("--checkpoint-max", type=int, default=5, help="Max checkpoints to keep")
    group.add_argument("--resume-checkpoint", default=None, type=Path, help="Path to checkpoint to resume")
    group.add_argument("--evaluate-every", default=None, type=int, help="How often to evaluate")
    group.add_argument("--model", default="clip-bert", help="What model to use")
    group.add_argument("--bert-checkpoint", help="If 'model' is 'clip-bert', what pretrained CLIP-BERT model weights to use")
    group.add_argument("--use-visual-prediction", action="store_true", default=False, help="Whether to give image features from CLIP based on text features, if False no image features are provided")

    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()

def get_lxmert_batch(batch):
    nbr_samples = len(batch)
    # the visual features do not matter for our lxmert version
    normalized_boxes = torch.empty((nbr_samples,)+LXMERT_NORMALIZED_BOXES_SHAPE).uniform_(0, 1)
    features = torch.empty((nbr_samples,)+LXMERT_FEATURES_SHAPE).uniform_(0, 10)
    batch.update(
        {
            "visual_feats": features,
            "visual_pos": normalized_boxes,
        }
    )
    return batch

def get_visualbert_batch(batch):
    nbr_samples = len(batch)
    # handle visualbert trying to get shape of visual_attention_mask
    batch.update(
        {
            "visual_attention_mask": torch.empty((nbr_samples,0)),
        }
    )
    return batch
    

def main(args):
    if args.local_rank is not None:
        device = torch.device(f"cuda:{args.local_rank}")

    assert args.text_dataset is not None, "--text-dataset must be set"
    if args.use_visual_prediction:
        assert args.model=="clip-bert", "visual prediction features from CLIP should only be used together with clip-bert"

    if args.model == "clip-bert":
        assert args.bert_checkpoint is not None, "A pretrained clip-bert checkpoint must be given for the finetuning process"
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")     
        config = BertConfig.from_pretrained("bert-base-uncased")
        model = BertImageForMaskedLM(config)
        model.load_state_dict(torch.load(args.bert_checkpoint, map_location="cpu")["module"], strict=False)

        batch_transformer = lambda batch: batch # if visual regression to be used, will automatically be included in batch
    elif args.model == "lxmert":
        model = LxmertForPreTraining.from_pretrained("unc-nlp/lxmert-base-uncased")
        # replace X layers between vision-language, such that language output doesn't depend on vision
        prev_encoder = copy.deepcopy(model.lxmert.encoder)
        model.lxmert.encoder.x_layers = torch.nn.ModuleList([LxmertLanguageOnlyXLayer(model.lxmert.encoder.config) for _ in range(model.lxmert.encoder.config.x_layers)])
        model.lxmert.encoder.load_state_dict(prev_encoder.state_dict())

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        batch_transformer = get_lxmert_batch
    elif args.model == "visualbert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        batch_transformer = get_visualbert_batch
    else:
        raise ValueError(f"args.model must be either clip-bert, lxmert or visualbert. Got {args.model}")

    model.train()

    model_engine, _, _, _ = deepspeed.initialize(args=args,
                                                 model=model,
                                                 model_parameters=model.parameters())

    train_path, val_path = args.text_dataset
    train_ds, val_ds = get_text_image_pretraining_dataset(train_path, 
                                                          val_path, 
                                                          tokenizer, 
                                                          image_features_path=None, 
                                                          use_visual_prediction=args.use_visual_prediction)


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

    best_test_loss = -1
    while True:
        for batch in train_dataloader:
            # adapt batch to model
            batch = batch_transformer(batch)
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
                new_test_loss = evaluate(model_engine, device, val_dataloader, batch_transformer)

                # save checkpoint with best test loss
                if new_test_loss < best_test_loss or best_test_loss < 0:
                    logging.info(f"New best checkpoint found!")
                    if best_test_loss > 0 and \
                        model_engine.global_rank == 0:
                        logging.info(f"Removing old best checkpoint: {best_checkpoint_dirname}")
                        shutil.rmtree(args.checkpoint_dir / best_checkpoint_dirname)
                    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    best_checkpoint_dirname = "best_global_step" + str(model_engine.global_steps)
                    model_engine.save_checkpoint(str(args.checkpoint_dir), tag=best_checkpoint_dirname)
                    best_test_loss = new_test_loss

                model_engine.module.train()

            # Checkpoint
            if args.checkpoint_dir is not None and \
                model_engine.global_steps % args.checkpoint_every == 0:
                checkpoints = get_all_checkpoints(args.checkpoint_dir)
                if len(checkpoints) >= args.checkpoint_max and \
                    model_engine.global_rank == 0 and \
                    model_engine.is_gradient_accumulation_boundary():
                    logging.info(f"Removing old checkpoint: {checkpoints[-1]}")
                    shutil.rmtree(str(args.checkpoint_dir / checkpoints[-1]))
                args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model_engine.save_checkpoint(str(args.checkpoint_dir))


@torch.no_grad()
def evaluate(model_engine, device, test_dataloader, batch_transformer):
    losses = []  # List of scalar tensors
    test_dl_iter = tqdm if model_engine.global_rank == 0 else iter
    for batch in test_dl_iter(test_dataloader):
        # adapt batch to model
        batch = batch_transformer(batch)
        batch = {k: v.to(device) for k, v in batch.items()}
        step_output = model_engine(**batch)
        losses.append(step_output.loss)
    
    stacked_losses = torch.stack(losses)  # (num_batches, )
    all_losses = [torch.zeros_like(stacked_losses) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(all_losses, stacked_losses)
    total_avg_loss = torch.cat(all_losses).mean()  # (num test examples, ) -> scalar

    if torch.distributed.get_rank() == 0:
        print("Average test loss: " + str(total_avg_loss.item()))
        if model_engine.tensorboard_enabled():
            model_engine.summary_writer.add_scalar("Test/loss", total_avg_loss.item(), model_engine.global_steps)
            model_engine.summary_writer.flush()

    return total_avg_loss.item()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    main(args)
