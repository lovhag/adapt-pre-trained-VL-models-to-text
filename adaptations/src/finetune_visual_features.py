import argparse
import torch
import transformers
import logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from training_utils import get_all_checkpoints
from datasets import get_text_image_pretraining_dataset
from clip_bert.modeling_bert import BertImageForMaskedLM
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer, DataCollatorForLanguageModeling, BertConfig, VisualBertForPreTraining, LxmertForPreTraining
from lxmert.alterations import LxmertLanguageOnlyXLayer
import copy
import os
import torch.nn.functional as F

LXMERT_FEATURES_SHAPE = (36, 2048)
LXMERT_NORMALIZED_BOXES_SHAPE = (36, 4)
VISUALBERT_FEATURES_SHAPE = (36, 2048)
CLIP_BERT_FEATURES_SHAPE = (512,)

def get_args():
    parser = argparse.ArgumentParser(description="Pretraines a bert-base-uncased model on either a pure text or text+image dataset")

    group = parser.add_argument_group("Data", "Data configuration")
    group.add_argument("--text-dataset", nargs=2, help="train and val files respectively")

    group = parser.add_argument_group("Training", "Training configuration")
    group.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory to load and save checkpoints to")
    group.add_argument("--checkpoint-every", type=int, default=1000, help="Checkpoint every X step")
    group.add_argument("--checkpoint-max", type=int, default=5, help="Max checkpoints to keep")
    group.add_argument("--resume-checkpoint", default=None, type=Path, help="Path to checkpoint to resume")
    group.add_argument("--evaluate-every", default=None, type=int, help="How often to evaluate")
    group.add_argument("--model", default="clip-bert", help="What model to use")
    group.add_argument("--bert-checkpoint", help="If 'model' is 'clip-bert', what pretrained CLIP-BERT model weights to use")
    group.add_argument("--batch-size", type=int, default=64, help="What batch size to use for train and val")
    group.add_argument("--tensorboard-logdir", type=str, default=None, help="Tensorboard logdir")
    group.add_argument("--lr", type=float, default=0.01, help="Learning rate")


    return parser.parse_args()

def get_lxmert_batch(batch, visual_features, visual_boxes):
    batch_size = batch["input_ids"].shape[0]
    visual_features = visual_features.unsqueeze(0).repeat(batch_size, 1, 1)
    visual_boxes = visual_boxes.unsqueeze(0).repeat(batch_size, 1, 1)
    batch.update(
        {
            "visual_feats": visual_features,
            "visual_pos": visual_boxes,
        }
    )
    return batch

def get_visualbert_batch(batch, visual_features, visual_boxes):
    batch_size = batch["input_ids"].shape[0]
    visual_embeds = visual_features.unsqueeze(0).repeat(batch_size, 1, 1)
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

    # labels must have same length as input sequence plus number of visual detections
    LABEL_PAD_IX = -100
    to_pad = VISUALBERT_FEATURES_SHAPE[0]
    labels = F.pad(batch["labels"], pad=(0, to_pad), mode='constant', value = LABEL_PAD_IX)

    batch.update(
        {
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
            "labels": labels
        }
    )

    return batch
    
def get_clipbert_batch(batch, visual_features, visual_boxes):
    batch_size = batch["input_ids"].shape[0]
    img_feats = visual_features.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1)
    batch.update(
        {
            "img_feats": img_feats
        }
    )
    return batch

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=args.tensorboard_logdir)

    assert args.text_dataset is not None, "--text-dataset must be set"

    if args.model == "clip-bert":
        assert args.bert_checkpoint is not None, "A pretrained clip-bert checkpoint must be given for the finetuning process"
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")     
        config = BertConfig.from_pretrained("bert-base-uncased")
        model = BertImageForMaskedLM(config)
        model.load_state_dict(torch.load(args.bert_checkpoint, map_location="cpu")["module"], strict=False)

        visual_features = torch.rand(CLIP_BERT_FEATURES_SHAPE, requires_grad=True)
        visual_boxes = None

        batch_transformer = get_clipbert_batch
    elif args.model == "lxmert":
        model = LxmertForPreTraining.from_pretrained("unc-nlp/lxmert-base-uncased")

        visual_features = torch.rand(LXMERT_FEATURES_SHAPE, requires_grad=True)
        visual_boxes = torch.rand(LXMERT_NORMALIZED_BOXES_SHAPE, requires_grad=True)

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        batch_transformer = get_lxmert_batch
    elif args.model == "visualbert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

        visual_features = torch.rand(VISUALBERT_FEATURES_SHAPE, requires_grad=True)
        visual_boxes = None

        batch_transformer = get_visualbert_batch
    else:
        raise ValueError(f"args.model must be either clip-bert, lxmert or visualbert. Got {args.model}")

    model.to(device)
    model.train()
    for param in model.parameters():
        param.requires_grad = False

    # the visual features are what we will fine-tune, not the model
    if args.model == "lxmert":
        optimizer = torch.optim.Adam([visual_features, visual_boxes], lr=args.lr)
    else:
        optimizer = torch.optim.Adam([visual_features], lr=args.lr)

    train_path, val_path = args.text_dataset
    train_ds, val_ds = get_text_image_pretraining_dataset(train_path, 
                                                          val_path, 
                                                          tokenizer, 
                                                          image_features_path=None, 
                                                          use_visual_prediction=False)
    
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)

    train_dataloader = DataLoader(train_ds, 
                                  batch_size=args.batch_size, 
                                  collate_fn=collator, 
                                  num_workers=4)
    val_dataloader = DataLoader(val_ds, 
                                batch_size=args.batch_size, 
                                collate_fn=collator,
                                num_workers=4)

    best_test_loss = -1
    step = -1
    while True:
        for batch in train_dataloader:
            step += 1
            optimizer.zero_grad()
            # adapt batch to model
            batch = batch_transformer(batch, visual_features, visual_boxes)
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()

            logging.info(f"step={step}\tloss={loss.item()}")
            writer.add_scalar('Train/loss', loss.item(), step)

            # Evaluate
            if args.evaluate_every is not None and \
                step % args.evaluate_every == 0:
                model.eval()
                new_test_loss = evaluate(model, visual_features, visual_boxes, device, val_dataloader, batch_transformer)
                writer.add_scalar('Test/loss', new_test_loss, step)

                # save checkpoint with best test loss
                if new_test_loss < best_test_loss or best_test_loss < 0:
                    logging.info(f"New best checkpoint found!")
                    if best_test_loss > 0:
                        logging.info(f"Removing old best checkpoint: {best_checkpoint_filename}")
                        os.remove(args.checkpoint_dir / best_checkpoint_filename)
                        if args.model == "lxmert":
                            os.remove(args.checkpoint_dir / best_checkpoint_box_filename)
                    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    best_checkpoint_filename = "best_global_step_features" + str(step) +".pt"
                    torch.save(visual_features, os.path.join(str(args.checkpoint_dir), best_checkpoint_filename))
                    if args.model == "lxmert":
                        best_checkpoint_box_filename = "best_global_step_boxes" + str(step) +".pt"
                        torch.save(visual_boxes, os.path.join(str(args.checkpoint_dir), best_checkpoint_box_filename))
                    best_test_loss = new_test_loss

                model.train()


@torch.no_grad()
def evaluate(model, visual_features, visual_boxes, device, test_dataloader, batch_transformer):
    losses = []  # List of scalar tensors
    for batch in tqdm(test_dataloader):
        # adapt batch to model
        batch = batch_transformer(batch, visual_features, visual_boxes)
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        losses.append(output.loss)
    
    stacked_losses = torch.stack(losses)  # (num_batches, )
    total_avg_loss = stacked_losses.mean()  # (num test examples, ) -> scalar

    print("Average test loss: " + str(total_avg_loss.item()))

    return total_avg_loss.item()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    main(args)
