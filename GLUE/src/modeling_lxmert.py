from transformers import LxmertPreTrainedModel, LxmertModel, LxmertConfig, PreTrainedTokenizerBase
from transformers.modeling_outputs import SequenceClassifierOutput

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional, Union, List, Tuple, Dict, Any

LXMERT_FEATURES_SHAPE = (1, 2048)
LXMERT_NORMALIZED_BOXES_SHAPE = (1, 4)

def get_lxmert_batch(batch, visual_feats=None, visual_boxes=None):
    # align the visual inputs
    batch_size = len(batch)
    if visual_feats is None:
        visual_feats = torch.empty((batch_size,)+LXMERT_FEATURES_SHAPE).uniform_(0, 10)
    else:
        visual_feats = visual_feats.unsqueeze(0).repeat(batch_size, 1, 1)
    if visual_boxes is None:
        visual_pos = torch.empty((batch_size,)+LXMERT_NORMALIZED_BOXES_SHAPE).uniform_(0, 1)
    else:
        visual_pos = visual_boxes.unsqueeze(0).repeat(batch_size, 1, 1)

    batch.update({
        "visual_feats": visual_feats,
        "visual_pos": visual_pos
    })

    return batch

@dataclass
class DataCollatorWithPaddingSkippingFeatures:
    """
    Data collator that will dynamically pad the inputs received, adapted to Lxmert. Will not touch visual features.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    features_to_skip: tuple = ()

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        separated_features = {key: [example[key] for example in features] for key in features[0].keys()}
        features_to_pad = {key: separated_features[key] for key in separated_features.keys() if key not in self.features_to_skip}
        batch = self.tokenizer.pad(
            features_to_pad,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # add features that should not be padded to batch, if they are there
        for key in self.features_to_skip:
            if key in separated_features:
                batch[key] = torch.stack(separated_features[key], 0)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

class LxmertConfigForSequenceClassification(LxmertConfig):
    r"""
    A standard Lxmert config adapted for sequence classification.
    """

    def __init__(
        self,
        classifier_dropout = None,
        **kwargs,
    ):
        self.classifier_dropout = classifier_dropout
        super().__init__(**kwargs)

class LxmertForSequenceClassification(LxmertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.lxmert = LxmertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        visual_feats: Optional[torch.FloatTensor] = None,
        visual_pos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        visual_attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        obj_labels: Optional[Dict[str, Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        matched_label: Optional[torch.LongTensor] = None,
        ans: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.lxmert(
            input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        pooled_output = outputs[2]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[3] + outputs[5]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.language_hidden_states,
            attentions=outputs.language_attentions,
        )