import torch
import logging
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, SequenceClassifierOutput, MaskedLMOutput, BertOnlyMLMHead


logger = logging.getLogger(__name__)

CLIP_EMBED_DIM = 512


class BertImageEmbeddings(BertEmbeddings):
    """
    Patched version of BertEmbeddings where no positional nor token_type embeddings are added where token_type_ids is -1
    """
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(torch.maximum(token_type_ids, torch.tensor(0, dtype=torch.long)))
        token_type_embeddings[token_type_ids == -1] = 0

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            position_embeddings = position_embeddings.repeat(token_type_ids.shape[0], 1, 1)
            position_embeddings[token_type_ids == -1] = 0
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertImageModel(BertModel):
    """
    Extends BertModel class to add image features to input (along with a projection layer to match transformer dim)
    It does so by using the `inputs_embeds` argument to the `forward` method
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.embeddings = BertImageEmbeddings(config)
        self.img_projection = torch.nn.Linear(config.img_feature_dim, self.config.hidden_size, bias=True)
        logger.info('BertImgModel Image Dimension: {}'.format(config.img_feature_dim))

    def forward(self, 
                input_ids,              # [batch, seq_len]
                img_feats=None,         # [batch, num_img_features, img_feature_dim]
                attention_mask=None,    # [batch, seq_len]
                token_type_ids=None,    # [batch, seq_len]
                inputs_embeds=None,
                **kwargs):

        device = input_ids.device
        inputs_embeds = self.embeddings.word_embeddings(input_ids)  # [batch, seq_len, hidden_size]

        # Image features
        if img_feats is not None:

            # Patch token_type_ids by adding -1 columns for image features
            if token_type_ids is None:
                token_type_ids = torch.zeros(inputs_embeds.size()[:-1], dtype=torch.long, device=device)
            minus_ones = -torch.ones((token_type_ids.shape[0], img_feats.shape[1]), 
                                        dtype=token_type_ids.dtype,
                                        device=device)
            token_type_ids = torch.cat((token_type_ids, minus_ones), dim=1)

            # Patch attention_mask
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=device)
            attention_mask = torch.cat((attention_mask, torch.ones(attention_mask.shape[0], 1, dtype=torch.long, device=device)), dim=1)

            proj_img_feats = self.img_projection(img_feats)
            inputs_embeds = torch.cat((inputs_embeds, proj_img_feats), dim=1)


        return super().forward(input_ids=None,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               inputs_embeds=inputs_embeds,
                               **kwargs)


class BertImageForMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        config.img_feature_dim = CLIP_EMBED_DIM
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertImageModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        img_feats=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            img_feats=img_feats,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        # Take only text token scores for MLM
        prediction_scores = prediction_scores[:, :input_ids.shape[1], :].contiguous()

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )