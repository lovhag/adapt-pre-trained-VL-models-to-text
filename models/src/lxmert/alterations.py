from transformers import LxmertXLayer  
import torch

try:
    from transformers import LxmertForQuestionAnswering
except:
    from transformers import LxmertForQuestionAnswering

class LxmertLanguageOnlyXLayer(LxmertXLayer):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        lang_feats,
        lang_attention_mask,
        visual_feats,
        visual_attention_mask,
        output_attentions=False,
    ):
        # handle cross attention without visual data to language processing
        lang_att_output = self.visual_attention.output(torch.zeros_like(lang_feats), lang_feats)
        
        attention_probs = None #lang_att_output[1:]
        lang_att_output, visual_att_output = self.self_att( # no mix
            lang_att_output,
            lang_attention_mask,
            torch.zeros_like(visual_feats), # replace visual with zeros, won't affect language
            visual_attention_mask,
        )

        lang_output, visual_output = self.output_fc(lang_att_output, visual_att_output) # no mix
        return (
            (
                lang_output,
                visual_output,
                attention_probs[0],
            )
            if output_attentions
            else (lang_output, visual_output)
        )