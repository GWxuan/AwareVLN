import inspect
import os
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.loss import soft_cross_entropy

from ...train.utils import calculate_loss_weight
from ..configuration_llava import LlavaConfig
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaLlamaConfig(LlavaConfig):
    model_type = "awarevln_llama"


class AwareVLNModel(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    """AwareVLN multimodal model (Llama + SigLIP) with reasoning/action token supervision."""

    config_class = LlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True

    def __init__(self, config: LlavaLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.init_vlm(config=config, *args, **kwargs)

        self.reason_token_id = getattr(config, "reason_token_id", None)
        self.act_token_id = getattr(config, "act_token_id", None)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(LlavaMetaModel, cls).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dpo_forward: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.freezed_module_patch()

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )

        support_packing = "seqlens_in_batch" in inspect.signature(self.llm.forward).parameters

        new_attention_mask = attention_mask
        new_position_ids = position_ids
        new_inputs_embeds = inputs_embeds
        new_labels = labels
        sorted_seqlens_in_batch = attention_mask.sum(-1).int()
        new_input_ids = input_ids

        if support_packing:
            outputs = self.llm.forward(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                seqlens_in_batch=sorted_seqlens_in_batch,
            )
        else:
            outputs = self.llm.forward(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        custom_soft_tokens = []
        if self.config.time_token_ids:
            custom_soft_tokens.extend(self.config.time_token_ids)

        if self.reason_token_id is not None:
            custom_soft_tokens.append(self.reason_token_id)
        if self.act_token_id is not None:
            custom_soft_tokens.append(self.act_token_id)

        if self.training and custom_soft_tokens and new_labels is not None:
            outputs.loss = soft_cross_entropy(
                outputs.logits,
                new_labels,
                soft_tokens=custom_soft_tokens,
                std=self.config.soft_ce_std,
            )

        loss_weight = calculate_loss_weight(new_labels)
        outputs.loss = outputs.loss * loss_weight

        if dpo_forward:
            return outputs.logits, new_labels

        return outputs


# Backward-compatible aliases for checkpoints / configs saved with older class names.
LlavaLlamaModel = AwareVLNModel
OneTwoVLANAVIDModel = AwareVLNModel

MODEL_CLASS_ALIASES = {
    "AwareVLNModel": AwareVLNModel,
    "LlavaLlamaModel": AwareVLNModel,
    "OneTwoVLANAVIDModel": AwareVLNModel,
}


def resolve_model_class(architecture: str) -> type:
    """Map config.architectures[0] to the current model class."""
    if architecture not in MODEL_CLASS_ALIASES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Supported: {sorted(MODEL_CLASS_ALIASES.keys())}"
        )
    return MODEL_CLASS_ALIASES[architecture]


REASON_TOKEN_STR = "<BEGIN_OF_REASONING>"
ACT_TOKEN_STR = "<BEGIN_OF_ACTION>"

# Register all historical model_type strings so old checkpoints load config correctly.
for _model_type in ("awarevln_llama", "llava_llama", "onetwovla_navid_llama"):
    AutoConfig.register(_model_type, LlavaLlamaConfig)

AutoModel.register(LlavaLlamaConfig, AwareVLNModel)
