# Adapted from https://github.com/huggingface/transformers/blob/9cf4f2aa9a9cecbb22e813931ef3bb72fc773540/src/transformers/models/phi/modeling_phi.py

from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


class DecomposedPhiCausalLM():
    """
    PhiCausalLM that uses the decomposed layer processing
    """
    def __init__(self, model):
        # Get the transformer part
        self.transformer = model.model
        self.lm_head = model.lm_head
        # self.original_forward = model.forward
        
        # Store references to transformer components
        self.layers = self.transformer.layers
        self.num_layers = len(self.layers)
        self.dtype = self.transformer.dtype
        self.config = self.transformer.config
        self._update_causal_mask = self.transformer._update_causal_mask

    def forward_till_penultimate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> dict:
        """
        Run forward pass through all layers except the last one.
        Returns intermediate states needed for the final layer.
        """
        output_attentions = output_attentions if output_attentions is not None else self.transformer.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.transformer.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.transformer.config.use_cache
        return_dict = return_dict if return_dict is not None else self.transformer.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.transformer.gradient_checkpointing and self.transformer.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            # print(
            #     "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            #     "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            # )

        # Embeddings
        if inputs_embeds is None:
            inputs_embeds = self.transformer.embed_tokens(input_ids)
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        inputs_embeds = self.transformer.embed_dropout(inputs_embeds)
        hidden_states = inputs_embeds

        # Prepare for layer processing
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        
        # Process all layers EXCEPT the last one
        for decoder_layer in self.layers[: self.num_layers - 1]: 
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.transformer.gradient_checkpointing and self.transformer.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    output_attentions,
                    use_cache,
                    past_key_values,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Return intermediate state (before final layer)
        return {
            'hidden_states': hidden_states,
            'past_key_values': past_key_values,
            'all_hidden_states': all_hidden_states,
            'all_self_attns': all_self_attns,
            'position_ids': position_ids,
            'use_cache': use_cache,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'use_legacy_cache': use_legacy_cache,
            'next_decoder_cache': next_decoder_cache,
            'causal_mask': causal_mask,
            'cache_position': cache_position
        }
    
    def forward_final_layer(
        self,
        intermediate_outputs: dict,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Run the final decoder layer, apply final layer norm, compute logits and loss.
        """
        # Extract intermediate state
        hidden_states = intermediate_outputs['hidden_states']
        all_hidden_states = intermediate_outputs['all_hidden_states']
        all_self_attns = intermediate_outputs['all_self_attns']
        position_ids = intermediate_outputs['position_ids']
        past_key_values = intermediate_outputs['past_key_values']
        use_cache = intermediate_outputs['use_cache']
        output_attentions = intermediate_outputs['output_attentions']
        output_hidden_states = intermediate_outputs['output_hidden_states']
        use_legacy_cache = intermediate_outputs['use_legacy_cache']
        next_decoder_cache = intermediate_outputs['next_decoder_cache']
        causal_mask = intermediate_outputs['causal_mask']
        cache_position = intermediate_outputs['cache_position']

        # Process the FINAL layer (last layer)
        decoder_layer = self.layers[self.num_layers - 1]
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Run final layer
        if self.transformer.gradient_checkpointing and self.transformer.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                output_attentions,
                use_cache,
                past_key_values,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
            
        # Apply final layer norm
        hidden_states = self.transformer.final_layernorm(hidden_states)

        # Add final hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        # Compute logits
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # Compute loss
        loss = None
        per_sample_losses = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if hasattr(self, '_compute_per_sample_loss') and self._compute_per_sample_loss:
                batch_size, seq_len = shift_labels.shape
                # Reshape for per-sample loss computation
                shift_logits_flat = shift_logits.view(batch_size, seq_len, -1)
                shift_labels_flat = shift_labels.view(batch_size, seq_len).to(shift_logits.device)
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                per_token_losses = loss_fct(
                    shift_logits_flat.view(-1, self.config.vocab_size),
                    shift_labels_flat.view(-1)
                ).view(batch_size, seq_len)
                per_sample_losses = per_token_losses.mean(dim=1)
                loss = per_sample_losses.mean()
            else:
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + layer_outputs[1:]
            if per_sample_losses is not None:
                return (per_sample_losses,) + output
            elif loss is not None:
                return (loss,) + output
            else:
                return output

        return CausalLMOutputWithPast(
            loss=per_sample_losses if per_sample_losses is not None else loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
