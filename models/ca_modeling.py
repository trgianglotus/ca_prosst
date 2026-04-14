"""
CA-ProSST: confidence-aware extension of ProSST.

Integration point: ss_hidden_states is produced once by ProSSTEmbeddings and then
passed unchanged into every disentangled-attention layer (aa2ss / ss2aa terms).
Scaling or masking ss_hidden_states before the encoder call applies the CA
signal uniformly to all layers without touching attention internals.

Four variants, selected by `config.ca_mode`:
    - "none": pass-through, equivalent to vanilla ProSST
    - "hard": zero ss embeddings for residues with pLDDT < ca_threshold
    - "soft": multiply ss embeddings by (pLDDT / 100)
    - "gate": learned per-residue sigmoid gate over [aa_embedding, conf]
    - "zero": zero all ss embeddings (sequence-only ablation; no pLDDT needed)

pLDDT input convention:
    Tensor of shape [B, L] in the 0..100 range, aligned to input_ids *including*
    the CLS/EOS special tokens. Special positions should be set to 100 upstream
    so they are never masked (they carry no residue semantics anyway).
    Missing structure: set pLDDT to 0 so hard-masking removes them entirely.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MaskedLMOutput

from .configuration_prosst import ProSSTConfig
from .modeling_prosst import ProSSTForMaskedLM


class CAProSSTConfig(ProSSTConfig):
    model_type = "CAProSST"

    def __init__(
        self,
        ca_mode: str = "none",
        ca_threshold: float = 70.0,
        ca_gate_hidden: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ca_mode = ca_mode
        self.ca_threshold = float(ca_threshold)
        self.ca_gate_hidden = int(ca_gate_hidden)


class CAProSSTForMaskedLM(ProSSTForMaskedLM):
    """ProSSTForMaskedLM with a confidence-aware transform on ss_embeddings.

    Loads cleanly from the HF `AI4Protein/ProSST-2048` checkpoint: the only new
    parameters (for `ca_mode="gate"`) live under `ca_gate.*` and are reported as
    missing keys on load, which HF initializes from the base `_init_weights`.
    """

    config_class = CAProSSTConfig

    def __init__(self, config):
        super().__init__(config)
        self.ca_mode = getattr(config, "ca_mode", "none")
        self.ca_threshold = float(getattr(config, "ca_threshold", 70.0))
        if self.ca_mode == "gate":
            h = config.hidden_size
            mlp_h = int(getattr(config, "ca_gate_hidden", 128))
            self.ca_gate = nn.Sequential(
                nn.Linear(h + 1, mlp_h),
                nn.GELU(),
                nn.Linear(mlp_h, 1),
            )
            for m in self.ca_gate:
                if isinstance(m, nn.Linear):
                    self._init_weights(m)
            # Bias the final linear so the gate starts near 1 (behaves like
            # vanilla ProSST at init — avoids destroying the pretrained signal
            # before the gate has learned anything).
            with torch.no_grad():
                self.ca_gate[-1].bias.fill_(2.0)
        else:
            self.ca_gate = None

    def freeze_base(self) -> None:
        """Freeze every parameter except the CA gate. No-op for non-gate modes."""
        for p in self.parameters():
            p.requires_grad = False
        if self.ca_gate is not None:
            for p in self.ca_gate.parameters():
                p.requires_grad = True

    def _apply_ca(
        self,
        ss_embeddings: torch.Tensor,
        aa_embeddings: torch.Tensor,
        plddt: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.ca_mode == "zero":
            return torch.zeros_like(ss_embeddings)
        if plddt is None or self.ca_mode == "none":
            return ss_embeddings
        conf = torch.clamp(plddt.to(ss_embeddings.dtype) / 100.0, 0.0, 1.0)

        if self.ca_mode == "hard":
            thr = self.ca_threshold / 100.0
            gate = (conf >= thr).to(ss_embeddings.dtype)
            return ss_embeddings * gate.unsqueeze(-1)

        if self.ca_mode == "soft":
            return ss_embeddings * conf.unsqueeze(-1)

        if self.ca_mode == "gate":
            feats = torch.cat([aa_embeddings, conf.unsqueeze(-1)], dim=-1)
            gate = torch.sigmoid(self.ca_gate(feats))
            return ss_embeddings * gate

        raise ValueError(f"Unknown ca_mode: {self.ca_mode}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        ss_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        plddt: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("must specify input_ids or inputs_embeds")

        device = (input_ids if input_ids is not None else inputs_embeds).device
        input_shape = (
            input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]
        )
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output, ss_embeddings = self.prosst.embeddings(
            input_ids=input_ids,
            ss_input_ids=ss_input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        if ss_embeddings is not None:
            ss_embeddings = self._apply_ca(ss_embeddings, embedding_output, plddt)

        encoder_outputs = self.prosst.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True,
            ss_hidden_states=ss_embeddings,
        )
        sequence_output = encoder_outputs.last_hidden_state
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,)
            if output_hidden_states:
                output = output + (encoder_outputs.hidden_states,)
            if output_attentions:
                output = output + (encoder_outputs.attentions,)
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )
