from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from einops import rearrange

from ..ops.linear_rms import TritonLinearRMSNormLayer
from ..ops.mlp_gelu import TritonGeluMlpLayer
from ..ops.linear import TritonLinearLayer
from ..ops.encoder_attention import TritonMaskedAttention
from ..ops.rope import TritonRopeLayer
from ..ops.rms import TritonRMSNormLayer

@dataclass
class Config:
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    max_position_embeddings: int = 514
    num_hidden_layers: int = 4
    vocab_size: int = 50265
    hidden_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-05
    hidden_act: str = "gelu"


class RotaryEmbedding(nn.Module):
    def __init__(self, config, base=10000, device=None):
        super().__init__()

        self.dim = int(config.hidden_size // config.num_attention_heads)
        self.max_position_embeddings = config.max_position_embeddings
        self.base = base
        self.register_buffer(
            "inv_freq",
            1.0
            / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
            ),
            persistent=False,
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, seq_len: int = None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # size = x.size()[2]
        position_ids = torch.arange(seq_len).unsqueeze(0)
        # position_ids = self.position_ids[:, :size].float()

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            1, 2
        )
        return freqs
    
class EncoderAttention(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.head_size = int(config.hidden_size // config.num_attention_heads)
        self.attention_bias = getattr(config, "attention_bias", True)
        self.layer_idx = layer_idx
        
        self.q = TritonLinearLayer(
            config.hidden_size, config.hidden_size, bias=self.attention_bias
        )
        self.k = TritonLinearLayer(
            config.hidden_size, config.hidden_size, bias=self.attention_bias
        )
        self.v = TritonLinearLayer(
            config.hidden_size, config.hidden_size, bias=self.attention_bias
        )
        self.out = TritonLinearRMSNormLayer(config.hidden_size, config.hidden_size) 
        self.num_attention_heads = config.num_attention_heads
        self.rope = TritonRopeLayer()
        self.sdpa = TritonMaskedAttention(self.head_size**-0.5)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.q(hidden_state)
        k = self.k(hidden_state)
        v = self.v(hidden_state)
       
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_attention_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_attention_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_attention_heads)
        if freqs is not None:
            q, k = self.rope(q.contiguous(), k.contiguous(), freqs) 

        out = self.sdpa(q,k,v,attention_mask)
        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.out(out+hidden_state)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, config, multiplier: Union[int, float] = 4) -> None:
        super().__init__()
        self.mlp = TritonGeluMlpLayer(config.hidden_size, int(multiplier) * config.hidden_size,config.hidden_size)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layerNorm = TritonRMSNormLayer(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self, hidden_state: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        output = self.mlp(hidden_state)
        output = self.dropout(output)
        output = self.layerNorm(output + input_tensor)
        return output
    

class EncoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, attention_type: str = None) -> None:
        super().__init__()
        self.attention =  EncoderAttention(config, layer_idx=layer_idx)
        self.feed_forward = FeedForward(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs: torch.Tensor = None,
    ) -> torch.Tensor:
        out = self.attention(
            hidden_state=hidden_state, attention_mask=attention_mask, freqs=freqs
        )
        out = self.feed_forward(out, hidden_state)
        return out
    

class EncoderModel(nn.Module):

    def __init__(
        self,
        config
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.emb_freq = torch.squeeze(RotaryEmbedding(config)(config.max_position_embeddings),dim=0)
        
            
        self.all_layer = nn.ModuleList(
            [
                EncoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        hidden_state = self.word_embeddings(input_ids)
        freqs = self.emb_freq[:seqlen,:].to(input_ids.device)

        attention_mask = attention_mask.type_as(hidden_state)
        attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_state.dtype).min

        for layer in self.all_layer:
            hidden_state = layer(hidden_state, attention_mask, freqs)
        return hidden_state

    @classmethod
    def from_config(
        cls,
        config
    ) -> nn.Module:
        return cls(config)