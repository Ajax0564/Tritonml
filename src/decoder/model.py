from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from einops import rearrange

from ..ops.linear_rms import TritonLinearRMSNormLayer
from ..ops.mlp_gelu import TritonGeluMlpLayer
from ..ops.linear import TritonLinearLayer
from ..ops.causal_attention import TritonCausalAttention
from ..ops.rope import TritonRopeLayer
from ..ops.rms import TritonRMSNormLayer


@dataclass
class CLMOutput(object):
    logits: torch.Tensor = None
    kv_cache: List[torch.FloatTensor] = None

@dataclass
class Config:
    hidden_size: int = 768
    num_attention_heads: int = 12
    max_position_embeddings: int = 514
    num_hidden_layers: int = 4
    vocab_size: int = 50265
    hidden_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-05


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
    
class DecoderAttention(nn.Module):
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
        # self.qkv = nn.Linear(config.hidden_size,3*config.hidden_size)
        self.query =TritonLinearLayer(
            config.hidden_size, config.hidden_size, bias=self.attention_bias
        )
        self.key = TritonLinearLayer(
            config.hidden_size, config.hidden_size, bias=self.attention_bias
        )
        self.value = TritonLinearLayer(
            config.hidden_size, config.hidden_size, bias=self.attention_bias
        )
        self.out = TritonLinearRMSNormLayer(config.hidden_size, config.hidden_size)
        self.num_attention_heads = config.num_attention_heads
        self.sdpa = TritonCausalAttention(self.head_size**-0.5,True)
        self.rope = TritonRopeLayer()

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        kv_cache: List[torch.FloatTensor] = None,
        start_pos: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: torch.Tensor of shape (batch, seq_len, embed_dim)`
            Attention_mask: torch.Tensor of shape (batch,1, seq_len, seqlen)`
            freqs: Positional freqs in case of RoPE embedding
            use_cace: Optional to use kvCache
            start_pos: in case of kvCache to get store kv-cache at start_pos
        return:
               hidden_states: torch.Tensor of shape (batch, seq_len, embed_dim)

        """
        q = self.query(hidden_state)
        k = self.key(hidden_state)
        v = self.value(hidden_state)
        # transform it into batch_size x no_of_heads x seqlen x head_dim for Multihead Attention
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_attention_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_attention_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_attention_heads)

        if freqs is not None:
            q, k = self.rope(q.contiguous(), k.contiguous(), freqs) 

        if use_cache:
            if kv_cache is None:
                raise ValueError("you need to pass kv_cache")
            k, v = kv_cache.update(self.layer_idx, k, v, start_pos)

        out = self.sdpa(q,k,v,attention_mask,start_pos)
        # transform it back into batch_size x seqlen x hidden_dim
        out = rearrange(out, "b h l d -> b l (h d)")

        return self.out(out+hidden_state), kv_cache

    
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
    

class DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int, attention_type: str = None) -> None:
        super().__init__()
        self.attention = DecoderAttention(config, layer_idx=layer_idx)
        
        self.feed_forward = FeedForward(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        kv_cache: List[torch.FloatTensor] = None,
        start_pos: Optional[int] = 0,
    ) -> torch.Tensor:
        out,kv_cache = self.attention(
            hidden_state=hidden_state,
            attention_mask=attention_mask,
            freqs=freqs,
            use_cache=use_cache,
            kv_cache=kv_cache,
            start_pos=start_pos,
        )
        out = self.feed_forward(out, hidden_state)
        return out, kv_cache

class LMHead(nn.Module):
    """Head for masked language modelling"""

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = TritonLinearLayer(config.hidden_size, config.hidden_size)
        self.layerNorm =TritonRMSNormLayer(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = TritonLinearLayer(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_state)
        x = nn.GELU()(x)
        x = self.layerNorm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x
    

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple
import torch


class DynamicCache:
    """
    A cache that grows dynamically as more tokens are generated.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self, config, is_gqa: bool = False) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = False

        self.layers = config.num_hidden_layers
        for _ in range(self.layers):
            self.key_cache.append([])
            self.value_cache.append([])

    def __len__(self) -> int:
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        if len(self.key_cache) == 0:
            return 0
        return self.key_cache[0].shape[-2]

    def update(
        self,
        index: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """

        # Update the cache first iteration'

        if len(self.key_cache[index]) == 0:
            self._seen_tokens = True
            self.key_cache[index] = key_states.clone()
            self.value_cache[index] = value_states.clone()
        else:
            self.key_cache[index] = torch.cat(
                [self.key_cache[index], key_states], dim=-2
            )
            self.value_cache[index] = torch.cat(
                [self.value_cache[index], value_states], dim=-2
            )

        return self.key_cache[index], self.value_cache[index]

    def get(self, index: int) -> Tuple[torch.Tensor]:
        if self._seen_tokens:
            return self.key_cache[index], self.value_cache[index]
        else:
            raise ValueError("there is no token available in kv-cache")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if self.key_cache is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return self.max_cache_len


class StaticCache:
    """
    A cache that grows dynamically as more tokens are generated.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(
        self,
        config,
        max_cache_len: int = None,
        dtype: torch.dtype = torch.float32,
        batch_size: int = 1,
        is_gqa: bool = False,
    ) -> None:
        self.head_size = int(config.hidden_size // config.num_attention_heads)
        self.heads = None
        self.batch_size = batch_size
        # if is_gqa:
        self.heads = getattr(config, "num_key_value_heads", None)
        if self.heads is None:

            self.heads = config.num_attention_heads

        self.max_cache_len = (
            config.max_position_embeddings if max_cache_len is None else max_cache_len
        )

        self.dtype = dtype

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.cache_shape = (
            self.batch_size,
            self.heads,
            self.max_cache_len,
            self.head_size,
        )

        self._seen_tokens = False
        self.layers = config.num_hidden_layers
        for _ in range(self.layers):
            blank_key_cache = torch.zeros(
                self.cache_shape, dtype=self.dtype, device=self.device
            )
            blank_value_cache = torch.zeros(
                self.cache_shape, dtype=self.dtype, device=self.device
            )
            self.key_cache.append(blank_key_cache)
            self.value_cache.append(blank_value_cache)

    def __len__(self) -> int:
        if self.key_cache is None:
            return 0
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return self.key_cache.shape[-2]

    def update(
        self,
        index: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """

        # Update the cache first iteration'

        bsz, head, seqlen, _ = key_states.shape
        if seqlen > self.key_cache[index].size()[2]:
            raise ValueError(
                f"{k.shape} is more than init k_cache size {self.key_cache}"
            )

        self.key_cache[index][:bsz, :, start_pos : start_pos + seqlen] = key_states
        self.value_cache[index][:bsz, :, start_pos : start_pos + seqlen] = value_states

        k = self.key_cache[index][:bsz, :, : start_pos + seqlen]
        v = self.value_cache[index][:bsz, :, : start_pos + seqlen]

        return k, v

    def get(self, index: int) -> Tuple[torch.Tensor]:
        if self._seen_tokens:
            return self.key_cache[index], self.value_cache[index]
        else:
            raise ValueError("there is no token available in kv-cache")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if self.key_cache is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None
    
class DecoderModel(nn.Module):

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
        print(
            "Encoder Ignoring sinusoidal or absolute position embeddings because rope,is enable"
        )
        self.all_layer = nn.ModuleList(
            [
                DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.lm_head =  LMHead(config=config)
        self.config = config

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / torch.sqrt(2 * len(self.all_layer))
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / torch.sqrt(2 * len(self.all_layer))
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        kv_cache: List[torch.FloatTensor] = None,
        start_pos: Optional[int] = 0,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _bsz, seqlen = input_ids.shape
        hidden_state = self.word_embeddings(input_ids)
        freqs = self.emb_freq[start_pos : start_pos + seqlen,:].to(input_ids.device)
        mask = None
        if seqlen > 1:
            mask = self.create_mask_for_decoder(
                input_ids=input_ids, attention_mask=attention_mask, start_pos=start_pos
            )
            mask = (1.0 - mask) * torch.finfo(
                hidden_state.dtype
            ).min  # invert it to to add directly to attention score

        for layer in self.all_layer:
            hidden_state, kv_cache = layer(
                hidden_state,
                mask,
                freqs=freqs,
                use_cache=use_cache,
                kv_cache=kv_cache,
                start_pos=start_pos,
            )
      
        logits = self.lm_head(hidden_state)
        return CLMOutput(logits=logits, kv_cache=kv_cache)
