# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod, ABC
import torch
from torch import nn
import math
import os

from vllm.logger import init_logger

logger = init_logger(__name__)


class TokenformerMLPAdapter(nn.Module):
    def __init__(self, layer, hidden_size, device):
        super().__init__()
        # Register the wrapped layer as a submodule so it's properly tracked
        self.layer = layer if isinstance(layer, nn.Module) else layer
        self.hidden_size = hidden_size
        self.num_heads = int(os.getenv("TOKENFORMER_NUM_HEADS", "4"))
        self.head_dim = hidden_size // self.num_heads
        self.tokenformer_r = int(os.getenv("TOKENFORMER_R", "32"))
        self.dtype = next(layer.parameters()).dtype

        # Register tokenformer parameters
        self.tokenformer_k = nn.Parameter(
            torch.zeros(self.num_heads, self.hidden_size, device=device, dtype=self.dtype)
        )
        self.tokenformer_v = nn.Parameter(
            torch.zeros(
                self.num_heads, self.hidden_size * self.tokenformer_r, device=device, dtype=self.dtype
            )
        )
        self.tokenformer_p = nn.Parameter(
            torch.zeros(self.tokenformer_r, self.hidden_size, device=device, dtype=self.dtype)
        )

        self.reset_parameters()

    def reset_parameters(self):
        k_gain = 3.0 / math.sqrt(self.hidden_size / self.num_heads)
        v_gain = 3.0 / math.sqrt(self.hidden_size)

        k_init_tensor = torch.empty_like(self.tokenformer_k, dtype=torch.bfloat16)
        torch.nn.init.normal_(k_init_tensor, std=k_gain)
        self.tokenformer_k.data.copy_(k_init_tensor)

        v_init_tensor = torch.empty_like(self.tokenformer_v, dtype=torch.bfloat16)
        torch.nn.init.uniform_(v_init_tensor, a=-v_gain, b=v_gain)
        self.tokenformer_v.data.copy_(v_init_tensor)

        p_init_tensor = torch.empty_like(self.tokenformer_p, dtype=torch.bfloat16)
        torch.nn.init.zeros_(p_init_tensor)
        self.tokenformer_p.data.copy_(p_init_tensor)

    # Call layer with all inputs and kwargs
    def forward(self, query: torch.Tensor):
        try:
            base_layer_results = self.layer(query)
        except Exception as e:
            logger.error(f"Error calling wrapped layer {type(self.layer).__name__}: {e}")
            logger.error(f"Layer attributes: {dir(self.layer)}")
            raise

        tokenformer_results = self.tokenformer_op_1(query)

        # sum the two outputs
        layer_and_adaptor_sum = base_layer_results + tokenformer_results
        return layer_and_adaptor_sum

    def tokenformer_op(self, query):

        return query @ self.tokenformer_k.transpose(0, 1) @ self.tokenformer_v

    def tokenformer_op_1(self, query):

        q = query.view(
            -1, self.num_heads, self.hidden_size // self.num_heads
        ).transpose(0, 1)
        k = self.tokenformer_k.view(
            -1, self.num_heads, self.hidden_size // self.num_heads
        ).transpose(0, 1)
        v = self.tokenformer_v.view(
            -1, self.num_heads, self.hidden_size * self.tokenformer_r // self.num_heads
        ).transpose(0, 1)

        result = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k.to(q.dtype),
            value=v.to(q.dtype),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,  # should be false for tokenformer
        )

        proj_down = (
            result.transpose(0, 1)
            .contiguous()
            .view([-1, self.hidden_size, self.tokenformer_r])
        )

        # tokenformer_p dims are [tokenformer_r, hidden_size]
        # query dims are [batch size, length, 1, hidden_size]
        # proj_down are [batch size, length, hidden_size, tokenformer_r]

        query_batch = query.view([-1, 1, self.hidden_size])

        result = torch.bmm(query_batch, proj_down) @ self.tokenformer_p.to(q.dtype)

        return result.view(query.shape)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override state_dict to include both adapter and wrapped layer parameters."""
        if destination is None:
            destination = {}
        
        # Add tokenformer parameters
        destination[prefix + 'tokenformer_k'] = self.tokenformer_k if keep_vars else self.tokenformer_k.data
        destination[prefix + 'tokenformer_v'] = self.tokenformer_v if keep_vars else self.tokenformer_v.data
        destination[prefix + 'tokenformer_p'] = self.tokenformer_p if keep_vars else self.tokenformer_p.data
        
        # Add wrapped layer's parameters with appropriate prefix
        # The wrapped layer's params should appear at the same level, not under 'layer'
        layer_state = self.layer.state_dict(prefix=prefix, keep_vars=keep_vars)
        destination.update(layer_state)
        
        return destination
    
    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle both adapter and wrapped layer parameters."""
        # Extract tokenformer parameters
        tokenformer_state = {}
        layer_state = {}
        
        for key, value in state_dict.items():
            if 'tokenformer_' in key:
                # Remove any prefix to get the parameter name
                param_name = key.split('.')[-1]
                if hasattr(self, param_name):
                    tokenformer_state[param_name] = value
            else:
                # This belongs to the wrapped layer
                layer_state[key] = value
        
        # Load tokenformer parameters
        for param_name, value in tokenformer_state.items():
            getattr(self, param_name).data.copy_(value)
        
        # Load wrapped layer parameters
        if layer_state:
            self.layer.load_state_dict(layer_state, strict=False)
    
    # Visualize the size of the parameters
    def __repr__(self):
        return (
            f"TokenformerMLPAdapter(\nhidden_size={self.hidden_size}\n(layer): "
            + self.layer.__repr__()
            + "\n)"
        )


class TokenformerAttentionAdapter(nn.Module):
    def __init__(self, layer, hidden_size, device):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
        self.dtype = next(layer.parameters()).dtype

        self.tokenformer_k = nn.Parameter(
            torch.zeros(self.hidden_size, self.hidden_size, device=device, dtype=self.dtype)
        )
        self.tokenformer_v = nn.Parameter(
            torch.zeros(self.hidden_size, self.hidden_size, device=device, dtype=self.dtype)
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = 3.0 / math.sqrt(self.hidden_size)

        k_init_tensor = torch.empty_like(self.tokenformer_k, dtype=torch.bfloat16)
        torch.nn.init.zeros_(k_init_tensor)
        self.tokenformer_k.data.copy_(k_init_tensor)

        v_init_tensor = torch.empty_like(self.tokenformer_v, dtype=torch.bfloat16)
        torch.nn.init.normal_(v_init_tensor, std=gain)
        self.tokenformer_v.data.copy_(v_init_tensor)

        # For the sliced operations, create tensors matching the slice shapes
        k_slice_init_tensor = torch.empty_like(self.tokenformer_k[0:1, :], dtype=torch.bfloat16)
        torch.nn.init.normal_(k_slice_init_tensor, std=gain)
        self.tokenformer_k.data[0:1, :].copy_(k_slice_init_tensor)

        v_slice_init_tensor = torch.empty_like(self.tokenformer_v[0:1, :], dtype=torch.bfloat16)
        torch.nn.init.zeros_(v_slice_init_tensor)
        self.tokenformer_v.data[0:1, :].copy_(v_slice_init_tensor)

    def forward(self, query, base_layer_results) -> torch.Tensor:

        tokenformer_results = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=self.tokenformer_k,
            value=self.tokenformer_v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,  # should be false for tokenformer
        )

        # sum the two outputs
        layer_and_adaptor_sum = base_layer_results + tokenformer_results
        return layer_and_adaptor_sum

    def __repr__(self):
        return (
            f"TokenformerAttentionAdapter(\nhidden_size={self.hidden_size}\n(layer): "
            + self.layer.__repr__()
            + "\n)"
        )


class TokenformerSurgeon(ABC):

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
        # Get config from model - handle different model structures
        if hasattr(model, 'config'):
            self.config = model.config
        elif hasattr(model, 'model') and hasattr(model.model, 'config'):
            self.config = model.model.config
        else:
            raise ValueError("Cannot find config in model structure")
            
        logger.info(f"TokenformerSurgeon initialized with config: hidden_size={self.config.hidden_size}")

    def _is_attn_layer(self, layer_name):
        return layer_name.split(".")[-1] == "attn"

    def _is_mlp_layer(self, layer_name):
        # Only wrap the MLP module itself, not its sub-modules like gate_up_proj
        return layer_name.endswith('.mlp')

    def _recursive_setattr(self, obj, attr, value):
        attr = attr.split(".", 1)
        if len(attr) == 1:
            setattr(obj, attr[0], value)
        else:
            self._recursive_setattr(getattr(obj, attr[0]), attr[1], value)

    def update_mlp(self, name, layer):
        """Try to wrap the layer with a TokenformerMLPAdaptor."""
        if not self._is_mlp_layer(name):
            return
        
        # Check if already wrapped to prevent double wrapping
        if isinstance(layer, TokenformerMLPAdapter):
            logger.warning(f"Layer {name} is already wrapped with TokenformerMLPAdapter, skipping")
            return

        logger.info(f"Wrapping layer {name} with TokenformerMLPAdaptor, layer type: {type(layer).__name__}")
        
        # Debug: Check what attributes the layer has
        if hasattr(layer, 'gate_up_proj'):
            logger.debug(f"  - Layer has gate_up_proj attribute")
        if hasattr(layer, 'forward'):
            logger.debug(f"  - Layer has forward method")

        # Wrap the layer with a TokenformerMLPAdapter
        self._recursive_setattr(
            self.model,
            name,
            TokenformerMLPAdapter(
                layer, self.config.hidden_size, device=self.device
            ),
        )

    @abstractmethod
    def update_attn(self, name, layer):
        pass

    def insert_adapter_modules(self):
        # Add tokenformer adapters for mlp and attention
        logger.info(f"Starting tokenformer adapter insertion...")
        mlp_count = 0
        attn_count = 0
        wrapped_modules = set()
        
        # Debug: Log all module names first
        all_modules = list(self.model.named_modules())
        logger.info(f"Found {len(all_modules)} total modules in model")
        for name, layer in all_modules[:10]:  # Log first 10 modules
            logger.debug(f"  Module: {name} -> {type(layer).__name__}")
        
        for name, layer in self.model.named_modules():
            # Skip if this is a sub-module of an already wrapped module
            if any(name.startswith(wrapped + '.') for wrapped in wrapped_modules):
                logger.debug(f"Skipping {name} (sub-module of wrapped module)")
                continue
                
            if self._is_mlp_layer(name):
                logger.info(f"Found MLP layer to wrap: {name}")
                self.update_mlp(name, layer)
                if not isinstance(layer, TokenformerMLPAdapter):  # Only count if actually wrapped
                    mlp_count += 1
                    wrapped_modules.add(name)
            elif self._is_attn_layer(name):
                logger.info(f"Found attention layer to wrap: {name}")
                self.update_attn(name, layer)
                attn_count += 1
                wrapped_modules.add(name)
        
        logger.info(f"Tokenformer adapter insertion complete: {mlp_count} MLP layers, {attn_count} attention layers wrapped")
        
        # Debug: Check model structure after wrapping
        logger.info("Model structure after wrapping:")
        for name, module in self.model.named_modules():
            if 'mlp' in name:
                logger.info(f"  {name}: {type(module).__name__}")
                if hasattr(module, 'tokenformer_k'):
                    logger.info(f"    -> Has tokenformer_k parameter")
                if hasattr(module, 'gate_up_proj'):
                    logger.info(f"    -> Has gate_up_proj attribute")
        
        return self.model
