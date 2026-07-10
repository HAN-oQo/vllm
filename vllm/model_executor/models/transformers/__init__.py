# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrapper around `transformers` models"""

from typing import TYPE_CHECKING

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vllm.model_executor.models.transformers.base import Base
from vllm.model_executor.models.transformers.causal import CausalMixin
from vllm.model_executor.models.transformers.legacy import LegacyMixin
from vllm.model_executor.models.transformers.moe import MoEMixin
from vllm.model_executor.models.transformers.multimodal import (
    MultiModalDummyInputsBuilder,
    MultiModalMixin,
    MultiModalProcessingInfo,
    MultiModalProcessor,
)
from vllm.model_executor.models.transformers.pooling import (
    EmbeddingMixin,
    SequenceClassificationMixin,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention import Attention


def vllm_attention_forward(
    # Transformers args
    module: "torch.nn.Module",
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    attention_mask: "torch.Tensor",
    # Transformers kwargs
    scaling: float | None = None,
    # vLLM kwargs
    attention_instances: dict[int, "Attention"] | None = None,
    **kwargs,
):
    self_attn = attention_instances[module.layer_idx]
    if scaling is not None:
        self_attn.impl.scale = float(scaling)
    hidden = query.shape[-2]
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))

    # MLA models (DeepSeek-V2/V3/etc.): `value`'s real per-head dim (`v_head_dim`) can be
    # narrower than `query`/`key`'s (`qk_nope_head_dim + qk_rope_head_dim`). `Attention` is
    # configured with one uniform `head_size` sized to query/key's width (see
    # `create_attention_instances`), matching vLLM's own native "naive" MLA fallback,
    # `DeepseekV2Attention` (vllm/model_executor/models/deepseek_v2.py:601-609), which pads
    # value up to that width with zeros before calling `Attention`, then slices the output
    # back down afterward. Mirror that here -- a no-op (v_head_dim == head_size) for every
    # non-MLA model.
    #
    # The pad/slice must be unconditional (no `if v_head_dim < head_size:` branch): under
    # full torch.compile + CUDA-graph capture, a Python-level branch around this allocation
    # produces incorrect (garbled) output on replay even though the branch always resolves
    # the same way for a given layer -- native's version never branches, which is why it
    # doesn't hit this. `F.pad` with zero padding and a full-range slice are both no-ops when
    # v_head_dim == head_size, so dropping the branch changes nothing for non-MLA models.
    head_size = self_attn.head_size
    v_head_dim = value.shape[-1]
    value = torch.nn.functional.pad(value, [0, head_size - v_head_dim])

    query, key, value = (x.reshape(hidden, -1) for x in (query, key, value))
    output = self_attn.forward(query, key, value)

    output = output.view(hidden, self_attn.num_heads, head_size)[..., :v_head_dim]
    output = output.reshape(hidden, -1)

    return output, None


ALL_ATTENTION_FUNCTIONS["vllm"] = vllm_attention_forward


# Text only models
class TransformersForCausalLM(CausalMixin, Base): ...


class TransformersMoEForCausalLM(MoEMixin, CausalMixin, Base): ...


# Multimodal models
@MULTIMODAL_REGISTRY.register_processor(
    MultiModalProcessor,
    info=MultiModalProcessingInfo,
    dummy_inputs=MultiModalDummyInputsBuilder,
)
class TransformersMultiModalForCausalLM(MultiModalMixin, CausalMixin, Base): ...


@MULTIMODAL_REGISTRY.register_processor(
    MultiModalProcessor,
    info=MultiModalProcessingInfo,
    dummy_inputs=MultiModalDummyInputsBuilder,
)
class TransformersMultiModalMoEForCausalLM(
    MoEMixin, MultiModalMixin, CausalMixin, Base
): ...


# Embedding models
class TransformersEmbeddingModel(EmbeddingMixin, LegacyMixin, Base): ...


class TransformersMoEEmbeddingModel(EmbeddingMixin, MoEMixin, Base): ...


@MULTIMODAL_REGISTRY.register_processor(
    MultiModalProcessor,
    info=MultiModalProcessingInfo,
    dummy_inputs=MultiModalDummyInputsBuilder,
)
class TransformersMultiModalEmbeddingModel(EmbeddingMixin, MultiModalMixin, Base): ...


# Sequence classification models
class TransformersForSequenceClassification(
    SequenceClassificationMixin, LegacyMixin, Base
): ...


class TransformersMoEForSequenceClassification(
    SequenceClassificationMixin, MoEMixin, Base
): ...


@MULTIMODAL_REGISTRY.register_processor(
    MultiModalProcessor,
    info=MultiModalProcessingInfo,
    dummy_inputs=MultiModalDummyInputsBuilder,
)
class TransformersMultiModalForSequenceClassification(
    SequenceClassificationMixin, MultiModalMixin, Base
): ...


def __getattr__(name: str):
    """Handle imports of non-existent classes with a helpful error message."""
    if name not in globals():
        raise AttributeError(
            "The Transformers modeling backend does not currently have a class to "
            f"handle the requested model type: {name}. Please open an issue at "
            "https://github.com/vllm-project/vllm/issues/new"
        )
    return globals()[name]
