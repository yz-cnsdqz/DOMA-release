# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Warp fields."""
from typing import Any, Iterable, Optional, Dict

from flax import linen as nn
import jax
import jax.numpy as jnp

from nerfies import glo
from nerfies import model_utils
from nerfies import modules
from nerfies import rigid_body as rigid
from nerfies import types


class SE3Field(nn.Module):
  """Network that predicts warps as an SE(3) field.

  Attributes:
    points_encoder: the positional encoder for the points.
    metadata_encoder: an encoder for metadata.
    alpha: the alpha for the positional encoding.
    skips: the index of the layers with skip connections.
    depth: the depth of the network excluding the logit layer.
    hidden_channels: the width of the network hidden layers.
    activation: the activation for each layer.
    metadata_encoded: whether the metadata parameter is pre-encoded or not.
    hidden_initializer: the initializer for the hidden layers.
    output_initializer: the initializer for the last logit layer.
  """
  num_freqs: int
  num_embeddings: int
  num_embedding_features: int
  min_freq_log2: int = 0
  max_freq_log2: Optional[int] = None
  use_identity_map: bool = True

  activation: types.Activation = nn.relu
  skips: Iterable[int] = (4,)
  trunk_depth: int = 6
  trunk_width: int = 128
  rotation_depth: int = 0
  rotation_width: int = 128
  pivot_depth: int = 0
  pivot_width: int = 128
  translation_depth: int = 0
  translation_width: int = 128
  metadata_encoder_type: str = 'glo'
  metadata_encoder_num_freqs: int = 1

  default_init: types.Initializer = nn.initializers.xavier_uniform()
  rotation_init: types.Initializer = nn.initializers.uniform(scale=1e-4)
  pivot_init: types.Initializer = nn.initializers.uniform(scale=1e-4)
  translation_init: types.Initializer = nn.initializers.uniform(scale=1e-4)

  use_pivot: bool = False
  use_translation: bool = False

  def setup(self):
    self.points_encoder = modules.AnnealedSinusoidalEncoder(
        num_freqs=self.num_freqs,
        min_freq_log2=self.min_freq_log2,
        max_freq_log2=self.max_freq_log2,
        use_identity=self.use_identity_map)

    if self.metadata_encoder_type == 'glo':
      self.metadata_encoder = glo.GloEncoder(
          num_embeddings=self.num_embeddings,
          features=self.num_embedding_features)
    elif self.metadata_encoder_type == 'time':
      self.metadata_encoder = modules.TimeEncoder(
          num_freqs=self.metadata_encoder_num_freqs,
          features=self.num_embedding_features)
    else:
      raise ValueError(
          f'Unknown metadata encoder type {self.metadata_encoder_type}')

    self.trunk = modules.MLP(
        depth=self.trunk_depth,
        width=self.trunk_width,
        hidden_activation=self.activation,
        hidden_init=self.default_init,
        skips=self.skips)

    branches = {
        'w':
            modules.MLP(
                depth=self.rotation_depth,
                width=self.rotation_width,
                hidden_activation=self.activation,
                hidden_init=self.default_init,
                output_init=self.rotation_init,
                output_channels=3),
        'v':
            modules.MLP(
                depth=self.pivot_depth,
                width=self.pivot_width,
                hidden_activation=self.activation,
                hidden_init=self.default_init,
                output_init=self.pivot_init,
                output_channels=3),
    }
    if self.use_pivot:
      branches['p'] = modules.MLP(
          depth=self.pivot_depth,
          width=self.pivot_width,
          hidden_activation=self.activation,
          hidden_init=self.default_init,
          output_init=self.pivot_init,
          output_channels=3)
    if self.use_translation:
      branches['t'] = modules.MLP(
          depth=self.translation_depth,
          width=self.translation_width,
          hidden_activation=self.activation,
          hidden_init=self.default_init,
          output_init=self.translation_init,
          output_channels=3)
    # Note that this must be done this way instead of using mutable operations.
    # See https://github.com/google/flax/issues/524.
    self.branches = branches

  def encode_metadata(self,
                      metadata: jnp.ndarray,
                      time_alpha: Optional[float] = None):
    if self.metadata_encoder_type == 'time':
      metadata_embed = self.metadata_encoder(metadata, time_alpha)
    elif self.metadata_encoder_type == 'glo':
      metadata_embed = self.metadata_encoder(metadata)
    else:
      raise RuntimeError(
          f'Unknown metadata encoder type {self.metadata_encoder_type}')

    return metadata_embed

  def warp(self,
           points: jnp.ndarray,
           metadata_embed: jnp.ndarray,
           extra: Dict[str, Any]):
    points_embed = self.points_encoder(points, alpha=extra.get('alpha'))
    inputs = jnp.concatenate([points_embed, metadata_embed], axis=-1)
    trunk_output = self.trunk(inputs)

    w = self.branches['w'](trunk_output)
    v = self.branches['v'](trunk_output)
    theta = jnp.linalg.norm(w, axis=-1)
    w = w / theta[..., jnp.newaxis]
    v = v / theta[..., jnp.newaxis]
    screw_axis = jnp.concatenate([w, v], axis=-1)
    transform = rigid.exp_se3(screw_axis, theta)

    warped_points = points
    if self.use_pivot:
      pivot = self.branches['p'](trunk_output)
      warped_points = warped_points + pivot

    warped_points = rigid.from_homogenous(
        transform @ rigid.to_homogenous(warped_points))

    if self.use_pivot:
      warped_points = warped_points - pivot

    if self.use_translation:
      t = self.branches['t'](trunk_output)
      warped_points = warped_points + t

    return warped_points

  def __call__(self,
               points: jnp.ndarray,
               metadata: jnp.ndarray,
               extra: Dict[str, Any],
               return_jacobian: bool = False,
               metadata_encoded: bool = False):
    """Warp the given points using a warp field.

    Args:
      points: the points to warp.
      metadata: metadata indices if metadata_encoded is False else pre-encoded
        metadata.
      extra: A dictionary containing
        'alpha': the alpha value for the positional encoding.
        'time_alpha': the alpha value for the time positional encoding
          (if applicable).
      return_jacobian: if True compute and return the Jacobian of the warp.
      metadata_encoded: if True assumes the metadata is already encoded.

    Returns:
      The warped points and the Jacobian of the warp if `return_jacobian` is
        True.
    """
    if metadata_encoded:
      metadata_embed = metadata
    else:
      metadata_embed = self.encode_metadata(metadata, extra.get('time_alpha'))

    out = {'warped_points': self.warp(points, metadata_embed, extra)}

    if return_jacobian:
      jac_fn = jax.jacfwd(self.warp, argnums=0)
      out['jacobian'] = jac_fn(points, metadata_embed, extra)

    return out
