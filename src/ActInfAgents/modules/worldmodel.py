import copy

import torch
import torch.nn.functional as F
from ActInfAgents.modules.lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from torch import nn


class WorldModel(nn.Module):
    """
    :param recurrence_type:
        None                - No recurrence, adds no extra layers
        lstm                - (Depreciated). Singular LSTM
        multi_layer_lstm    - Multi-layer LSTM. Uses n_recurrence_layers to determine number of consecututive LSTMs
            Does NOT support ragged batching
        multi_masked_lstm   - Multi-layer LSTM that supports ragged batching via the first vector. This model is slower
            Uses n_recurrence_layers to determine number of consecututive LSTMs
        transformer         - Dense transformer
    :param init_norm_kwargs: kwargs for all FanInInitReLULayers.
    """

    def __init__(
        self,
        recurrence_type="lstm",
        # impala_width=1,
        # impala_chans=(16, 32, 32),
        # obs_processing_width=256,
        hidsize=512,
        single_output=False,  # True if we don't need separate outputs for action/value outputs
        # img_shape=None,
        # scale_input_img=True,
        # only_img_input=False,
        init_norm_kwargs={},
        # impala_kwargs={},
        # Unused argument assumed by forc.
        # input_shape=None,  # pylint: disable=unused-argument
        active_reward_monitors=None,
        # img_statistics=None,
        # first_conv_norm=False,
        # diff_mlp_embedding=False,
        attention_mask_style="clipped_causal",
        attention_heads=8,
        attention_memory_size=2048,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        n_recurrence_layers=1,
        recurrence_is_residual=True,
        timesteps=128,
        use_pre_lstm_ln=True,  # Not needed for transformer
        # **unused_kwargs,
    ):
        """
        Hacked OpenAI VPT model for its hierarchical structure
        and potentially good off-the-shelf foundational model perf.

        Also doesn't rely on actions, which is good for our world model?
        TODO: DOC STRING TS
        """
        super().__init__()
        assert recurrence_type in [
            "multi_layer_lstm",
            "multi_layer_bilstm",
            "multi_masked_lstm",
            "transformer",
            "none",
        ]

        active_reward_monitors = active_reward_monitors or {}

        self.single_output = single_output

        # chans = tuple(int(impala_width * c) for c in impala_chans)
        self.hidsize = hidsize

        # Dense init kwargs replaces batchnorm/groupnorm with layernorm
        self.init_norm_kwargs = init_norm_kwargs
        self.dense_init_norm_kwargs = copy.deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True

        # # Setup inputs
        # self.img_preprocess = ImgPreprocessing(img_statistics=img_statistics, scale_img=scale_input_img)
        # self.img_process = ImgObsProcess(
        #     cnn_outsize=256,
        #     output_size=hidsize,
        #     inshape=img_shape,
        #     chans=chans,
        #     nblock=2,
        #     dense_init_norm_kwargs=self.dense_init_norm_kwargs,
        #     init_norm_kwargs=init_norm_kwargs,
        #     first_conv_norm=first_conv_norm,
        #     **impala_kwargs,
        # )

        self.pre_lstm_ln = nn.LayerNorm(hidsize) if use_pre_lstm_ln else None
        self.diff_obs_process = None

        self.recurrence_type = recurrence_type

        self.recurrent_layer = None
        self.recurrent_layer = ResidualRecurrentBlocks(
            hidsize=hidsize,
            timesteps=timesteps,
            recurrence_type=recurrence_type,
            is_residual=recurrence_is_residual,
            use_pointwise_layer=use_pointwise_layer,
            pointwise_ratio=pointwise_ratio,
            pointwise_use_activation=pointwise_use_activation,
            attention_mask_style=attention_mask_style,
            attention_heads=attention_heads,
            attention_memory_size=attention_memory_size,
            n_block=n_recurrence_layers,
        )

        self.lastlayer = FanInInitReLULayer(hidsize, hidsize, layer_type="linear", **self.dense_init_norm_kwargs)
        self.final_ln = torch.nn.LayerNorm(hidsize)

        self.state_mask = None  # attribute to hold state mask

    def output_latent_size(self):
        return self.hidsize

    def forward(self, x, state_mask, xf_state, context):
        """
        Runs forward for WorldModel. Takes in previous state and current observation.
        """
        first = context["first"]
        if self.pre_lstm_ln is not None:
            x = self.pre_lstm_ln(x)

        if self.recurrent_layer is not None:
            x, state_mask, xf_state = self.recurrent_layer(x, first, state_mask, xf_state)

        x = F.relu(x, inplace=False)

        x = self.lastlayer(x)
        x = self.final_ln(x)
        pi_latent = vf_latent = x
        if self.single_output:
            return pi_latent, state_mask, xf_state
        return (pi_latent, vf_latent), state_mask, xf_state

    def initial_state(self, batch_size):
        if self.recurrent_layer:
            return self.recurrent_layer.initial_state(batch_size)
        else:
            return None