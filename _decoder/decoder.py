#pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments, too-many-branches
"""decoder part of the transformer
"""
# import torch
from torch import nn
from _encoder.encoder import LayerNormalization, PositionwiseFeedForward,\
                    MultiHeadAttentionAPE

class DecoderLayerAPE(nn.Module):
    """Decoder layer
    """
    def __init__(self, device, d_model, seq_length, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.self_attention = MultiHeadAttentionAPE(device, d_model, seq_length, num_heads)
        self.layer_norm1 = LayerNormalization(device, parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(device, d_model, ffn_hidden, drop_prob)
        self.layer_norm3 = LayerNormalization(device, parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)


    def forward(self, y_val, decoder_mask, key_cache=None, value_cache=None):
        """Forward prop
        """
        _y = y_val.clone()
        y_val, key_cache, value_cache  = self.self_attention(y_val, mask=decoder_mask)
        y_val = self.dropout1(y_val)
        y_val = self.layer_norm1(y_val + _y)

        # _x = x_val.clone()
        _y = y_val.clone()
        y_val = self.ffn(y_val)
        y_val = self.dropout3(y_val)
        y_val = self.layer_norm3(y_val + _y)

        return y_val, key_cache, value_cache

class SequentialDecoder(nn.Module):
    """Sequential decoder"""
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, yseq, mask, key_caches=None, value_caches=None):
        """Forward pass"""
        if key_caches is None:
            key_caches = [None] * len(self.layers)
        if value_caches is None:
            value_caches = [None] * len(self.layers)

        new_key_caches = []
        new_value_caches = []

        for idx, layer in enumerate(self.layers):
            yseq, key_cache, value_cache = layer(
                yseq, mask, key_caches[idx], value_caches[idx]
            )
            new_key_caches.append(key_cache)
            new_value_caches.append(value_cache)

        return yseq, new_key_caches, new_value_caches

class DecoderAPE(nn.Module):
    """Main decoder class
    """
    def __init__(self, device, d_model, d_vocab, ffn_hidden,
                 seq_length, num_heads, drop_prob, num_layers) -> None:
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayerAPE(device, d_model,
                                    seq_length,ffn_hidden,num_heads,drop_prob)\
                                      for _ in range(num_layers)])
        self.linear_layer = nn.Linear(d_model, d_vocab)

    def forward(self, y_val, decoder_mask, key_caches=None, value_caches=None):
        """Forward layer
        """
        x_val, key_caches, value_caches = self.layers(y_val,
                                    decoder_mask, key_caches, value_caches)
        x_val = self.linear_layer(x_val)
        return x_val, key_caches, value_caches
