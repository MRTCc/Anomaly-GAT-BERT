import os
import sys
import torch.nn as nn

from models.graph_embedding import get_embedder
from models.transformer import get_transformer_encoder

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


# Anomaly Transformer
class AnomalyTransformer(nn.Module):
    def __init__(self, embedding, transformer_encoder, mlp_layers, d_embed, patch_size, max_seq_len):
        """
            Anomaly Transformer model for sequence data anomaly detection. Paper: https://arxiv.org/abs/2305.04468v1

            Args:
                embedding (nn.Module): Embedding layer to feed data into the Transformer encoder.
                transformer_encoder (nn.Module): Transformer encoder body.
                mlp_layers (nn.Module): MLP layers to return output data.
                d_embed (int): Embedding dimension in the Transformer encoder.
                patch_size (int): Number of data points for an embedded vector.
                max_seq_len (int): Maximum length of the sequence (window size).
        """

        super(AnomalyTransformer, self).__init__()
        self.embedding = embedding
        self.transformer_encoder = transformer_encoder
        self.mlp_layers = mlp_layers

        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.data_seq_len = patch_size * max_seq_len

    def forward(self, x):
        """
            Forward pass for the Anomaly Transformer model.

            Args:
                x (torch.Tensor): Input tensor of shape (n_batch, n_token, d_data) = (_, max_seq_len*patch_size, _).

            Returns:
                torch.Tensor: Output tensor of shape (n_batch, data_seq_len, d_embed).
        """
        n_batch = x.shape[0]

        # embedded_out = x.view(n_batch, self.max_seq_len, self.patch_size, -1).view(n_batch, self.max_seq_len, -1)
        # embedded_out = self.embedding(embedded_out)
        embedded_out = self.embedding(x)

        transformer_out = self.transformer_encoder(embedded_out)  # Encode data.
        output = self.mlp_layers(transformer_out)  # Reconstruct data.
        return output.view(n_batch, self.max_seq_len, self.patch_size, -1).view(n_batch, self.data_seq_len, -1)


# Get Anomaly Transformer.
def get_anomaly_transformer(input_d_data,
                            output_d_data,
                            patch_size,
                            d_embed=512,
                            hidden_dim_rate=4.,
                            max_seq_len=512,
                            positional_encoding=None,
                            relative_position_embedding=True,
                            transformer_n_layer=12,
                            transformer_n_head=8,
                            dropout=0.1,
                            alpha=0.2):
    """
    Factory function to create an Anomaly Transformer model.

    Args:
        input_d_data (int): Data input dimension.
        output_d_data (int): Data output dimension.
        patch_size (int): Number of data points per embedded feature.
        d_embed (int): Embedding dimension (in Transformer encoder).
        hidden_dim_rate (float): Hidden layer dimension rate to d_embed.
        max_seq_len (int): Maximum length of the sequence (window size).
        positional_encoding (str or None): Positional encoding for embedded input; None/Sinusoidal/Absolute.
        relative_position_embedding (bool): Relative position embedding option.
        transformer_n_layer (int): Number of Transformer encoder layers.
        transformer_n_head (int): Number of heads in the multi-head attention module.
        dropout (float): Dropout rate.
        alpha (float): Negative slope used in the Leaky ReLU activation function.

    Returns:
        AnomalyTransformer: An instance of the AnomalyTransformer model.
    """

    hidden_dim = int(hidden_dim_rate * d_embed)

    embedding = get_embedder(in_features=input_d_data,
                             seq_len=max_seq_len*patch_size,
                             patch_size=patch_size,
                             out_features=d_embed,
                             dropout_prob=dropout,
                             alpha=alpha)

    transformer_encoder = get_transformer_encoder(d_embed=d_embed,
                                                  positional_encoding=positional_encoding,
                                                  relative_position_embedding=relative_position_embedding,
                                                  n_layer=transformer_n_layer,
                                                  n_head=transformer_n_head,
                                                  d_ff=hidden_dim,
                                                  max_seq_len=max_seq_len,
                                                  dropout=dropout)
    mlp_layers = nn.Sequential(nn.Linear(d_embed, hidden_dim),
                               nn.GELU(),
                               nn.Linear(hidden_dim, output_d_data * patch_size))

    nn.init.xavier_uniform_(mlp_layers[0].weight)
    nn.init.zeros_(mlp_layers[0].bias)
    nn.init.xavier_uniform_(mlp_layers[2].weight)
    nn.init.zeros_(mlp_layers[2].bias)

    return AnomalyTransformer(embedding,
                              transformer_encoder,
                              mlp_layers,
                              d_embed,
                              patch_size,
                              max_seq_len)
