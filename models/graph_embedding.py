import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
        Graph Attention Layer. Inspired by https://arxiv.org/abs/1710.10903

        Args:
            n_nodes (int): Number of nodes in the graph.
            in_features (int): Number of features per node.
            out_features (int): Number of output features per node.
            dropout_prob (float): Dropout probability for node dropout.
            alpha (float): Negative slope used in the LeakyReLU activation function.
            concat (bool): Flag indicating whether to use the final ELU activation function.
            is_training (bool): Flag indicating whether the model is in training mode.
    """

    def __init__(self, n_nodes, in_features, out_features, dropout_prob, alpha, concat=True, is_training: bool = True):
        super(GraphAttentionLayer, self).__init__()

        self.n_nodes = n_nodes
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_prob = dropout_prob
        self.alpha = alpha
        self.concat = concat
        self.is_training = is_training

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.sigmoid = nn.Sigmoid()

        # learnable parameters
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        """
            Forward pass for the Graph Attention Layer.

            Args:
                h (torch.Tensor): Node features of shape (n_batch, n_nodes, n_features).
                adj (torch.Tensor): Adjacency matrix of shape (n_nodes, n_nodes).

            Returns:
                torch.Tensor: New node embeddings of shape (n_batch, n_nodes, out_features).
        """

        hW = torch.matmul(h, self.W)  # xW shape -> (n_batch, n_nodes, out_features)

        hW_1 = torch.matmul(hW, self.a[:self.out_features, :])  # shape -> (n_batch, n_nodes, 1)
        hW_2 = torch.matmul(hW, self.a[self.out_features:, :])  # shape -> (n_batch, n_nodes, 1)
        e = hW_1 + torch.transpose(input=hW_2, dim0=2, dim1=1)  # (broadcast add) -> (n_batch, n_nodes, n_nodes)
        e = self.leakyrelu(e)
        e = torch.softmax(e, dim=2)
        e = torch.dropout(input=e, p=self.dropout_prob, train=self.is_training)

        zero_vec = -9e15 * torch.ones_like(e)  # shape -> (n_batch, n_nodes, n_nodes)
        adj = adj.to(e.device)
        attention = torch.where(adj > 0, e, zero_vec)  # shape -> (n_batch, n_nodes, n_nodes)

        new_h = self.sigmoid(torch.matmul(attention, hW))  # shape -> (n_batch, n_nodes, out_features)
        return new_h


class GraphAttentionEmbedding(nn.Module):
    """
    Graph Attention Embedding.

    Args:
        in_features (int): Number of input features.
        seq_len (int): Sequence length (equivalent of window size: number of samples in a window)
        patch_size (int): Patch size (number of samples in a patch)
        out_features (int): Number of output features.
        dropout_prob (float): Dropout probability for node dropout.
        alpha (float): Negative slope used in the LeakyReLU activation function.
        concat (bool): Flag indicating whether to use the final ELU activation function.
        is_training (bool): Flag indicating whether the model is in training mode.
    """

    def __init__(self, in_features, seq_len, patch_size, out_features, dropout_prob: float = 0.5, alpha: float = 0.2,
                 concat: bool = True, is_training: bool = True):
        super(GraphAttentionEmbedding, self).__init__()

        self.seq_len = seq_len
        self.patch_size = patch_size

        self.feature_adj = torch.ones((in_features, in_features))
        self.feature_gat = GraphAttentionLayer(n_nodes=in_features,
                                               in_features=seq_len,
                                               out_features=seq_len,
                                               dropout_prob=dropout_prob,
                                               alpha=alpha,
                                               concat=concat,
                                               is_training=is_training)

        self.temporal_adj = torch.ones((seq_len, seq_len))
        self.temporal_gat = GraphAttentionLayer(n_nodes=seq_len,
                                                in_features=in_features,
                                                out_features=in_features,
                                                dropout_prob=dropout_prob,
                                                alpha=alpha,
                                                concat=concat,
                                                is_training=is_training)

        assert seq_len % patch_size == 0
        self.n_patches = int(seq_len / patch_size)
        self.embedding_adj = torch.ones((self.n_patches, self.n_patches))
        self.embedding_gat = GraphAttentionLayer(n_nodes=patch_size,
                                                 in_features=patch_size*3*in_features,
                                                 out_features=out_features,
                                                 dropout_prob=dropout_prob,
                                                 alpha=alpha,
                                                 concat=concat,
                                                 is_training=is_training)

    def forward(self, x):
        """
        Forward pass for the Graph Attention Embedding.

        Args:
            x (torch.Tensor): Input tensor of shape (n_batch, n_samples, n_features).

        Returns:
            torch.Tensor: Embedded representation of shape (n_batch, n_patches, out_features).
        """

        x_feat = self.feature_gat(x.permute(0, 2, 1), self.feature_adj)  # (n_batch, in_features, seq_len)
        x_feat = x_feat.transpose(dim0=1, dim1=2)  # # (n_batch, seq_len, in_features)

        x_temp = self.temporal_gat(x, self.temporal_adj)  # (n_batch, seq_len, in_features)

        x_patched = torch.cat((x, x_feat, x_temp), dim=2).view(x.shape[0], self.n_patches, -1)
        x_embed = self.embedding_gat(x_patched, self.embedding_adj)

        return x_embed


def get_embedder(in_features,
                 seq_len,
                 patch_size,
                 out_features,
                 dropout_prob: float = 0.5,
                 alpha: float = 0.2,
                 concat: bool = True):
    """
        Factory function to create a GraphAttentionEmbedding instance.

        Args:
            in_features (int): Number of input features.
            seq_len (int): Sequence length.
            patch_size (int): Patch size.
            out_features (int): Number of output features.
            dropout_prob (float): Dropout probability for node dropout.
            alpha (float): Negative slope used in the LeakyReLU activation function.
            concat (bool): Flag indicating whether to use the final ELU activation function.

        Returns:
            GraphAttentionEmbedding: An instance of the GraphAttentionEmbedding class.
    """
    return GraphAttentionEmbedding(in_features=in_features,
                                   seq_len=seq_len,
                                   patch_size=patch_size,
                                   out_features=out_features,
                                   dropout_prob=dropout_prob,
                                   alpha=alpha,
                                   concat=concat)
