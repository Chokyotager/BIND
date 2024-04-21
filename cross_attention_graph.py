import torch
from torch import nn
from torch_geometric.utils import unbatch

from torch_geometric.data import Data, Batch


class CrossAttentionGraphModule (nn.Module):

    def __init__(self, num_heads, node_feature_size, latent_size, dropout=0.1):

        super().__init__()

        self.leaky_relu = nn.LeakyReLU()

        self.q_dense = nn.Linear(node_feature_size, node_feature_size)
        self.k_dense = nn.Linear(latent_size, node_feature_size)
        self.v_dense = nn.Linear(latent_size, node_feature_size)

        self.attention = nn.MultiheadAttention(node_feature_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.ln1 = nn.LayerNorm(node_feature_size)

        self.dense1 = nn.Linear(node_feature_size, node_feature_size)
        self.ln2 = nn.LayerNorm(node_feature_size)

    def forward(self, graph_nodes, graph_batch, conditioning_vector, conditioning_attention_mask):

        unbatched_sequences = unbatch(graph_nodes, graph_batch)

        largest_batch_nodes = max([x.shape[0] for x in unbatched_sequences])
        feature_size = unbatched_sequences[0].shape[1]

        all_padded_batches = list()
        attention_masks = list()

        for current_batch in unbatched_sequences:

            number_pad_notes = largest_batch_nodes - current_batch.shape[0]

            pad_zeros = torch.zeros([number_pad_notes, feature_size]).to(current_batch.device)

            padded_batch = torch.cat([current_batch, pad_zeros], dim=0)

            all_padded_batches.append(padded_batch)

            attention_mask = [1] * current_batch.shape[0] + [0] * number_pad_notes

            attention_masks.append(attention_mask)

        batch_nodes = torch.stack(all_padded_batches, dim=0)
        node_mask = torch.Tensor(attention_masks).bool().to(batch_nodes.device)

        q = self.q_dense(batch_nodes)
        k = self.k_dense(conditioning_vector)
        v = self.v_dense(conditioning_vector)

        attn_output, av_attn = self.attention(q, k, v, key_padding_mask=conditioning_attention_mask, average_attn_weights=False)

        av_attn = torch.mean(av_attn.detach(), dim=1)

        x = self.ln1(attn_output + batch_nodes)

        x = self.leaky_relu(self.dense1(x)) + x
        x = self.ln2(x)

        # Node reconstruction
        graphs = list()

        for batch_i in range(x.shape[0]):

            current_nodes = x[batch_i]
            current_mask = node_mask[batch_i]

            current_nodes = current_nodes[current_mask]

            current_graph = Data(x=current_nodes)

            graphs.append(current_graph)

        new_batch = Batch.from_data_list(graphs)

        return new_batch.x, av_attn