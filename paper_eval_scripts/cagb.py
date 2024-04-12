import torch
from torch import nn
from torch_geometric.utils import unbatch

class CrossAttentionGraphBlock (nn.Module):

    def __init__(self, num_heads, latent_size, node_feature_size, dropout=0.1):

        super().__init__()

        self.leaky_relu = nn.LeakyReLU()

        self.q_dense = nn.Linear(latent_size, latent_size)
        self.k_dense = nn.Linear(latent_size, latent_size)
        self.v_dense = nn.Linear(node_feature_size, latent_size)

        self.attention = nn.MultiheadAttention(latent_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(latent_size)

        self.dense1 = nn.Linear(latent_size, latent_size)
        self.ln2 = nn.LayerNorm(latent_size)

    def forward(self, x, conditioning_graph_nodes, conditioning_graph_batch):

        unbatched_sequences = unbatch(conditioning_graph_nodes, conditioning_graph_batch)

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
        attention_mask = ~torch.Tensor(attention_masks, device=batch_nodes.device).bool()

        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(batch_nodes)

        attn_output, _ = self.attention(q, k, v, key_padding_mask=attention_mask)

        x = self.ln1(attn_output + x)

        x = self.leaky_relu(self.dense1(x)) + x
        x = self.ln2(x)

        return x