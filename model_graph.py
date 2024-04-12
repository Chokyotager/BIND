import torch
from torch import nn
from torch_geometric.utils import unbatch

from torch_geometric.data import Data, Batch

import torch_geometric.nn as gnn


class CrossAttentionGraphBlock (nn.Module):

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


class PoolWrapper (nn.Module):

    # To allow batching - in which each graph
    # Is treated independently

    def __init__ (self, pool):

        super().__init__()

        self.pool = pool

    def forward (self, x, batch):

        unbatched_sequences = unbatch(x, batch)

        output = list()

        for unbatched in unbatched_sequences:

            output.append(self.pool(unbatched))

        return torch.cat(output, dim=0)


class ConditionModel (nn.Module):

    def __init__ (self):

        super().__init__()

        self.leaky_relu = nn.LeakyReLU()

        self.x_embed = nn.Linear(15, 16)
        self.x_embed_ln = gnn.LayerNorm(16)

        self.e_embed = nn.Linear(5, 2)
        self.e_embed_ln = nn.LayerNorm(2)

        self.conv1 = gnn.GATv2Conv(16, 64, edge_dim=2, negative_slope=0.01)
        self.ln1 = gnn.LayerNorm(64)

        self.crossattention1 = CrossAttentionGraphBlock(num_heads=16, node_feature_size=64, latent_size=1280)
        
        self.conv2 = gnn.GATv2Conv(64, 64, edge_dim=2, negative_slope=0.01)
        self.ln2 = gnn.LayerNorm(64)

        self.crossattention2 = CrossAttentionGraphBlock(num_heads=16, node_feature_size=64, latent_size=1280)

        self.conv3 = gnn.GATv2Conv(64, 64, edge_dim=2, negative_slope=0.01)
        self.ln3 = gnn.LayerNorm(64)

        self.crossattention3 = CrossAttentionGraphBlock(num_heads=16, node_feature_size=64, latent_size=1280)

        self.conv4 = gnn.GATv2Conv(64, 64, edge_dim=2, negative_slope=0.01)
        self.ln4 = gnn.LayerNorm(64)

        self.crossattention4 = CrossAttentionGraphBlock(num_heads=16, node_feature_size=64, latent_size=1280)

        self.conv5 = gnn.GATv2Conv(64, 64, edge_dim=2, negative_slope=0.01)
        self.ln5 = gnn.LayerNorm(64)

        # Global pool
        self.pool = PoolWrapper(gnn.LCMAggregation(64, 1024))

        self.dense1 = nn.Linear(1024, 1024)
        self.lnf1 = nn.LayerNorm(1024)

        self.dense2 = nn.Linear(1024, 1024)
        self.lnf2 = nn.LayerNorm(1024)

        self.ki_head = nn.Linear(1024, 1)
        self.ic50_head = nn.Linear(1024, 1)
        self.kd_head = nn.Linear(1024, 1)
        self.ec50_head = nn.Linear(1024, 1)

        self.temperature = torch.nn.Parameter(torch.tensor(0.07), requires_grad=True)

        self.classifier_head = nn.Linear(1024, 1)

    def forward (self, graph, hidden_states, attention_mask, return_attentions=False):

        padding_mask = ~attention_mask.bool()
        hidden_states = [x.detach().float() for x in hidden_states]

        graph = graph.sort()

        x, a, e = graph.x, graph.edge_index, graph.edge_features

        x = self.x_embed(x)
        x = self.leaky_relu(x)
        x = self.x_embed_ln(x, graph.batch)

        e = self.e_embed(e)
        e = self.leaky_relu(e)
        e = self.e_embed_ln(e)

        x = self.conv1(x, a, e)
        x = self.leaky_relu(x)
        x = self.ln1(x, graph.batch)

        x, av_attn1 = self.crossattention1(x, graph.batch, hidden_states[0], padding_mask)

        x = self.conv2(x, a, e)
        x = self.leaky_relu(x)
        x = self.ln2(x, graph.batch)

        x, av_attn2 = self.crossattention2(x, graph.batch, hidden_states[10], padding_mask)

        x = self.conv3(x, a, e)
        x = self.leaky_relu(x)
        x = self.ln3(x, graph.batch)

        x, av_attn3 = self.crossattention3(x, graph.batch, hidden_states[20], padding_mask)

        x = self.conv4(x, a, e)
        x = self.leaky_relu(x)
        x = self.ln4(x, graph.batch)

        x, av_attn4 = self.crossattention4(x, graph.batch, hidden_states[30], padding_mask)

        x = self.conv5(x, a, e)
        x = self.leaky_relu(x)
        x = self.ln5(x, graph.batch)

        x = self.pool(x, graph.batch)

        x = self.dense1(x)
        x = self.leaky_relu(x)
        x = self.lnf1(x)

        x = self.dense2(x)
        x = self.leaky_relu(x)
        x = self.lnf2(x)

        ki = self.ki_head(x)
        ic50 = self.ic50_head(x)
        kd = self.kd_head(x)
        ec50 = self.ec50_head(x)
        
        classification = self.classifier_head(x) * torch.exp(self.temperature)

        if return_attentions:
            return ki, ic50, kd, ec50, classification, [av_attn1, av_attn2, av_attn3, av_attn4]
        
        else:
            return ki, ic50, kd, ec50, classification