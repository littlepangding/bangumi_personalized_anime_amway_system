import torch
import torch.nn as nn

NUM_EMBED = 5 + 6 + 1
NUM_LABEL = 6  # 5 or 6
HASH_SIZE = 21000
NUM_GT = 5
NUM_LT = 6


class MLP(nn.Module):
    def __init__(self, layer_dims, dropout=0.1):
        """
        Args:
            layer_dims (List[int]): [input_dim, hidden1, hidden2, ..., output_dim]
            dropout (float): Dropout probability (applied after activation on all hidden layers)
        """
        super().__init__()
        layers = []

        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]

            layers.append(nn.Linear(in_dim, out_dim))

            if i < len(layer_dims) - 2:  # skip BN/activation/dropout on final layer
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.SiLU())  # Swish activation
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PaaSModel(nn.Module):
    def __init__(
        self,
        embed_dim,
    ):
        super().__init__()
        # sparse emb for user rating history
        self.lt_emb = nn.ModuleList(
            nn.EmbeddingBag(HASH_SIZE, embed_dim, mode="max") for i in range(NUM_LT)
        )
        self.gt_emb = nn.ModuleList(
            nn.EmbeddingBag(HASH_SIZE, embed_dim, mode="max") for i in range(NUM_GT)
        )
        # show id
        self.show_emb = nn.Embedding(HASH_SIZE, embed_dim)

        self.linears = nn.ModuleList(
            nn.Linear((NUM_LT + NUM_GT + 1) * embed_dim, 5) for i in range(NUM_LABEL)
        )  # simple binary classifier

    def forward(
        self,
        lt_input_and_offsets,
        gt_input_and_offsets,
        show_ids,
    ):
        lt_val = [
            self.lt_emb[i](
                lt_input_and_offsets[i][0],
                lt_input_and_offsets[i][1],
            )
            for i in range(NUM_LT)
        ]
        # print(f"lt_val[0].shape: {lt_val[0].shape}")
        gt_val = [
            self.gt_emb[i](
                gt_input_and_offsets[i][0],
                gt_input_and_offsets[i][1],
            )
            for i in range(NUM_GT)
        ]
        show_val = self.show_emb(show_ids)
        # print(f"show_val.shape: {show_val.shape}")
        val = torch.cat(lt_val + gt_val + [show_val], dim=1)
        # print(val.shape)
        outs = [self.linears[i](val) for i in range(NUM_LABEL)]
        return outs
