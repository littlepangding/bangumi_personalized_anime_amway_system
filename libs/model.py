import torch
import torch.nn as nn
import json
from libs import bases

NUM_EMBED = 5 + 6 + 1
NUM_LABEL = 6  # 5 or 6
HASH_SIZE = 21000
NUM_GT = 5
NUM_LT = 6

NUM_EMBED = NUM_LT + NUM_GT + 1


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

            if i < len(layer_dims) - 1:
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.SiLU())  # Swish activation
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PaaSModel(nn.Module):
    def __init__(self, embed_dim, over="", dropout=0.0, share_embed=False, mode="max"):
        super().__init__()
        #
        # sparse emb for user rating history
        self.embed_dim = embed_dim
        self.share_embed = share_embed
        if not share_embed:
            self.lt_emb = nn.ModuleList(
                nn.EmbeddingBag(HASH_SIZE, embed_dim, mode=mode) for i in range(NUM_LT)
            )
            self.gt_emb = nn.ModuleList(
                nn.EmbeddingBag(HASH_SIZE, embed_dim, mode=mode) for i in range(NUM_GT)
            )
        else:
            self.lt_emb = nn.EmbeddingBag(HASH_SIZE, embed_dim, mode=mode)
            self.gt_emb = nn.EmbeddingBag(HASH_SIZE, embed_dim, mode=mode)
        # show id
        self.show_emb = nn.Embedding(HASH_SIZE, embed_dim)

        # prepare dot mask
        self.dot_mask = torch.ones(NUM_EMBED, NUM_EMBED).triu() == 1

        # prepare over arch
        self.over_input_size = self._get_over_input_size(embed_dim)
        self.over = [int(v) for v in over.split("-") if v]

        if self.over:
            self.over_layer_dims = [self.over_input_size] + self.over
            self.over_mlp = MLP(
                layer_dims=self.over_layer_dims,
                dropout=dropout,
            )

        if self.over:
            self.over_output_size = self.over[-1]
        else:
            self.over_output_size = self.over_input_size
        self.linears = nn.ModuleList(
            nn.Linear(self.over_output_size, 1) for i in range(NUM_LABEL)
        )  # simple binary classifier

    def _get_over_input_size(self, embed_dim):
        num_emb = NUM_LT + NUM_GT + 1
        return num_emb * embed_dim + num_emb * (num_emb + 1) // 2

    def _get_dot(
        self,
        embeds,
    ):
        # embed b * d * n
        outs = torch.matmul(
            torch.transpose(embeds, 1, 2),  # b * n * d
            embeds,  # b * d * n
        )  # b * n * n
        return outs[:, self.dot_mask] / self.embed_dim

    def forward(
        self,
        lt_input_and_offsets,
        gt_input_and_offsets,
        show_ids,
    ):
        # sparse
        lt_val = [
            (
                self.lt_emb[i](
                    lt_input_and_offsets[i][0],
                    lt_input_and_offsets[i][1],
                )
                if not self.share_embed
                else self.lt_emb(
                    lt_input_and_offsets[i][0],
                    lt_input_and_offsets[i][1],
                )
            )
            for i in range(NUM_LT)
        ]
        gt_val = [
            (
                self.gt_emb[i](
                    gt_input_and_offsets[i][0],
                    gt_input_and_offsets[i][1],
                )
                if not self.share_embed
                else self.gt_emb(
                    gt_input_and_offsets[i][0],
                    gt_input_and_offsets[i][1],
                )
            )
            for i in range(NUM_GT)
        ]
        show_val = self.show_emb(show_ids)

        # sparse interaction
        embeds = torch.stack(lt_val + gt_val + [show_val], dim=2)
        dots = self._get_dot(embeds)
        cats = torch.cat(lt_val + gt_val + [show_val], dim=1)

        over = torch.cat([dots, cats], dim=1)
        # over
        if self.over:
            over = self.over_mlp(over)

        # final pred
        outs = [self.linears[i](over).squeeze(1) for i in range(NUM_LABEL)]
        return outs


def load_model(config_path, ckp_path, device):
    print(f"Loading model from ckp_path: {ckp_path}")
    config = bases.get_config(config_path)
    print(f"config: {json.dumps(config, indent=4)}")
    model = torch.load(ckp_path, weights_only=False).to(device)
    model.eval()
    return model
