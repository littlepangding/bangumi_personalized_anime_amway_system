import torch
import torch.nn as nn


class PaaSModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        hash_size,
    ):
        super().__init__()
        # sparse emb for user rating history
        self.lt_emb = nn.ModuleList(
            nn.EmbeddingBag(hash_size, embed_dim, mode="max") for i in range(6)
        )
        self.gt_emb = nn.ModuleList(
            nn.EmbeddingBag(hash_size, embed_dim, mode="max") for i in range(5)
        )
        # show id
        self.show_emb = nn.Embedding(hash_size, embed_dim)

        self.linears = nn.ModuleList(
            nn.Linear((6 + 5 + 1) * embed_dim, 5) for i in range(5)
        )  # simple binary classifier

    def forward(
        self,
        lt_input_and_offsets,
        gt_input_and_offsets,
        show_ids,
        device,
    ):
        lt_val = [
            self.lt_emb[i](
                lt_input_and_offsets[i][0].to(device),
                lt_input_and_offsets[i][1].to(device),
            )
            for i in range(6)
        ]
        # print(f"lt_val[0].shape: {lt_val[0].shape}")
        gt_val = [
            self.gt_emb[i](
                gt_input_and_offsets[i][0].to(device),
                gt_input_and_offsets[i][1].to(device),
            )
            for i in range(5)
        ]
        show_val = self.show_emb(show_ids.to(device))
        # print(f"show_val.shape: {show_val.shape}")
        val = torch.cat(lt_val + gt_val + [show_val], dim=1)
        # print(val.shape)
        outs = [self.linears[i](val) for i in range(5)]
        return outs
