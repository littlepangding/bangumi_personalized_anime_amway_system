import argparse
import os
import json
from libs import bases, preproc
from libs.model import PaaSModel, NUM_LABEL
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from libs.exp import sum_epoch_losses, train, eval

MAX_ITER = None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path")
    parser.add_argument("data")
    parser.add_argument("--cuda", default="cuda")
    args = parser.parse_args()

    config_base = bases.get_config(os.path.join(args.exp_path, "config.json"))
    print(f"exp config:\n{json.dumps(config_base, indent=4)}")
    (
        out_user_train,
        out_user_val,
        out_user_eval,
        in_user_train,
        in_user_val,
        in_user_eval,
        new_to_old,
        old_to_new,
        max_id,
    ) = preproc.load_split_data(args.data)

    # data prep
    dataset = preproc.UserShowRatingDataset(out_user_train)
    loader = DataLoader(
        dataset, batch_size=2048, collate_fn=preproc.custom_collate_fn, shuffle=True
    )

    dataset_val = preproc.UserShowRatingDataset(out_user_val)
    loader_val = DataLoader(
        dataset_val, batch_size=2048, collate_fn=preproc.custom_collate_fn, shuffle=True
    )

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    model = PaaSModel(
        embed_dim=config_base["model"]["sparse"]["embed_dim"],
        over=config_base["model"]["over"],
        dropout=config_base["train"]["dropout"],
        share_embed=config_base["model"]["sparse"]["share_embed"],
        mode=config_base["model"]["sparse"]["pooling"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config_base["train"]["lr"],
        weight_decay=config_base["train"]["weight_decay_dense"],
    )
    losses = [nn.BCEWithLogitsLoss(reduction="none") for i in range(NUM_LABEL)]

    prev_val_loss = None
    for epoch in range(config_base["train"]["max_epoch"]):
        print(f"Starting epoch: {epoch}, training started")
        # train
        epoch_loss = train(
            epoch, loader, model, losses, optimizer, device, max_iter=MAX_ITER
        )
        sum_epoch_losses("Train", epoch, epoch_loss)

        # validate
        print(f"End of training for epoch: {epoch}, validation started")
        epoch_loss_val = eval(loader_val, model, losses, device, max_iter=MAX_ITER)
        overall_val_loss = sum_epoch_losses("Val(Out User)", epoch, epoch_loss_val)
        if prev_val_loss is not None and overall_val_loss > prev_val_loss:
            break
        prev_val_loss = overall_val_loss
        torch.save(model, os.path.join(args.exp_path, f"model_{epoch%3:02d}.ckp"))
