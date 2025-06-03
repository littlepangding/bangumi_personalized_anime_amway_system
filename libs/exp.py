from libs import preproc
from libs.model import NUM_LABEL


def get_sum_loss(losses, preds, weights, labels):
    loss_vals = [
        (losses[i](preds[i], labels[i]) * weights[i]).mean() for i in range(NUM_LABEL)
    ]
    total_loss = 0
    for i in range(NUM_LABEL):
        total_loss += loss_vals[i]
    return total_loss, loss_vals


def infer_and_get_losses(
    device,
    model,
    losses,
    lt_input_and_offsets,
    gt_input_and_offsets,
    show_ids,
    labels,
    weights,
):
    (
        lt_input_and_offsets,
        gt_input_and_offsets,
        show_ids,
        labels,
        weights,
    ) = preproc.to_device(
        device,
        lt_input_and_offsets,
        gt_input_and_offsets,
        show_ids,
        labels,
        weights,
    )
    preds = model(lt_input_and_offsets, gt_input_and_offsets, show_ids)
    (
        total_loss,
        loss_vals,
    ) = get_sum_loss(losses, preds, weights, labels)
    return (
        total_loss,
        loss_vals,
    )


def eval(loader, model, losses, device, max_iter=None):
    model.eval()
    epoch_loss = [[] for i in range(NUM_LABEL)]

    for iter, (
        lt_input_and_offsets,
        gt_input_and_offsets,
        show_ids,
        labels,
        weights,
    ) in enumerate(loader):
        (
            _,
            loss_vals,
        ) = infer_and_get_losses(
            device,
            model,
            losses,
            lt_input_and_offsets,
            gt_input_and_offsets,
            show_ids,
            labels,
            weights,
        )
        for i in range(NUM_LABEL):
            epoch_loss[i].append(loss_vals[i].item())
        if max_iter is not None and iter == max_iter:
            break
    return epoch_loss


def train(epoch, loader, model, losses, optimizer, device, max_iter=None):
    epoch_loss = [[] for i in range(NUM_LABEL)]
    model.train()
    for iter, (
        lt_input_and_offsets,
        gt_input_and_offsets,
        show_ids,
        labels,
        weights,
    ) in enumerate(loader):
        (
            total_loss,
            loss_vals,
        ) = infer_and_get_losses(
            device,
            model,
            losses,
            lt_input_and_offsets,
            gt_input_and_offsets,
            show_ids,
            labels,
            weights,
        )
        # optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        #
        for i in range(NUM_LABEL):
            epoch_loss[i].append(loss_vals[i].item())

        if iter % 100 == 0:
            print(
                f"Epoch: {epoch}\tIter: {iter}\tloss_vals:\t{[epoch_loss[i][-1] for i in range(NUM_LABEL)]}"
            )
        if max_iter is not None and iter == max_iter:
            break
    return epoch_loss


def sum_epoch_losses(name, epoch, epoch_loss):
    average_loss = [sum(l) / len(l) for l in epoch_loss]
    overall_sum_loss = sum(average_loss)
    print(
        f"Report: {name}\tEpoch {epoch}, total loss: {overall_sum_loss}\tAverage Weighted Loss: {average_loss}"
    )
    return overall_sum_loss
