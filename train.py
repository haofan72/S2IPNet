from utils.tools import *

import numpy as np
import torch
from tqdm import tqdm


args = para_parser()

device = select_gpu(device_number=args.gpu_number)
print(f"device:{device}")

args.save_path = make_dir(args.dataset, args.model)
model, cal_loss, train_dataloader, test_dataloader = train_head(args)
model.to(device)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=args.min_lr
)

mIoU = mIoUMetric(1)
save_train_log(args, args.save_path)


def training_epoch(e):
    model.train()
    loss_set = []
    tbar = tqdm(train_dataloader)
    for _, batch_image, batch_label in tbar:
        batch_image = batch_image.to(device)
        batch_label = batch_label.to(device)

        preds = model(batch_image)
        if sum(w != 0 for w in args.decay_weights) == 1:
            nonzero_idx = next(i for i, w in enumerate(
                args.decay_weights) if w != 0)
            w = args.decay_weights[nonzero_idx]
            loss = w * cal_loss(preds[nonzero_idx], batch_label)
        else:
            loss = 0
            for i, pred in enumerate(preds):
                w = (
                    args.decay_weights[i]
                    if i < len(args.decay_weights)
                    else args.decay_weights[-1]
                )
                loss = loss + w * cal_loss(pred, batch_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_set.append(loss.item())
        tbar.set_description(
            "Epoch %d, training loss %.4f, lr %.6f"
            % (e, np.mean(loss_set), optimizer.param_groups[0]["lr"])
        )

    losses_avg = np.mean(loss_set)
    return losses_avg


def testing_epoch(e):
    model.eval()
    loss_set = []
    tbar = tqdm(test_dataloader)
    with torch.no_grad():
        mIoU.reset()
        for name, batch_image, batch_label in tbar:
            batch_image = batch_image.to(device)
            batch_label = batch_label.to(device)

            preds = model(batch_image)
            if sum(w != 0 for w in args.decay_weights) == 1:
                nonzero_idx = next(
                    i for i, w in enumerate(args.decay_weights) if w != 0
                )
                w = args.decay_weights[nonzero_idx]
                loss = w * cal_loss(preds[nonzero_idx], batch_label)
                pred = preds[nonzero_idx]
            else:
                loss = 0
                for i, pred in enumerate(preds):
                    w = (
                        args.decay_weights[i]
                        if i < len(args.decay_weights)
                        else args.decay_weights[-1]
                    )
                    loss = loss + w * cal_loss(pred, batch_label)
                pred = preds[0]
            loss_set.append(loss.item())

            mIoU.update(pred, batch_label)
            tbar.set_description("Epoch %d, testing loss %.4f" %
                                 (e, np.mean(loss_set)))
        _, mean_IOU = mIoU.get()
        test_loss = np.mean(loss_set)
        print(f"Epoch {e}, Mean IoU: {mean_IOU}， test loss: {test_loss}")
    return test_loss, mean_IOU


if __name__ == "__main__":
    print("Model Training .....")
    for epoch in range(args.start_epoch, args.epochs):
        train_loss = training_epoch(epoch)
        test_loss, mean_IOU = testing_epoch(epoch)
        scheduler.step()
