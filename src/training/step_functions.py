import torch


def train_fn(pl_module, batch):
    img = batch["image"]
    tab = batch["num"]
    y = batch["label"]

    logits = pl_module.model(img, tab, labels=y)
    loss = pl_module.criterion(logits, y)
    preds = torch.argmax(logits, dim=1)
    return loss, preds, y


def valid_fn(pl_module, batch):
    img = batch["image"]
    tab = batch["num"]
    y = batch["label"]

    if pl_module.use_ema and pl_module.model_ema is not None:
        logits = pl_module.model_ema.module(img, tab, labels=y)
    else:
        logits = pl_module.model(img, tab, labels=y)

    loss = pl_module.criterion(logits, y)
    preds = torch.argmax(logits, dim=1)
    return loss, preds, y