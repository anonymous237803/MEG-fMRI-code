import torch
import torch.nn as nn
import time
from utils import CorrelationLoss, save_checkpoint, load_checkpoint
import wandb
import numpy as np
import yaml
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def temporal_smoothness(neurons):
    # neurons: (B,T,N)
    diff = neurons[:, 1:] - neurons[:, :-1]
    return (diff**2).mean()


def train(model, dataloader, optimizer, device, meg_loss_weight=1.0, fmri_loss_weight=1.0, smooth_loss_weight=0.001, softmax_T=0, subject="A"):
    model.train()

    # whether training with segment
    use_segment = dataloader.batch_size > 1
    spacing = dataloader.dataset.spacing
    meg_valid = 1000
    fmri_valid = 10

    # define loss function
    loss_func = CorrelationLoss(dim=0)
    meg_weight = torch.from_numpy(np.load(f"data/{subject}_meg_repeat_corrs.npy"))
    fmri_weight = torch.from_numpy(np.load(f"data/Moth_{subject}_fmri-{spacing}_repeat_corrs.npy"))
    meg_weight = meg_weight - torch.min(meg_weight) + 1e-5
    fmri_weight = fmri_weight - torch.min(fmri_weight) + 1e-5
    if softmax_T > 0:
        meg_weight = F.softmax(meg_weight / softmax_T, dim=0)
        fmri_weight = F.softmax(fmri_weight / softmax_T, dim=0)

    epoch_loss = 0.0
    epoch_meg_loss = 0.0
    epoch_fmri_loss = 0.0
    for embeds, gt_meg, gt_fmri in dataloader:

        # load embeds and ground truth
        embeds = embeds.to(device)
        if use_segment:
            gt_meg = gt_meg[:, -meg_valid:, :]
            gt_fmri = gt_fmri[:, -fmri_valid:, :]

        # forward pass
        neurons, pred_meg, pred_fmri = model(embeds)
        pred_meg = pred_meg.cpu()
        pred_fmri = pred_fmri.cpu()
        if use_segment:
            pred_meg = pred_meg[:, -meg_valid:, :]
            pred_fmri = pred_fmri[:, -fmri_valid:, :]

        # flatten batch
        gt_meg = gt_meg.reshape(-1, gt_meg.shape[-1])
        gt_fmri = gt_fmri.reshape(-1, gt_fmri.shape[-1])
        pred_meg = pred_meg.reshape(-1, pred_meg.shape[-1])
        pred_fmri = pred_fmri.reshape(-1, pred_fmri.shape[-1])

        # compute loss
        meg_loss = loss_func(pred_meg, gt_meg, weight=meg_weight)
        fmri_loss = loss_func(pred_fmri, gt_fmri, weight=fmri_weight)
        smooth_loss = temporal_smoothness(neurons)
        loss = meg_loss_weight * meg_loss + fmri_loss_weight * fmri_loss + smooth_loss_weight * smooth_loss
        print(f"meg_loss: {meg_loss.item()}, fmri_loss: {fmri_loss.item()}")
        if wandb.run is not None:
            wandb.log({"story_meg_loss": meg_loss.item(), "story_fmri_loss": fmri_loss.item()})

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # record loss
        epoch_loss += loss.item()
        epoch_meg_loss += meg_loss.item()
        epoch_fmri_loss += fmri_loss.item()

        # save memory
        del embeds, gt_meg, gt_fmri, pred_meg, pred_fmri, loss, meg_loss, fmri_loss
        torch.cuda.empty_cache()

    epoch_loss /= len(dataloader)
    epoch_meg_loss /= len(dataloader)
    epoch_fmri_loss /= len(dataloader)
    return epoch_loss, epoch_meg_loss, epoch_fmri_loss


def validate(model, dataloader, device, meg_loss_weight=1.0, fmri_loss_weight=1.0, softmax_T=0, subject="A"):
    model.eval()
    assert dataloader.batch_size == 1, "Validation should be done with batch size 1"
    spacing = dataloader.dataset.spacing
    
    # define loss function
    loss_func = CorrelationLoss(dim=0)
    meg_weight = torch.from_numpy(np.load(f"data/{subject}_meg_repeat_corrs.npy"))
    fmri_weight = torch.from_numpy(np.load(f"data/Moth_{subject}_fmri-{spacing}_repeat_corrs.npy"))
    meg_weight = meg_weight - torch.min(meg_weight) + 1e-5
    fmri_weight = fmri_weight - torch.min(fmri_weight) + 1e-5
    if softmax_T > 0:
        meg_weight = F.softmax(meg_weight / softmax_T, dim=0)
        fmri_weight = F.softmax(fmri_weight / softmax_T, dim=0)
    
    val_loss = 0.0
    val_meg_loss = 0.0
    val_fmri_loss = 0.0
    for embeds, gt_meg, gt_fmri in dataloader:
        
        # load embeds and ground truth
        embeds = embeds.to(device)
        
        # forward pass
        _, pred_meg, pred_fmri = model(embeds)
        pred_meg = pred_meg.cpu()
        pred_fmri = pred_fmri.cpu()

        # flatten batch
        gt_meg = gt_meg.reshape(-1, gt_meg.shape[-1])
        gt_fmri = gt_fmri.reshape(-1, gt_fmri.shape[-1])
        pred_meg = pred_meg.reshape(-1, pred_meg.shape[-1])
        pred_fmri = pred_fmri.reshape(-1, pred_fmri.shape[-1])
        
        # compute loss
        meg_loss = loss_func(pred_meg, gt_meg, weight=meg_weight)
        fmri_loss = loss_func(pred_fmri, gt_fmri, weight=fmri_weight)
        loss = meg_loss_weight * meg_loss + fmri_loss_weight * fmri_loss
        print(f"val_meg_loss: {meg_loss.item()}, val_fmri_loss: {fmri_loss.item()}")
        
        # record loss
        val_loss += loss.item()
        val_meg_loss += meg_loss.item()
        val_fmri_loss += fmri_loss.item()
        
        # save memory
        del embeds, gt_meg, gt_fmri, pred_meg, pred_fmri, loss, meg_loss, fmri_loss
        torch.cuda.empty_cache()
        
    val_loss /= len(dataloader)
    val_meg_loss /= len(dataloader)
    val_fmri_loss /= len(dataloader)
    return val_loss, val_meg_loss, val_fmri_loss


# -----------------------
# Example usage in main
# -----------------------
if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from model import *
    from dataset import *
    from utils import *

    # check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    # training config
    name = "WdPnFq-seg8-flexconv4-A"
    with open(f"config/{name}.yaml", "r") as f:
        config = yaml.safe_load(f)
    MEG_SUBJECT = config["MEG_SUBJECT"]
    FMRI_SUBJECT = config["FMRI_SUBJECT"]
    use_segment = config["use_segment"]
    epoch_num = config["epoch_num"]
    spacing = config["spacing"]
    meg_loss_weight = config["meg_loss_weight"]
    fmri_loss_weight = 0.0
    fmri_loss_weight_conf = config["fmri_loss_weight"]
    softmax_T = config["softmax_T"]
    dataset_params = config["dataset"]
    model_params = config["model"]
    optim_params = config["optimizer"]
    print(f"{name}, {spacing}, use_segment: {use_segment}, meg_loss_weight: {meg_loss_weight}, fmri_loss_weight: {fmri_loss_weight}")
    
    # specify stories
    train_story_list = dict(
        Moth1=["souls", "avatar", "legacy", "odetostepfather"],
        Moth2=["howtodraw", "myfirstdaywiththeyankees", "naked", "life"],
        Moth3=["tildeath", "fromboyhoodtofatherhood", "sloth", "exorcism"],
        Moth4=["adollshouse", "inamoment", "theclosetthatateeverything", "adventuresinsayingyes", "haveyoumethimyet"],
        Moth5=["thatthingonmyarm", "eyespy", "itsabox", "hangtime"],
    )
    train_stories = [story for session in train_story_list.keys() for story in train_story_list[session]]
    val_stories = [["swimmingwithastronauts1", "swimmingwithastronauts2"]]

    # create dataset
    if use_segment:
        train_dataset_class = StorySegmentDataset
        train_batch_size = 8
    else:
        train_dataset_class = StoryDataset
        train_batch_size = 1
    train_dataset = train_dataset_class(
        MEG_SUBJECT,
        FMRI_SUBJECT,
        train_stories,
        name=name,
        spacing=spacing,
        **dataset_params,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataset = StoryDataset(
        MEG_SUBJECT,
        FMRI_SUBJECT,
        val_stories,
        name=name,
        spacing=spacing,
        pca_meg=train_dataset.pca_meg,
        pca_mri=train_dataset.pca_mri,
        **dataset_params,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    embed_dim = train_dataset.embed_dim

    # load forward solution to get the lead field
    fname_fwd = f"data/{MEG_SUBJECT}-{spacing}-fwd.fif"
    fwd = mne.read_forward_solution(fname_fwd, verbose=False)
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False)
    lead_field = fwd_fixed["sol"]["data"]
    lead_field = torch.from_numpy(lead_field)
    n_channels, n_neurons = lead_field.shape
    print(f"n_channels: {n_channels}, n_neurons: {n_neurons}")

    # --- TRAIN!!! --- #
    model = TransformerSourceModel(
        embed_dim=embed_dim,
        lead_field=lead_field,
        **model_params,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **optim_params)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    min_val_loss, min_val_meg_loss, min_val_fmri_loss = float("inf"), float("inf"), float("inf")
    start_epoch = 0
    meg_only_epoch = 20

    # load checkpoint if exists
    ckpt_path = None
    if ckpt_path is not None:
        start_epoch, min_val_loss, min_val_meg_loss, min_val_fmri_loss = load_checkpoint(ckpt_path, model, optimizer, scheduler, device=device)
        fmri_loss_weight = fmri_loss_weight_conf

    for epoch in range(start_epoch, epoch_num):

        if epoch == meg_only_epoch:
            fmri_loss_weight = fmri_loss_weight_conf
            for g in optimizer.param_groups:
                g["lr"] = optim_params["lr"] * 0.2
            print(f"Switched to joint training!")

        # train
        epoch_loss, epoch_meg_loss, epoch_fmri_loss = train(model, train_dataloader, optimizer, device, 
                                                            subject=MEG_SUBJECT, meg_loss_weight=meg_loss_weight, fmri_loss_weight=fmri_loss_weight, softmax_T=softmax_T)

        # validate
        with torch.no_grad():
            val_loss, val_meg_loss, val_fmri_loss = validate(model, val_dataloader, device, 
                                                             subject=MEG_SUBJECT, meg_loss_weight=meg_loss_weight, fmri_loss_weight=fmri_loss_weight_conf, softmax_T=softmax_T)
        scheduler.step(val_loss)

        # # save val checkpoint
        # if val_loss < min_val_loss:
        #     min_val_loss = val_loss
        #     filename = f"trained_models/{name}_val-loss-min.pth"
        #     save_checkpoint(filename, model, optimizer, scheduler, epoch, min_val_loss, min_val_meg_loss, min_val_fmri_loss)
        #     print(f"find new min val_loss: {min_val_loss} at epoch {epoch + 1}")