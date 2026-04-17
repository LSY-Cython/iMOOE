"""
Implementation of our proposed MoE-based physics-informed invariant learning.
"""

import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import yaml
import argparse
from datasets.loaders import get_pde_dataloader
from models.forecasting import Forecaster
from losses import *
from metrics import *
from models.framework import STEFunction
from optim import Adam as AdamFNO
from utils import *
import time
import os


def main(config):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ablation study identifiers
    is_select = int(config["expert"]["is_select"])
    is_TI = int(config["optim"]["with_temporal_invariance"])
    is_ID = int(config["optim"]["with_ic_division"])
    lambda_mask = config["optim"]["lambda_mask"]
    lambda_vrex = config["optim"]["lambda_vrex"]
    lambda_rich = config["optim"]["lambda_rich"]
    rich_start_step = config["optim"]["rich_start_step"]
    rich_end_step = config["optim"]["rich_end_step"]
    ablate_idx = f"{args.expert_network}_select{is_select}_TI{is_TI}_ID{is_ID}_" \
                 f"lm{lambda_mask}_lv{lambda_vrex}_lr{lambda_rich}_rs{rich_start_step}"

    if args.expert_network == "FNO2d":
        width = config["expert"]["FNO2d"]["width"]
        ablate_idx += f"_width{width}"
    if args.expert_network == "OFormerUniform2d":
        latent_channels = config["expert"]["OFormerUniform2d"]["latent_channels"]
        encoder_depth = config["expert"]["OFormerUniform2d"]["encoder_depth"]
        ablate_idx += f"_latent{latent_channels}_depth{encoder_depth}"

    # create model folder
    model_dir = config["output"]["model_dir"] + f"_{ablate_idx}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # load data
    data_cfg = config["data"]
    num_env_train = data_cfg["num_env_train"]
    train_loader, test_id_loader, test_ood_loader = get_pde_dataloader(cfg_data=data_cfg)
    ntrain = num_env_train*data_cfg["n_data_per_env_train"]
    ntest_id = len(test_id_loader)*data_cfg["batch_size_test"]
    ntest_ood = len(test_ood_loader)*data_cfg["batch_size_test"]
    print(f"ntrain={ntrain}, ntest_id={ntest_id}, ntest_ood={ntest_ood}")

    # create model
    model = Forecaster(config=config, exp_net=args.expert_network, int_method=args.int_method,
                       int_options=args.int_options, int_step_scale=args.int_step_scale, device=device).to(device)
    model_size = count_params(model)
    print(f"{args.expert_network}_moe size: {model_size}")

    # optimization
    opt_cfg = config["optim"]
    n_epochs = opt_cfg["n_epochs"]
    if "FNO" in args.expert_network:
        optimizer = AdamFNO(model.parameters(), lr=opt_cfg["init_lr"], weight_decay=opt_cfg["weight_decay"])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=opt_cfg["init_lr"], weight_decay=opt_cfg["weight_decay"])
    p1 = int(0.75 * n_epochs)
    p2 = int(0.9 * n_epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)
    for name, param in model.named_parameters():
        print(name, param.requires_grad, param.size())

    # loss function
    lp_loss = RelativeL2(reduction=False)
    sp_loss = SobolevNorm(k=1, group=True, size_average=True, reduction=False)

    # training stage
    if args.is_train:
        print(f"Training stage begins")
        train_error = []
        lambda_vrex_temp = 0.0
        for epoch in range(n_epochs):
            model.train()
            train_loss = {"total": 0, "pred": 0, "mask": 0, "vrex": 0, "freq": 0}
            start_time = time.time()

            # linear scheduler to modulate the weight of OOD and feature enriching objectives
            warm_step = opt_cfg["warm_step"]
            const_step = opt_cfg["const_step"]
            step_size = opt_cfg["step_size"]
            if epoch <= warm_step:
                lambda_vrex_temp = 0.0
            elif warm_step < epoch <= const_step:
                if (epoch-warm_step) % step_size == 0:
                    lambda_vrex_temp = lambda_vrex * (epoch-warm_step) / (const_step-warm_step)
            else:
                lambda_vrex_temp = lambda_vrex
            if epoch > rich_end_step or epoch < rich_start_step:
                lambda_rich_temp = 0.0
            else:
                lambda_rich_temp = lambda_rich
            print(f"epoch={epoch}, ood_weight={lambda_vrex_temp}, freq_rich_weight={lambda_rich_temp}")

            for i, data in enumerate(train_loader):
                # divide training environments by exogenous parameters, i.e. concept shift
                env_indices = data["env_index"].detach().cpu().numpy()  # (B, )
                state = data["state"].to(device)  # (B, Nx, Ny, C, Nt)
                t = data["time"].to(device)  # (B, Nt)
                context = data["context"].to(device)  # (B, env_dim)
                output = model(x=state, c=context, t=t, mode="train")

                # ode autoregressive prediction
                preds, targets = output["preds"], output["targets"]   # (B*(Nt-init_step), Nx, Ny, C)
                masks = output["masks"]  # (C_diff, num_exp)
                eval_steps = output["eval_steps"]  # (B*(Nt-init_step), )

                # regularization on element selection module
                # calculate sparsification loss
                sparse_loss = torch.norm(masks, p=1)
                # calculate diversification loss
                pair_similarity = torch.exp(-torch.cdist(masks.T, masks.T))  # (num_expert, num_expert)
                diverse_loss = torch.sum(torch.triu(pair_similarity, diagonal=1))  # excluding diagonal values
                num_non_zero = 0.5 * masks.shape[1] * (masks.shape[1] - 1)
                mask_loss = 0.0 * sparse_loss + diverse_loss / num_non_zero  # only diversity loss is better

                # calculate prediction loss
                batch_lp_loss = lp_loss(preds, targets)  # (B*(Nt-init_step), )
                pred_loss = torch.mean(batch_lp_loss)

                # impose sobolev norm regularization on high-frequency modes
                batch_freq_loss = sp_loss(preds, targets)  # (B*(Nt-init_step), )
                freq_loss = torch.mean(batch_freq_loss)

                # calculate OOD objectives
                env_indices = np.repeat(env_indices, preds.shape[0] // env_indices.shape[0])  # (B*(Nt-init_step), )
                env_loss_set = []
                # whether to divide training environments by time steps, i.e. covariate shift
                if is_TI:
                    # divide batch loss by both environment and time invariance
                    inv_keys = list(zip(env_indices, eval_steps))  # (B*(Nt-init_step), )
                else:
                    # divide batch loss merely by environment invariance
                    inv_keys = env_indices  # (B*(Nt-init_step), )
                inv_loss_groups = {}
                for k in range(len(inv_keys)):
                    key = inv_keys[k]
                    value = batch_lp_loss[k]  # not involve frequency regularization
                    if key in inv_loss_groups:
                        inv_loss_groups[key].append(value)
                    else:
                        inv_loss_groups[key] = [value]
                for _, inv_loss in inv_loss_groups.items():
                    env_loss_set.append(torch.mean(torch.stack(inv_loss)))

                # calculate V-Rex equality loss
                env_loss_set = torch.stack(env_loss_set)
                vrex_loss = torch.var(env_loss_set)

                # treat each initial condition as an independent context
                if is_ID:
                    if not is_TI:
                        batch_lp_loss = rearrange(batch_lp_loss, '(b t) -> b t', b=state.shape[0])
                        batch_lp_loss = torch.mean(batch_lp_loss, dim=1)  # (B, )
                    vrex_loss = torch.var(batch_lp_loss)

                # overall training objective
                total_loss = pred_loss + mask_loss*lambda_mask + vrex_loss*lambda_vrex_temp + freq_loss*lambda_rich_temp

                optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                optimizer.step()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                train_loss["total"] += total_loss.item()
                train_loss["pred"] += pred_loss.item()
                train_loss["mask"] += mask_loss.item()
                train_loss["vrex"] += vrex_loss.item()
                train_loss["freq"] += freq_loss.item()
            end_time = time.time()
            train_loss["total"]/=len(train_loader); train_loss["pred"]/=len(train_loader)
            train_loss["mask"]/=len(train_loader); train_loss["vrex"]/=len(train_loader)
            train_loss["freq"]/=len(train_loader)
            lr_scheduler.step()
            if epoch >= n_epochs - 10:
                torch.save(model.state_dict(), f"{model_dir}/{epoch}.pt")
                print(f"Save {model_dir}/{epoch}.pt")
            print(f"[epoch {epoch}/{n_epochs}]: total_loss={train_loss['total']}, pred_loss={train_loss['pred']}, "
                  f"mask_loss={train_loss['mask']}, vrex_loss={train_loss['vrex']}, freq_loss={train_loss['freq']}, "
                  f"time={end_time-start_time}s")
            if epoch % 50 == 0:
                print(f"Differentiation element mask: {masks}")
            train_error.append(train_loss["total"])

        output_file_name = f"{args.dataset}_moe_env{num_env_train}_N{ntrain}_{ablate_idx}"
        plot_train_loss(train_error, output_file_name)

    # Testing stage
    else:
        # Load weights
        if not args.is_adapt:
            best_epoch = args.best_epoch
            ablate_idx += f"_{args.test_type}"
            ckpt_file = f"{model_dir}/{best_epoch}.pt"
        else:
            best_epoch_adapt = args.best_epoch_adapt
            adapt_idx = f"_adapt_{args.test_type}_{args.adapt_net}"
            ablate_idx += adapt_idx
            ckpt_file = f"{model_dir + adapt_idx}/{best_epoch_adapt}.pt"
        model.load_state_dict(torch.load(ckpt_file))
        print(f"Load {ckpt_file}")
        output_file_name = f"{args.dataset}_moe_env{num_env_train}_N{ntrain}_{ablate_idx}"

        model.eval()
        if args.test_type == "ID":
            test_loader = test_id_loader
        elif args.test_type == "OOD":
            test_loader = test_ood_loader
        else:
            raise NotImplementedError
        print(f"{args.test_type} testing stage begins")

        # Analyzing 0-1 element selection module
        expert_set = model.moe_model.expert_set
        select_type = config["expert"]["select_type"]
        for k in range(len(expert_set)):
            if select_type == "hard":
                mask = STEFunction.apply(expert_set[k].W.param)
            elif select_type == "soft":
                mask = F.softmax(expert_set[k].W.param)
            else:
                raise NotImplementedError
            print(f"Expert {k} mask: {mask}")

        # Evaluation metrics
        metrics = {"nMSE": {}, "fRMSE": {}}
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                start_time = time.time()
                state = data["state"].to(device)
                context = data["context"].to(device)
                t = data["time"].to(device)

                output = model(x=state, c=context, t=t, mode="test")

                pred_x = output["preds"]  # (B, Nx, Ny, C, Nt-init_step)
                targets = output["targets"].to(device)  # (B, Nx, Ny, C, Nt-init_step)

                err_nmse = cal_nMSE(pred_x, targets)  # (B, )
                err_frmse = cal_fRMSE(pred_x, targets)  # (B, )
                if np.isnan(err_nmse) or np.isnan(err_frmse):
                    continue
                metrics["nMSE"][f"case {i}"] = float(err_nmse[0])
                metrics["fRMSE"][f"case {i}"] = float(err_frmse[0])
                end_time = time.time()

                # Plot trajectories
                if args.is_plot:
                    init_step = data_cfg["init_step"]
                    pred_all = torch.cat((state[..., :init_step], pred_x), dim=-1)  # (B, Nx, Ny, C, Nt)
                    if "ns" in args.dataset:
                        plot_state_data(state, pred_all, t, channel=0, t_fraction=1, plt_cfg=config["plot"],
                                        ablate_idx=ablate_idx, fig_name=f"w_{args.test_type}_{i}")
                    elif "dr" in args.dataset:
                        plot_state_data(state, pred_all, t, channel=0, t_fraction=1, plt_cfg=config["plot"],
                                        ablate_idx=ablate_idx, fig_name=f"u_{args.test_type}_{i}")
                        plot_state_data(state, pred_all, t, channel=1, t_fraction=1, plt_cfg=config["plot"],
                                        ablate_idx=ablate_idx, fig_name=f"v_{args.test_type}_{i}")
                    else:
                        raise NotImplementedError

                print(f"{args.test_type} testing case {i}: env={context.detach().cpu().numpy()}, "
                      f"nMSE={err_nmse}, fRMSE={err_frmse}, time={end_time-start_time}s")
        avr_nMSE = np.mean(list(metrics["nMSE"].values()))
        avr_fRMSE = np.mean(list(metrics["fRMSE"].values()))
        std_nMSE = np.std(list(metrics["nMSE"].values()))
        std_fRMSE = np.std(list(metrics["fRMSE"].values()))
        print(f"{args.test_type} testing results: avr_nMSE={avr_nMSE}, avr_fRMSE={avr_fRMSE}")
        print(f"{args.test_type} testing results: std_nMSE={std_nMSE}, std_fRMSE={std_fRMSE}")
        with open(f"output/{output_file_name}.json", "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    # input args
    parser = argparse.ArgumentParser(description="NDE-MoE")
    parser.add_argument("--is_train", type=bool, default=True)
    # parser.add_argument("--is_train", type=bool, default=False)
    parser.add_argument("--best_epoch", type=int, default=499)

    parser.add_argument("--test_type", type=str, default="OOD", help="options: ['ID', 'OOD']")

    parser.add_argument("--is_plot", type=bool, default=False)

    parser.add_argument("--config_file", type=str, default="dr2d_moe.yaml")
    # parser.add_argument("--config_file", type=str, default="ns2d_moe.yaml")
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--dataset", type=str, default="dr2d")
    # parser.add_argument("--dataset", type=str, default="ns2d")

    parser.add_argument("--expert_network", type=str, default="FNO2d")
    # parser.add_argument("--expert_network", type=str, default="OFormerUniform2d")

    # rk4 will induce vastly more computation time than "euler"
    parser.add_argument("--int_method", type=str, default="euler", help="options: ['euler', 'midpoint', 'rk4']")
    parser.add_argument("--int_options", type=str, default="{}")
    parser.add_argument("--int_step_scale", type=float, default=1.0)

    args = parser.parse_args()
    print(args)

    # configuration
    cfg_path = "configs/" + args.config_file
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # fix seed
    fix_seed(args.seed)

    # run exp
    main(config)
