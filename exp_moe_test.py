import yaml
import argparse
from datasets.loaders import get_pde_dataloader
from models.forecasting import Forecaster
from metrics import *
from utils import *
import time
import pandas as pd


def main(config, test_type, best_epoch):
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
    ablate_idx = f"{args.expert_network}_select{is_select}_TI{is_TI}_ID{is_ID}_" \
                 f"lm{lambda_mask}_lv{lambda_vrex}_lr{lambda_rich}_rs{rich_start_step}"

    if args.expert_network == "FNO2d":
        width = config["expert"]["FNO2d"]["width"]
        ablate_idx += f"_width{width}"

    # create model folder
    model_dir = config["output"]["model_dir"] + f"_{ablate_idx}"

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

    # Load weights
    ckpt_file = f"{model_dir}/{best_epoch}.pt"
    model.load_state_dict(torch.load(ckpt_file))
    print(f"Load {ckpt_file}")
    output_file_name = f"{args.dataset}_moe_env{num_env_train}_N{ntrain}_{ablate_idx}"

    model.eval()
    if test_type == "ID":
        test_loader = test_id_loader
    elif test_type == "OOD":
        test_loader = test_ood_loader
    else:
        raise NotImplementedError
    print(f"{test_type} testing stage begins")

    # Evaluation metrics
    metrics = {"nMSE": {}, "fRMSE": {}}
    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
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
    avr_nMSE = np.mean(list(metrics["nMSE"].values()))
    avr_fRMSE = np.mean(list(metrics["fRMSE"].values()))
    std_nMSE = np.std(list(metrics["nMSE"].values()))
    std_fRMSE = np.std(list(metrics["fRMSE"].values()))
    print(f"[epoch={epoch}] {test_type} testing results: time={end_time-start_time}s")
    print(f"{test_type} testing results: avr_nMSE={avr_nMSE}, avr_fRMSE={avr_fRMSE}")
    print(f"{test_type} testing results: std_nMSE={std_nMSE}, std_fRMSE={std_fRMSE}")
    return avr_nMSE, avr_fRMSE, metrics, output_file_name


if __name__ == "__main__":
    # input args
    parser = argparse.ArgumentParser(description="NDE-MoE")

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

    # batch testing on average metrics
    results = {"test_epoch": [], "ID_nMSE": [], "OOD_nMSE": [], "ID_fRMSE": [], "OOD_fRMSE": []}
    n_epochs = config["optim"]["n_epochs"]
    for epoch in range(n_epochs):
        if epoch >= n_epochs - 10:
            id_nmse, id_frmse, _, output_file_name = main(config, test_type="ID", best_epoch=epoch)
            ood_nmse, ood_frmse, _, _ = main(config, test_type="OOD", best_epoch=epoch)

            results["test_epoch"].append(epoch)
            results["ID_nMSE"].append(id_nmse)
            results["OOD_nMSE"].append(ood_nmse)
            results["ID_fRMSE"].append(id_frmse)
            results["OOD_fRMSE"].append(ood_frmse)

    # write to csv
    res_df = pd.DataFrame(results)
    output_file_name += "_average"
    excel_file_path = f"output/{output_file_name}.xlsx"
    res_df.to_excel(excel_file_path, index=False, header=True)
    print(f"Output excel path: {excel_file_path}")