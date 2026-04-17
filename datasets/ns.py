import torch
import math
import numpy as np
import pickle as pkl
import os
import itertools
import time
from grf import GaussianRF


def solve_navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):
    """
    :param w0: initial vorticity
    :param f: forcing term
    :param visc: viscosity (1/Re)
    :param T: final time
    :param delta_t: internal time-step for solve (decrease if blow-up)
    :param record_steps: number of in-time snapshots to record
    :return: sol: vorticity trajectory in (bsize, s, s, record_steps)
             sol_t: eval time in (record_steps, )
    """

    #Grid size - must be power of 2
    N = w0.size()[-1]

    #Maximum frequency
    k_max = math.floor(N/2.0)

    #Number of steps to final time
    steps = math.ceil(T/delta_t)

    #Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)

    #Forcing to Fourier space
    f_h = torch.fft.rfft2(f)

    #If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    #Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)

    #Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    #Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0
    #Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    #Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    #Record counter
    c = 0
    #Physical time
    t = 0.0
    for j in range(steps):
        #Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        #Velocity field in x-direction = psi_y
        q = 2. * math.pi * k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N, N))

        #Velocity field in y-direction = -psi_x
        v = -2. * math.pi * k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N, N))

        #Partial x of vorticity
        w_x = 2. * math.pi * k_x * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(N, N))

        #Partial y of vorticity
        w_y = 2. * math.pi * k_y * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(N, N))

        #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.rfft2(q*w_x + v*w_y)

        #Dealias
        F_h = dealias*F_h

        #Crank-Nicolson update
        w_h = (-delta_t*F_h + delta_t*f_h + (1.0 - 0.5*delta_t*visc*lap)*w_h)/(1.0 + 0.5*delta_t*visc*lap)

        #Update real time (used only for recording)
        t += delta_t

        if (j+1) % record_time == 0:
            #Solution in physical space
            w = torch.fft.irfft2(w_h, s=(N, N))

            #Record solution and time
            sol[...,c] = w
            sol_t[c] = t

            c += 1

    return sol, sol_t


def gen_ns2d_dataset(cfg):
    n_env = len(cfg["params_eq"])
    n_data_per_env = cfg["n_data_per_env"]
    data = {}
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    s = cfg["resolution"]
    t_span = cfg["t_span"]
    record_steps = cfg["Nt"]-1  # Number of snapshots from solution
    bsize = cfg["bsize"]  # solve equations in batches (order of magnitude speed-up)

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

    # Forcing function: 0.1*(sin(omiga*pi(x+y)) + cos(omiga*pi(x+y)))
    coord = torch.linspace(0, 1, s + 1, device=device)
    coord = coord[0:-1]
    X, Y = torch.meshgrid(coord, coord, indexing='ij')

    for env_index in range(n_env):
        data[f"env_{env_index}"] = []
        env_params = cfg["params_eq"][env_index]
        data_index = 0
        for i in range(n_data_per_env//bsize):
            start_time = time.time()
            # Sample random fields
            w0 = GRF.sample(bsize)  # (bsize, s, s)
            # Varying context parameters
            vis = env_params["viscosity"]  # viscosity coefficient
            omiga = env_params["omiga"]  # force frequency coefficient
            f = 0.1 * (torch.sin(omiga * math.pi * (X + Y)) + torch.cos(omiga * math.pi * (X + Y)))

            # Solve NS
            sol, sol_t = solve_navier_stokes_2d(w0, f, vis, t_span, 1e-4, record_steps)
            sol_trace = torch.cat((w0.unsqueeze(-1), sol), dim=-1)  # (bsize, S, S, Nt)
            print(f"eval_time={sol_t}s")
            end_time = time.time()

            for j in range(bsize):
                state_trace = sol_trace[j].unsqueeze(2).detach().cpu().numpy()  # (S, S, 1, Nt)
                sol_time = np.round((end_time - start_time) / bsize, 4)
                print(f'generating NS2d-{cfg["mode"]} env{env_index} data{data_index} '
                      f'params{env_params} shape{state_trace.shape} time={sol_time}s')
                solution = dict()  # global pointer
                solution['state'] = state_trace
                solution['t_step'] = np.linspace(0, t_span, cfg["Nt"])
                solution['env_index'] = env_index
                solution['env_params'] = env_params
                data[f"env_{env_index}"].append(solution)
                data_index += 1

    out_dir = cfg["path"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data_path = "ns2d_" + cfg["mode"] + f"_env{n_env}_N{n_env*n_data_per_env}" \
                + f"_S{s}_T{cfg['Nt']}" + ".pkl"
    with open(os.path.join(out_dir, data_path), "wb") as f:
        pkl.dump(data, f)


def exp_ns2d_generation(num_env_train, n_data_per_env_train, num_env_test, n_data_per_env_test,
                        vis_low=1e-5, vis_high=1e-3, omiga=2, is_test=False):
    # Define ID/OOD physical parameter ranges
    train_vis_vals = np.linspace(vis_low, vis_high, num_env_train)
    id_vis_vals = np.linspace(vis_low, vis_high, num_env_test)
    ood_vis_vals_left = np.linspace(vis_low*0.5, vis_low*0.8, num_env_test//2)
    ood_vis_vals_right = np.linspace(vis_high*1.2, vis_high*2, num_env_test//2)
    ood_vis_vals = np.concatenate((ood_vis_vals_left, ood_vis_vals_right))
    train_omiga_vals = [omiga]
    id_omiga_vals = [omiga]
    ood_omiga_vals = [omiga-1, omiga+1]
    train_env_params = list(itertools.product(train_vis_vals, train_omiga_vals))
    id_env_params = list(itertools.product(id_vis_vals, id_omiga_vals))
    ood_env_params = list(itertools.product(ood_vis_vals, id_omiga_vals))

    # Generate training dataset
    print(f"n_train: {num_env_train * n_data_per_env_train} instances")
    ns2d_train_config = {
        "n_data_per_env": n_data_per_env_train,
        "bsize": 32 if n_data_per_env_train % 32 == 0 else n_data_per_env_train,
        "t_span": 50.0,
        "Nt": 30+1,  # number of time steps, options: [20, 30, 50]
        "resolution": 64,  # number of spatial steps
        "mode": "train",
        "path": "data/ns2d",
        "params_eq": [  # pde coefficients
            {"viscosity": vis, "omiga": omiga} for vis, omiga in train_env_params]
    }
    gen_ns2d_dataset(ns2d_train_config)

    # Generate ID testing dataset
    print(f"n_id_test: {num_env_test * n_data_per_env_test} instances")
    ns2d_test_id_config = dict()
    ns2d_test_id_config.update(ns2d_train_config)
    ns2d_test_id_config["n_data_per_env"] = n_data_per_env_test
    ns2d_test_id_config["bsize"] = n_data_per_env_test
    ns2d_test_id_config["mode"] = "test_id"
    ns2d_test_id_config["params_eq"] = [{"viscosity": vis, "omiga": omiga} for vis, omiga in id_env_params]
    if is_test:
        gen_ns2d_dataset(ns2d_test_id_config)

    # Generate OOD testing dataset
    print(f"n_ood_test: {num_env_test * n_data_per_env_test} instances")
    ns2d_test_ood_config = dict()
    ns2d_test_ood_config.update(ns2d_test_id_config)
    ns2d_test_ood_config["mode"] = "test_ood"
    ns2d_test_ood_config["params_eq"] = [{"viscosity": vis, "omiga": omiga} for vis, omiga in ood_env_params]
    if is_test:
        gen_ns2d_dataset(ns2d_test_ood_config)


if __name__ == "__main__":
    # exp_ns2d_generation(num_env_train=16, n_data_per_env_train=64, num_env_test=4, n_data_per_env_test=32, is_test=True)

    # exp_ns2d_generation(num_env_train=16, n_data_per_env_train=64, num_env_test=16, n_data_per_env_test=8, is_test=False)
    # exp_ns2d_generation(num_env_train=4, n_data_per_env_train=256, num_env_test=16, n_data_per_env_test=8, is_test=False)
    # exp_ns2d_generation(num_env_train=16, n_data_per_env_train=256, num_env_test=16, n_data_per_env_test=8, is_test=True)

    # exp_ns2d_generation(num_env_train=64, n_data_per_env_train=16, num_env_test=16, n_data_per_env_test=8, is_test=False)

    exp_ns2d_generation(num_env_train=16, n_data_per_env_train=64, num_env_test=16, n_data_per_env_test=8, is_test=True)