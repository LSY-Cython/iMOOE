import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import os
import pickle as pkl


class DiffReact2dSim:
    def __init__(
        self,
        Du: float = 1e-3,
        Dv: float = 5e-3,
        k: float = 5e-3,
        t: float = 50,
        tdim: int = 501,
        x_left: float = -1.0,
        x_right: float = 1.0,
        xdim: int = 50,
        y_bottom: float = -1.0,
        y_top: float = 1.0,
        ydim: int = 50,
        n_block: int = 6,
        seed: int = 0,
    ):
        """
        Constructor method initializing the parameters for the diffusion
        sorption problem.
        :param Du: The diffusion coefficient of u
        :param Dv: The diffusion coefficient of v
        :param k: The reaction parameter
        :param t: Stop time of the simulation
        :param tdim: Number of simulation steps
        :param x_left: Left end of the 2D simulation field
        :param x_right: Right end of the 2D simulation field
        :param xdim: Number of spatial steps between x_left and x_right
        :param y_bottom: bottom end of the 2D simulation field
        :param y_top: top end of the 2D simulation field
        :param ydim: Number of spatial steps between y_bottom and y_top
        :param n_block: Number of local sources in initial conditions
        """

        # Set class parameters
        self.Du = Du
        self.Dv = Dv
        self.k = k

        self.T = t
        self.X0 = x_left
        self.X1 = x_right
        self.Y0 = y_bottom
        self.Y1 = y_top

        self.Nx = xdim
        self.Ny = ydim
        self.Nt = tdim

        # Calculate grid size and generate grid
        self.dx = (self.X1 - self.X0) / (self.Nx)
        self.dy = (self.Y1 - self.Y0) / (self.Ny)
        print(f"dx: {self.dx}, dy: {self.dy}")

        self.x = np.linspace(self.X0 + self.dx / 2, self.X1 - self.dx / 2, self.Nx)
        self.y = np.linspace(self.Y0 + self.dy / 2, self.Y1 - self.dy / 2, self.Ny)

        # Time steps to store the simulation results
        self.t = np.linspace(0, self.T, self.Nt)

        self.n_block = n_block
        self.seed = seed

    def generate_sample(self):
        """
        Single sample generation using the parameters of this simulator.
        :return: The generated sample as numpy array(t, x, y, num_features)
        """
        # # Generate initial conditions from N(0, 1)
        # rng = np.random.default_rng(self.seed)
        # u0 = rng.standard_normal(self.Nx * self.Ny)
        # v0 = rng.standard_normal(self.Nx * self.Ny)

        # Generate initial conditions formed by local sources
        u0 = 0.95 * np.ones((self.Nx, self.Ny))
        v0 = 0.05 * np.ones((self.Nx, self.Ny))
        for _ in range(self.n_block):
            rx = int(self.Nx / 10)
            ry = int(self.Ny / 10)
            sx = np.random.randint(low=0, high=self.Nx - rx, size=1)[0]
            sy = np.random.randint(low=0, high=self.Ny - ry, size=1)[0]
            u0[sx:(sx + rx), sy:(sy + ry)] = 0.
            v0[sx:(sx + rx), sy:(sy + ry)] = 1.

        u0 = u0.reshape(self.Nx * self.Ny)
        v0 = v0.reshape(self.Nx * self.Ny)
        u0 = np.concatenate((u0, v0))

        # Generate arrays as diagonal inputs to the Laplacian matrix, using finite volume method
        main_diag = (
            -2 * np.ones(self.Nx) / self.dx**2 - 2 * np.ones(self.Nx) / self.dy**2
        )
        main_diag[0] = -1 / self.dx**2 - 2 / self.dy**2
        main_diag[-1] = -1 / self.dx**2 - 2 / self.dy**2
        main_diag = np.tile(main_diag, self.Ny)
        main_diag[: self.Nx] = -2 / self.dx**2 - 1 / self.dy**2
        main_diag[self.Nx * (self.Ny - 1) :] = -2 / self.dx**2 - 1 / self.dy**2
        main_diag[0] = -1 / self.dx**2 - 1 / self.dy**2
        main_diag[self.Nx - 1] = -1 / self.dx**2 - 1 / self.dy**2
        main_diag[self.Nx * (self.Ny - 1)] = -1 / self.dx**2 - 1 / self.dy**2
        main_diag[-1] = -1 / self.dx**2 - 1 / self.dy**2

        left_diag = np.ones(self.Nx)
        left_diag[0] = 0
        left_diag = np.tile(left_diag, self.Ny)
        left_diag = left_diag[1:] / self.dx**2

        right_diag = np.ones(self.Nx)
        right_diag[-1] = 0
        right_diag = np.tile(right_diag, self.Ny)
        right_diag = right_diag[:-1] / self.dx**2

        bottom_diag = np.ones(self.Nx * (self.Ny - 1)) / self.dy**2

        top_diag = np.ones(self.Nx * (self.Ny - 1)) / self.dy**2

        # Generate the sparse Laplacian matrix
        diagonals = [main_diag, left_diag, right_diag, bottom_diag, top_diag]
        offsets = [0, -1, 1, -self.Nx, self.Nx]
        self.lap = diags(diagonals, offsets)

        # Solve the diffusion reaction problem
        prob = solve_ivp(self.rc_ode, (0, self.T), u0, t_eval=self.t)
        ode_data = prob.y

        sample_u = np.transpose(ode_data[: self.Nx * self.Ny]).reshape(
            -1, self.Ny, self.Nx
        )
        sample_v = np.transpose(ode_data[self.Nx * self.Ny :]).reshape(
            -1, self.Ny, self.Nx
        )

        return np.stack((sample_u, sample_v), axis=-1)

    def rc_ode(self, t, y):
        """
        Solves a given equation for a particular time step.
        :param t: The current time step
        :param y: The equation values to solve
        :return: A finite volume solution
        """

        # Separate y into u and v
        u = y[: self.Nx * self.Ny]
        v = y[self.Nx * self.Ny :]

        # Calculate reaction function for each unknown
        react_u = u - u**3 - self.k - v
        react_v = u - v

        # Calculate time derivative for each unknown
        u_t = react_u + self.Du * (self.lap @ u)
        v_t = react_v + self.Dv * (self.lap @ v)

        # Stack the time derivative into a single array y_t
        return np.concatenate((u_t, v_t))


def solve_diff_react_2d(Du, Dv, k, t, tdim, xdim, ydim, n_block, seed):
    dr2d_sim_obj = DiffReact2dSim(
        Du=Du,
        Dv=Dv,
        k=k,
        t=t,
        tdim=tdim,
        x_left=-1.0,
        x_right=1.0,
        xdim=xdim,
        y_bottom=-1.0,
        y_top=1.0,
        ydim=ydim,
        n_block=n_block,
        seed=seed)
    data_sample = dr2d_sim_obj.generate_sample()  # (T, Nx, Ny, C)
    data_sample = np.moveaxis(data_sample, 0, -1)  # (Nx, Ny, C, T)
    return data_sample


def gen_dr2d_dataset(cfg):
    n_env = len(cfg["params_eq"])
    n_data_per_env = cfg["n_data_per_env"]
    data = {}

    for env_index in range(n_env):
        data[f"env_{env_index}"] = []
        env_params = cfg["params_eq"][env_index]
        for data_index in range(n_data_per_env):
            solution = {}
            sample_seed = env_index * n_data_per_env + data_index
            imax = np.iinfo(np.int32).max
            sample_seed = sample_seed if (cfg["mode"] == "train") else (imax - sample_seed)

            state_trace = solve_diff_react_2d(
                Du=env_params["Du"], Dv=env_params["Dv"], k=env_params["k"],
                t=cfg["t_span"], tdim=cfg["Nt"], xdim=cfg["Nx"], ydim=cfg["Ny"],
                n_block=cfg["n_block"], seed=sample_seed)

            print(f'generating DR2d-{cfg["mode"]} env{env_index} data{data_index} seed{sample_seed} '
                  f'params{env_params} shape{state_trace.shape}')
            solution['state'] = state_trace
            solution['t_step'] = np.linspace(0, cfg["t_span"], cfg["Nt"])
            solution['env_index'] = env_index
            solution['env_params'] = env_params
            data[f"env_{env_index}"].append(solution)

    out_dir = cfg["path"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # data_path = "dr2d_" + cfg["mode"] + f"_env{n_env}_N{n_env*n_data_per_env}" \
    #             + f"_Nx{cfg['Nx']}_Ny{cfg['Ny']}_T{cfg['Nt']}" + ".pkl"
    data_path = "dr2d_" + cfg["mode"] + f"_env{n_env}_N{n_env * n_data_per_env}" \
                + f"_Nx{cfg['Nx']}_Ny{cfg['Ny']}_T{cfg['Nt']}" + ".pkl"
    with open(os.path.join(out_dir, data_path), "wb") as f:
        pkl.dump(data, f)


def exp_dr2d_generation(num_env_train, n_data_per_env_train, num_env_test, n_data_per_env_test,
                        Du, Dv, k, id_low_scale=1, id_high_scale=2, ood_low_scale=2, ood_high_scale=3, is_test=False):
    # Define ID/OOD physical parameter ranges
    id_lows = np.array([Du*id_low_scale, Dv*id_low_scale, k*id_low_scale])
    id_highs = np.array([Du*id_high_scale, Dv*id_high_scale, k*id_high_scale])
    ood_lows = np.array([Du*ood_low_scale, Dv*ood_low_scale, k*ood_low_scale])
    ood_highs = np.array([Du*ood_high_scale, Dv*ood_high_scale, k*ood_high_scale])
    np.random.seed(2025)
    train_env_params = np.round(np.random.uniform(id_lows, id_highs, size=(num_env_train, len(id_highs))), 4)
    id_env_params = np.round(np.random.uniform(id_lows, id_highs, size=(num_env_test, len(id_highs))), 4)
    ood_env_params = np.round(np.random.uniform(ood_lows, ood_highs, size=(num_env_test, len(ood_highs))), 4)

    # Generate training dataset
    print(f"n_train: {num_env_train * n_data_per_env_train} instances")
    dr2d_train_config = {
        "n_data_per_env": n_data_per_env_train,
        "t_span": 20,  # prone to fall inside attractors
        "Nt": 20+1,  # number of time steps, which is not equal to int_steps in solve_inp
        "Nx": 64,  # number of spatial steps
        "Ny": 64,  # number of spatial steps
        "n_block": 6,  # number of initial local sources
        "mode": "train",
        "path": "data/dr2d",
        "params_eq": [  # pde coefficients
            {"Du": Du, "Dv": Dv, "k": k} for Du, Dv, k in train_env_params]
    }
    gen_dr2d_dataset(dr2d_train_config)

    # Generate ID testing dataset
    print(f"n_id_test: {num_env_test * n_data_per_env_test} instances")
    dr2d_test_id_config = dict()
    dr2d_test_id_config.update(dr2d_train_config)
    dr2d_test_id_config["n_data_per_env"] = n_data_per_env_test
    dr2d_test_id_config["mode"] = "test_id"
    dr2d_test_id_config["params_eq"] = [{"Du": Du, "Dv": Dv, "k": k} for Du, Dv, k in id_env_params]
    if is_test:
        gen_dr2d_dataset(dr2d_test_id_config)

    # Generate OOD testing dataset
    print(f"n_ood_test: {num_env_test * n_data_per_env_test} instances")
    dr2d_test_ood_config = dict()
    dr2d_test_ood_config.update(dr2d_test_id_config)
    dr2d_test_ood_config["mode"] = "test_ood"
    dr2d_test_ood_config["params_eq"] = [{"Du": Du, "Dv": Dv, "k": k} for Du, Dv, k in ood_env_params]
    if is_test:
        gen_dr2d_dataset(dr2d_test_ood_config)


if __name__ == "__main__":
    exp_dr2d_generation(num_env_train=16, n_data_per_env_train=64, num_env_test=16, n_data_per_env_test=8,
                        Du=1e-3, Dv=5e-3, k=5e-3, is_test=True)

