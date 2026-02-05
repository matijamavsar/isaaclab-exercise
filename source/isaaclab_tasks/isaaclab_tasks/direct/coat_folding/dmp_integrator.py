# dmp_integrator.py

import torch


class BatchDMPIntegrator:
    """
    Batch DMP integrator that initializes or re‐initializes any subset of
    batch‐indices (phase x, position y, velocity z, goal, weights, tau, etc.)
    in one single method: `reset_indices(...)`.  There is no separate `reset_state`.
    After calling `reset_indices(...)` at least once (for all indices you care about),
    you may call `step()` repeatedly to advance every entry in parallel.

    Usage:
        # 1) Create integrator (no state yet)
        integrator = BatchDMPIntegrator(N_basis=20, dof=3, device=torch.device('cuda'))

        # 2) Initialize all B environments at once:
        #    Pass `indices=torch.arange(B)` so that everything is set up.
        integrator.reset_indices(
            indices=torch.arange(B, device=device),
            batch=full_batch_tensor,    # shape (B, 2*d + d*Nb + 1)
            tau=full_tau_tensor,        # shape (B,)
            dt=0.01,
            variant=1
        )

        # 3) Step forward (all B):
        y_t, dy_t, ddy_t = integrator.step()  # each is shape (B, dof)

        # 4) Later, re‐initialize only a subset, e.g. [0,2,3], with new params:
        integrator.reset_indices(
            indices=torch.tensor([0,2,3], device=device),
            batch=new_full_batch_tensor,
            tau=new_tau_tensor,
            dt=0.01,                # can remain the same or change
            variant=1               # can remain the same or change
        )
        # Only entries 0,2,3 get restarted; the others continue from their current x,y,z.

        # 5) Continue stepping for all B (0,2,3 have fresh state; others keep going).
        y_next, dy_next, ddy_next = integrator.step()
    """

    def __init__(self, N_basis: int, dof: int, device: torch.device,
                 a_x: float = 2.0, a_z: float = 4.0):
        self.Nb  = N_basis
        self.dof = dof
        self.dev = device

        # DMP gains (as tensors on device)
        self.a_x = torch.tensor(a_x, device=self.dev)
        self.a_z = torch.tensor(a_z, device=self.dev)

        # Precompute RBF centers & widths once
        c = torch.exp(-self.a_x * torch.linspace(0, 1, self.Nb, device=self.dev))
        sigma2 = (torch.diff(c) / 2) ** 2
        sigma2 = torch.cat([sigma2, sigma2[-1:]], dim=0)
        self.c      = c            # shape: (Nb,)
        self.sigma2 = sigma2       # shape: (Nb,)

        # Placeholders; actual tensors get created on first reset_indices(...) call
        self.tau   = None          # shape: (B,)
        self.dt    = None          # float
        self.variant = None        # int, either 1 or 2

        self.y0    = None          # shape: (B, dof)
        self.goal  = None          # shape: (B, dof)
        self.w     = None          # shape: (B, dof, Nb)

        self.x     = None          # shape: (B,)
        self.y     = None          # shape: (B, dof)
        self.z     = None          # shape: (B, dof)

        # We keep a copy of the original tau so that if future calls to
        # reset_indices(...) do not supply a new tau, we can fall back to the original.
        self._tau0 = None


    def _extract(self, batch: torch.Tensor):
        """
        Given `batch` of shape (B, 2*d + d*Nb + 1), pull out:

          y0   = batch[:, :d]                  -> shape (B, dof)
          goal = batch[:, d:2*d]               -> shape (B, dof)
          w    = batch[:, 2*d:-1].view(B, dof, Nb) -> shape (B, dof, Nb)

        The final “+1” column is ignored here.
        """
        B, total_dim = batch.shape
        d, Nb = self.dof, self.Nb

        y0   = batch[:,            :d]            # (B, dof)
        goal = batch[:,      d : 2*d]              # (B, dof)
        w    = batch[:,  2*d : -1].reshape(B, d, Nb)  # (B, dof, Nb)
        return y0, goal, w


    def reset_indices(self,
                      indices:   torch.LongTensor,
                      batch:     torch.Tensor,
                      tau:       torch.Tensor,
                      dt:        float,
                      variant:   int):
        """
        (Re-)initialize exactly the specified batch-entries:

        indices:  1D LongTensor of shape (K,), subset of [0..B-1].
        batch:    Full tensor of shape (B, 2*d + d*Nb + 1).  We re-extract
                  y0, goal, w from these rows at the given indices.
        tau:      Tensor of shape (B,).  We set tau[indices] = tau[indices].
        dt:       Float timestep; stored internally (overwrites any previous dt).
        variant:  Integer (1 or 2); stored internally (overwrites any previous variant).

        For each i in `indices`, we do:
          1) y0[i], goal[i], w[i] ← extracted from `batch[i]`
          2) tau[i] ← tau[i]
          3) x[i] ← 1.0
          4) y[i] ← y0[i]
          5) z[i] ← 0.0

        If this is the **first** time you call reset_indices(...),
        then B, d, Nb, etc. get their shapes determined here, and
        all “global” buffers (y0, goal, w, tau, x, y, z) get allocated.

        Subsequent calls affect **only** the listed `indices`, leaving
        the other entries of x,y,z,tau, etc. exactly as they were.
        """
        indices = indices.to(self.dev).long()

        # On the very first call, we need to allocate all buffers
        if self.tau is None:
            # Move everything to device
            batch = batch.to(self.dev)
            tau   = tau.to(self.dev)

            B = batch.shape[0]  # number of parallel environments
            d = self.dof
            Nb = self.Nb

            # Store dt and variant
            self.dt      = dt
            self.variant = variant

            # Keep a copy of the “original” tau (so future calls can fall back if needed)
            self._tau0 = tau.clone()

            # Extract y0, goal, w for every env
            self.y0, self.goal, self.w = self._extract(batch)
            # Allocate
            self.tau = tau.clone()                   # (B,)
            self.x   = torch.ones((B,), device=self.dev)                    # (B,)
            self.y   = self.y0.clone()              # (B, dof)
            self.z   = torch.zeros_like(self.y)      # (B, dof)
            self.t   = torch.zeros_like(self.x)

            # Finally, reset those indices explicitly (in case indices is not all of [0..B-1])
            # (This overwrites exactly those rows, but since we just set y=x etc. for all, it’s
            #  redundant if indices = all, but required if indices is a strict subset.)
            y0_batch, goal_batch, w_batch = self._extract(batch)
            self.y0   [indices] = y0_batch   [indices]
            self.goal [indices] = goal_batch [indices]
            self.w    [indices] = w_batch    [indices]
            self.tau  [indices] = tau        [indices]
            self.x    [indices] = 1.0
            self.y    [indices] = y0_batch   [indices]
            self.z    [indices] = 0.0

        else:
            # Already initialized once.  We simply overwrite the specified indices.
            batch = batch.to(self.dev)
            tau   = tau.to(self.dev)

            # Keep (dt, variant) up to date
            self.dt      = dt
            self.variant = variant

            B = self.x.shape[0]
            if indices.max().item() >= B or indices.min().item() < 0:
                raise IndexError("Some entries in `indices` are out of range 0..(B-1).")

            # Re‐extract y0, goal, w from `batch`
            y0_batch, goal_batch, w_batch = self._extract(batch)

            # Overwrite exactly those indices:
            self.y0   [indices] = y0_batch   [indices]
            self.goal [indices] = goal_batch [indices]
            self.w    [indices] = w_batch    [indices]

            # Overwrite tau for those indices; if tau shape=(B,), simply pick out indices
            if tau.numel() == B:
                self.tau[indices] = tau[indices]
            else:
                raise ValueError("`tau` must be shape (B,) when calling reset_indices(...) after initialization.")

            # Reset phase & state for those indices
            self.x[indices] = 1.0
            self.y[indices] = self.y0[indices].clone()
            self.z[indices] = 0.0
            self.t[indices] = 0.0


    def step(self):
        """
        Advance one time‐step of DMP for all B in parallel.

        Returns:
          y_t   : current positions,   shape (B, dof)
          dy_t  : current velocities, shape (B, dof)
          ddy_t : current accelerations, shape (B, dof)
        """
        if self.tau is None:
            raise RuntimeError("Must call `reset_indices(...)` at least once before calling `step()`.")

        # 1) Update phase variable x:  dx = -a_x * x / tau
        dx = -self.a_x * self.x / self.tau              # (B,)
        self.x = self.x + dx * self.dt                  # (B,)

        # 2) Compute RBF activations ψᵢ(x) = exp( -0.5 * ( (x - c[i])^2 ) / sigma2[i] )
        #    psi: (B, Nb)
        psi = torch.exp(-0.5 * ((self.x.unsqueeze(-1) - self.c)**2) / self.sigma2)

        # 3) Normalize each row so that Σᵢ ψᵢ = 1
        psi_norm = psi / psi.sum(dim=-1, keepdim=True)  # (B, Nb)

        # 4) Forcing term f(x): (B, dof)
        #    w: (B, dof, Nb), so we do elementwise multiply with psi_norm (B, Nb)
        fx = self.x.unsqueeze(-1) * (self.w * psi_norm.unsqueeze(1)).sum(dim=-1)  # (B, dof)

        # 5) Compute dz, dy according to chosen variant
        if self.variant == 1:
            dz = self.a_z * (self.a_z/4 * (self.goal - self.y) - self.z) + fx  # (B, dof)
        else:
            a_z2 = self.a_z * 3
            K    = (a_z2**2) / 4
            D    = a_z2
            r    = self.goal  # treat “r” simply as goal
            dz   = (K*(r - self.y)
                    - D*self.z
                    - self.x.unsqueeze(-1)*(r - self.y0)*K
                    + K*fx)  # (B, dof)

        # Scale by tau:
        dz  = dz  / self.tau.unsqueeze(-1)  # (B, dof)
        dy  = self.z / self.tau.unsqueeze(-1)  # (B, dof)

        # 6) Integrate z, y forward
        self.z = self.z + dz * self.dt        # (B, dof)
        self.y = self.y + dy * self.dt        # (B, dof)

        self.t += self.dt

        return self.t, self.y, dy, dz / self.tau.unsqueeze(-1)

    def encode(self, pos_data: torch.Tensor, time: torch.Tensor = None,
               vel_data: torch.Tensor = None, num_weights: int = None,
               a_z: float = None, a_x: float = None):
        """
        Encode demonstration trajectories into DMP weights via least-squares.
        Supports batched input:
          pos_data: (B, T, dof) or (T, dof)
        Returns:
          W: (B, dof, Nb) or (dof, Nb) if single
        """
        y = pos_data.to(self.dev).float()
        # add batch dim if necessary
        if y.ndim == 2:
            y = y.unsqueeze(0)  # (1, T, dof)
        B, T, d = y.shape
        assert d == self.dof, "pos_data.dof mismatch"
        # time vector and dt
        if time is None or (torch.is_tensor(time) and time.numel() == 1):
            dt = float(time) if torch.is_tensor(time) else 1.0
            tvec = torch.arange(T, device=self.dev) * dt
        else:
            tvec = time.to(self.dev).float()
            dt = float(tvec[1] - tvec[0])
        tau = tvec[-1]
        # velocities
        if vel_data is None:
            dy = (y[:,1:,:] - y[:,:-1,:]) / dt
            dy = torch.cat([dy, dy[:,-1:,:]], dim=1)
        else:
            dy = vel_data.to(self.dev).float()
        # accelerations
        ddy = (dy[:,1:,:] - dy[:,:-1,:]) / dt
        ddy = torch.cat([ddy, ddy[:,-1:,:]], dim=1)
        # parameters
        Nb = num_weights or self.Nb
        ax = float(a_x) if a_x else float(self.a_x)
        az = float(a_z) if a_z else float(self.a_z)
        bz = az / 4.0
        # RBF centers & widths
        c = torch.exp(-ax * torch.linspace(0,1,Nb, device=self.dev))
        sigma = (torch.diff(c)/2)**2
        sigma = torch.cat([sigma, sigma[-1:]], dim=0)
        # phase variable xphase: (T,)
        xphase = torch.exp(-ax * tvec / tau)
        # forcing term: ft: (B, T, dof)
        ft = ddy * tau**2 - az * ( bz * (y[:,:, -d:] - y) - dy * tau)
        # regression matrix A: (T, Nb)
        psi = torch.exp(-0.5 * (xphase.unsqueeze(1) - c)**2 / sigma)
        A = xphase.unsqueeze(1) * psi / psi.sum(dim=1, keepdim=True)
        # solve per batch: A @ W_b^T = ft_b
        W = torch.zeros((B, d, Nb), device=self.dev)
        for b in range(B):
            # least squares: sol shape (Nb, dof)
            sol = torch.linalg.lstsq(A, ft[b]).solution  # (Nb, dof)
            W[b] = sol.T
        # squeeze batch if single
        if W.shape[0] == 1:
            return W.squeeze(0)
        return W


