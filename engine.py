#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import math
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from bed_reader import open_bed
import signal, sys
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# =========================== Debug controls ===========================
DEBUG_LEVEL = 0
DUMP_PREFIX: Optional[str] = None

def set_debug(level: int, dump_prefix: Optional[str] = None):
    global DEBUG_LEVEL, DUMP_PREFIX
    DEBUG_LEVEL = int(level)
    DUMP_PREFIX = dump_prefix

def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def _stat(x):
    if isinstance(x, torch.Tensor):
        x = x.detach()
        nan_cnt = torch.count_nonzero(~torch.isfinite(x)).item()
        xf = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return (float(torch.min(xf)),
                float(torch.max(xf)),
                float(torch.mean(xf)),
                float(torch.std(xf)),
                float(torch.linalg.norm(xf)),
                int(nan_cnt))
    arr = np.asarray(x)
    nan_cnt = np.count_nonzero(~np.isfinite(arr))
    xf = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return (float(np.min(xf)),
            float(np.max(xf)),
            float(np.mean(xf)),
            float(np.std(xf)),
            float(np.linalg.norm(xf)),
            int(nan_cnt))

def dbg(name: str, x=None, *, level=1, sample=5):
    if DEBUG_LEVEL < level:
        return
    hdr = f"[DBG{level}] {name}"
    if x is None:
        logging.debug(hdr); return
    try:
        if isinstance(x, torch.Tensor):
            shape = tuple(x.shape); dtype = str(x.dtype); device = str(x.device)
            s = _stat(x)
            logging.debug(f"{hdr}: tensor shape={shape} dtype={dtype} device={device} "
                          f"min={s[0]:.3e} max={s[1]:.3e} mean={s[2]:.3e} std={s[3]:.3e} "
                          f"||x||={s[4]:.3e} NaNs={s[5]}")
            if sample and x.numel() > 0 and DEBUG_LEVEL >= level+1:
                flat = x.flatten().detach().cpu().numpy()
                k = min(sample, flat.size)
                idx = np.linspace(0, flat.size-1, num=k, dtype=int)
                logging.debug(f"{hdr} sample={flat[idx]}")
        else:
            arr = np.asarray(x); shape = arr.shape; dtype = str(arr.dtype)
            s = _stat(arr)
            logging.debug(f"{hdr}: array shape={shape} dtype={dtype} "
                          f"min={s[0]:.3e} max={s[1]:.3e} mean={s[2]:.3e} std={s[3]:.3e} "
                          f"||x||={s[4]:.3e} NaNs={s[5]}")
            if sample and arr.size > 0 and DEBUG_LEVEL >= level+1:
                k = min(sample, arr.size)
                idx = np.linspace(0, arr.size-1, num=k, dtype=int)
                logging.debug(f"{hdr} sample={arr.flatten()[idx]}")
        if DUMP_PREFIX and DEBUG_LEVEL >= 3:
            n = (x.numel() if isinstance(x, torch.Tensor) else np.asarray(x).size)
            if n <= 5_000_000:
                path = f"{DUMP_PREFIX}.{name.replace(' ', '_')}.npy"
                np.save(path, _to_numpy(x))
                logging.debug(f"{hdr} dumped to {path}")
    except Exception as exc:
        logging.debug(f"{hdr} (debug failed): {exc}")

# =========================== Precision / Device ===========================
DTYPE = torch.float64

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================== Small helpers ===========================
def to_dev(x, device, dtype=DTYPE):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype, non_blocking=True)
    return torch.as_tensor(x, device=device, dtype=dtype)

def unit_np(v, eps=1e-18):
    v = np.asarray(v, float).reshape(-1)
    n = np.linalg.norm(v)
    if n <= eps:
        out = np.zeros_like(v); out[0] = 1.0
        return out
    return v / n

def geodesic_step_np(alpha, g_proj, step, theta_max=np.deg2rad(25.0), eps=1e-18):
    a = np.asarray(alpha, float).reshape(-1)
    d = -np.asarray(g_proj, float).reshape(-1)
    d = d - (a @ d) * a
    nd = float(np.linalg.norm(d))
    if nd < 1e-20:
        return a.reshape(-1, 1)
    theta = min(step * nd, float(theta_max))
    u = d / nd
    a_new = a * np.cos(theta) + u * np.sin(theta)
    return (a_new / (np.linalg.norm(a_new) + eps)).reshape(-1, 1)

def geodesic_move_np(alpha, dir_tan, theta, eps=1e-18):
    """Move along a unit tangent direction dir_tan by geodesic angle theta (radians)."""
    a = np.asarray(alpha, float).reshape(-1)
    u = np.asarray(dir_tan, float).reshape(-1)
    u = u - (a @ u) * a
    nu = np.linalg.norm(u)
    if nu <= 1e-20 or theta == 0.0:
        return a.reshape(-1)
    u = u / nu
    a_new = a*np.cos(theta) + u*np.sin(theta)
    a_new = a_new / (np.linalg.norm(a_new) + eps)
    return a_new.reshape(-1)

def tangent_basis_np(alpha, tol=1e-12):
    """Orthonormal tangent basis at alpha on S^{L-1} via Gram-Schmidt."""
    a = unit_np(alpha).reshape(-1)
    L = a.size
    T_cols = []
    for i in range(L):
        v = np.zeros(L); v[i] = 1.0
        v = v - (a @ v)*a
        for u in T_cols:
            v = v - (u @ v)*u
        nv = np.linalg.norm(v)
        if nv > 1e-10:
            T_cols.append(v / nv)
        if len(T_cols) >= L-1:
            break
    if len(T_cols) == 0:
        return np.zeros((L,0), float)
    return np.column_stack(T_cols)

def make_pair_index(L: int):
    ii, jj = [], []
    for l in range(L):
        for k in range(l, L):
            ii.append(l); jj.append(k)
    ii = torch.tensor(ii, dtype=torch.int64)
    jj = torch.tensor(jj, dtype=torch.int64)
    return ii, jj

def _parse_csv_list(arg: str):
    if arg is None:
        return None
    items = [x.strip() for x in arg.split(",") if x.strip() != ""]
    return items if len(items) > 0 else None

def _norm_name(s: str) -> str:
    return " ".join(str(s).split()).lower()

def enforce_positive_on_feature(alpha: np.ndarray, env_cols_used, feature_name: str) -> np.ndarray:
    if not feature_name:
        return alpha
    name2idx = {_norm_name(c): i for i, c in enumerate(env_cols_used)}
    key = _norm_name(feature_name)
    idx = name2idx.get(key, None)
    if idx is None:
        logging.warning("force-positive feature '%s' not found; skipping.", feature_name)
        return alpha
    a = alpha.reshape(-1)
    return (-alpha) if (a[idx] < 0.0) else alpha

# =========================== Signals ===========================
_STOP_FLAGS = {"stop": False}
def _handle_sig(signum, frame):
    logging.warning("Received signal %s. Will stop after this iteration...", signum)
    _STOP_FLAGS["stop"] = True
for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, _handle_sig)
    except Exception:
        pass

# =========================== Env type detection & flipping ===========================
def _finite_unique(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors='coerce')
    vals = vals[np.isfinite(vals)]
    return np.unique(vals.values)

def _is_integerish_array(arr: np.ndarray, tol=1e-12) -> bool:
    if arr.size == 0:
        return False
    frac = np.abs(arr - np.round(arr))
    return bool(np.all(frac <= tol))

def detect_env_is_binary_or_categorical(col: pd.Series, max_levels: int) -> Tuple[bool, str]:
    u = _finite_unique(col)
    if u.size == 2:
        return True, "binary"
    if (u.size > 1) and (u.size <= max_levels) and _is_integerish_array(u):
        return True, "categorical"
    return False, "none"

def flip_env_column_inplace(df: pd.DataFrame, colname: str):
    col = pd.to_numeric(df[colname], errors='coerce')
    mn = np.nanmin(col.values); mx = np.nanmax(col.values)
    df[colname] = (mn + mx) - col

# =========================== Data loading / whitening ===========================
def build_anchor_from_names(env_cols_in_E, anchor_names):
    if not anchor_names:
        return None
    L = len(env_cols_in_E)
    v = np.zeros(L, dtype=np.float64); found = 0
    for nm in anchor_names:
        if nm in env_cols_in_E:
            v[env_cols_in_E.index(nm)] = 1.0; found += 1
        else:
            logging.warning("Anchor column '%s' not found; ignoring.", nm)
    if found == 0:
        logging.warning("No anchor columns found; disabling anchor.")
        return None
    v /= (np.linalg.norm(v) + 1e-18)
    return v

def _standardize_and_whiten(env_df_full: pd.DataFrame,
                            env_cols_used: List[str],
                            bed_fids_sel: pd.Index):
    E_df = env_df_full.set_index('FID')
    if E_df.index.dtype != bed_fids_sel.dtype:
        E_df.index = E_df.index.astype(str)
    E_train_df = E_df.loc[bed_fids_sel].copy()
    E_train = E_train_df.drop(columns=['IID']).to_numpy(dtype=np.float64)

    # Standardize
    dbg("E before standardize (train)", E_train, level=2)
    E_mean = E_train.mean(axis=0, keepdims=True)
    E_std  = E_train.std(axis=0, ddof=1, keepdims=True)
    E_std  = np.where(E_std == 0.0, 1.0, E_std)
    Ez_train = (E_train - E_mean) / E_std
    dbg("E after standardize (train)", Ez_train, level=2)

    # Whitening (shrinkage)
    C_E = np.cov(Ez_train, rowvar=False)
    L = C_E.shape[0]
    rho = 0.10
    tau = np.trace(C_E) / float(L)
    C_shrink = (1.0 - rho) * C_E + rho * (tau * np.eye(L))
    w, V = np.linalg.eigh(C_shrink)
    w_floor = np.maximum(w, 1e-3 * np.median(w))
    inv_sqrt = V @ np.diag(1.0 / np.sqrt(w_floor)) @ V.T
    dbg("Lw_inv", inv_sqrt, level=2)

    # Apply
    E_whiten_train = Ez_train @ inv_sqrt
    E_whiten_train = np.nan_to_num(E_whiten_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    dbg("E whitened (train)", E_whiten_train, level=1)

    return E_whiten_train, inv_sqrt, E_mean, E_std

def load_real_E_and_row_idx(
    bed_prefix,
    env_file,
    num_samples,
    env_cols,
    pheno_file=None,
    pheno_name="PHENO",
    lifestyle_envs: bool = False, 
):
    dbg("load: start", level=1)
    G = open_bed(f"{bed_prefix}.bed")
    dbg("bed shape", np.array(G.shape), level=1)

    env_df = pd.read_csv(env_file, sep='\t')
    dbg("env_df cols", np.array(env_df.columns), level=2)

    pheno_df = None

    if not {'FID', 'IID'}.issubset(env_df.columns):
        raise ValueError("env_file must contain FID and IID")

    # ----- lifestyle env name handling (unchanged from your version) -----
    lifestyle_names = [
        'Age','Walked','Moderate PA','Vigorous PA','Time watching TV','Sleep duration',
        'Sleep duration res','Cooked veg','Oily fish','Non-oily fish','Processed meat','Poultry','Beef',
        'Lamb/mutton','Pork','Cheese','Salt added to food','Tea','Alcohol frequency','Smoking status','TDI'
    ]
    non_dietary_names = [
        'Age','Walked','Moderate PA','Vigorous PA','Time watching TV','Sleep duration',
        'Sleep duration res','Alcohol frequency','Smoking status','TDI'
    ]
    env_names = lifestyle_names[:]
    for var in ['age', 'gender']:
        env_names.extend([f'{l} x {var}' for l in non_dietary_names])
    env_names.append('Gender')

    if lifestyle_envs:
        # treat the env file as having columns: FID, IID, env_names...
        col_names = np.concatenate([['FID', 'IID'], env_names])
        if len(col_names) != env_df.shape[1]:
            raise ValueError(
                f"Env header length mismatch: file has {env_df.shape[1]} cols, "
                f"expected {len(col_names)}."
            )
        env_df.columns = col_names
        env_base = env_names
    else:
        # default: all columns except IDs
        env_base = [c for c in env_df.columns if c not in ('FID', 'IID')]

    # ---------------------- env_cols_used logic -----------------------
    if env_cols is None:
        env_cols_used = env_base
    else:
        missing = [c for c in env_cols if c not in env_df.columns]
        if len(missing) > 0:
            logging.warning("Missing --env-cols ignored: %s", ",".join(missing))
        env_cols_used = [c for c in env_cols if c in env_df.columns]
        if len(env_cols_used) == 0:
            raise ValueError("None of the requested --env-cols found.")
    dbg("env_cols_used", np.array(env_cols_used), level=1)

    env_df_full = env_df[['FID', 'IID'] + env_cols_used].copy()

    # ---------------------- phenotype (raw) -----------------------
    if pheno_file:
        pheno_df = pd.read_csv(pheno_file, sep=' ')
        if pheno_name not in pheno_df.columns:
            cols = list(pheno_df.columns)
            if len(cols) >= 3:
                pheno_df = pheno_df.rename(columns={cols[2]: pheno_name})
            else:
                raise ValueError(f"Phenotype column '{pheno_name}' not found.")
        pheno_df = pheno_df[['FID', 'IID', pheno_name]].copy()
        pheno_df[pheno_name] = pd.to_numeric(pheno_df[pheno_name], errors='coerce')

    # ---------------------- valid FIDs (env & optional pheno) -----------------------
    def valid_fids(df: pd.DataFrame):
        vals = df.iloc[:, 2:]
        mk = (~vals.isna()) & (vals != -9)
        return set(df.loc[mk.all(axis=1), 'FID'])

    # If pheno_file provided: intersect env + pheno
    # Else: just use valid IDs from env_df
    sets = [valid_fids(env_df_full)]
    if pheno_df is not None:
        sets.append(valid_fids(pheno_df))
    valid = set.intersection(*sets)

    bed_fids = pd.Index(open_bed(f"{bed_prefix}.bed").iid.astype(str))
    fids = np.intersect1d(bed_fids.values, np.array(list(valid)).astype(str))
    if num_samples is not None and num_samples > 0:
        fids = fids[:num_samples]
    dbg("selected FIDs count", np.array([len(fids)]), level=1)

    row_mask = np.isin(bed_fids.values, fids)
    row_idx = np.where(row_mask)[0]
    bed_fids_sel = pd.Index(bed_fids.values[row_idx], dtype=str)

    # ---------------------- Standardize + whiten E -----------------------
    E_whiten_train, inv_sqrt, E_mean, E_std = _standardize_and_whiten(
        env_df_full, env_cols_used, bed_fids_sel
    )

    return (
        G,
        row_idx.astype(np.int64),
        E_whiten_train.astype(np.float64),
        bed_fids_sel,
        pheno_df,
        env_cols_used,
        inv_sqrt,
        env_df_full,
        E_mean, E_std
    )


# =========================== Simulation (optional) ===========================
def simulate_y_from_bed(bed_prefix, row_idx, E_dev, base_seed, sigma_g, sigma_gxe, sigma_nxe,
                        col_start, col_stop, block_snps, device):
    G = open_bed(f"{bed_prefix}.bed")
    total_m = G.shape[1]
    col_stop = int(total_m if col_stop is None else min(total_m, col_stop))
    assert col_stop > col_start, "Empty SNP range"

    N = int(row_idx.size)
    M_sel = int(col_stop - col_start)
    L = int(E_dev.shape[1])

    dbg("simulate: params [N,M_sel,L]", np.array([N, M_sel, L]), level=1)

    rng = np.random.default_rng(base_seed)
    alpha = rng.standard_normal((L, 1)).astype(np.float64)
    alpha = alpha / (np.linalg.norm(alpha) + 1e-18)
    if alpha[0] < 0:
        alpha = -alpha
    alpha_true = alpha.copy()
    dbg("alpha_true init", alpha_true, level=2)
    alpha_t = to_dev(alpha.ravel(), device)

    e = (E_dev @ alpha_t).reshape(N, 1)

    beta  = to_dev(rng.standard_normal((M_sel, 1)).astype(np.float64), device) * math.sqrt(sigma_g / M_sel)
    gamma = to_dev(rng.standard_normal((M_sel, 1)).astype(np.float64), device) * math.sqrt(sigma_gxe / M_sel)

    y = torch.zeros((N, 1), device=device, dtype=DTYPE)

    n_blocks = math.ceil(M_sel / block_snps)
    with tqdm(total=n_blocks, desc="Simulating y (Torch blocks)", unit="blk", dynamic_ncols=True) as bar:
        rel = 0
        for s in range(col_start, col_stop, block_snps):
            ecol = min(s + block_snps, col_stop)
            b = ecol - s
            Xb_np = G.read(index=(row_idx, slice(s, ecol))).astype(np.float64, copy=False)
            Xb = to_dev(Xb_np, device)

            tmp  = torch.nan_to_num(Xb)
            cnt  = torch.sum(torch.isfinite(Xb), dim=0, dtype=DTYPE).clamp_min(1.0)
            sumv = torch.sum(tmp, dim=0, dtype=DTYPE)
            sum2 = torch.sum(tmp*tmp, dim=0, dtype=DTYPE)
            mean_b = sumv / cnt
            var_b  = (sum2 / cnt - mean_b.to(torch.float64)**2).clamp_min(0.0).to(DTYPE)
            std_b  = torch.where(var_b == 0, torch.ones_like(var_b), torch.sqrt(var_b))
            Xb = (tmp - mean_b) / std_b

            e = (E_dev @ alpha_t).reshape(N, 1)
            y += Xb @ beta[rel:rel+b]
            y += (Xb * e) @ gamma[rel:rel+b]
            rel += b

            bar.set_postfix({"b": b}, refresh=False)
            bar.update(1)

    sigma_e = max(1.0 - float(sigma_g) - float(sigma_gxe) - float(sigma_nxe), 1e-8)
    rng2 = np.random.default_rng(base_seed + 7)
    nxe_term = e * to_dev(rng2.standard_normal((N, 1)).astype(np.float64), device) * math.sqrt(sigma_nxe)
    y = y + nxe_term
    noise = to_dev(rng2.standard_normal((N, 1)).astype(np.float64), device) * math.sqrt(sigma_e)
    y = y + noise

    y = (y - torch.mean(y, dim=0)) / (torch.std(y, dim=0) + 1e-12)
    y_np = y.detach().cpu().numpy().reshape(-1).astype(np.float64)
    return y_np, alpha_true.astype(np.float64)

# =========================== Sigma solve (A,b) for α given C ===========================
@torch.no_grad()
def assemble_Ab_sigma_torch(a, C, ridge=1e-10, nonneg_sigma: bool = False):
    E = C["E"]; y = C["y"]
    KW = C["KW"]; Kdiag = C["Kdiag"]
    Gy = C["G_y"]; Ge = C["G_e"]
    KgxeW = C["KgxeW"]; KgxeMat = C["Kgxe_mat"]
    pi = C["pair_i"]; pj = C["pair_j"]
    N = C["N"]; B = C["B"]
    device = E.device

    e  = E @ a
    e2 = e * e

    w_pairs = a[pi] * a[pj]
    KgxeW_cast = KgxeW.to(w_pairs.dtype)
    KgxeW_a = torch.tensordot(w_pairs, KgxeW_cast, dims=([0],[0]))  # (N,B)
    invB = 1.0 / float(B)
    tr_gg = invB * torch.sum(KW * KW)
    tr_ge = invB * torch.sum(KW * KgxeW_a)
    tr_ee = invB * torch.sum(KgxeW_a * KgxeW_a)

    tr_KK_nxe     = torch.sum(e2 * e2)
    tr_KK_g_nxe   = a @ (Ge @ a)
    tr_KK_gxe_nxe = torch.sum((e2 * e2) * Kdiag)
    tr_K_nxe      = torch.sum(e2)
    tr_K          = C["tr_K"]
    tr_K_gxe      = a @ (KgxeMat @ a)

    yKy1 = C["yKy1"]
    yKy2 = a @ (Gy @ a)
    y2   = y @ y

    A = torch.stack([
        torch.stack([tr_gg,       tr_ge,         tr_KK_g_nxe,  tr_K]),
        torch.stack([tr_ge,       tr_ee,         tr_KK_gxe_nxe,tr_K_gxe]),
        torch.stack([tr_KK_g_nxe, tr_KK_gxe_nxe, tr_KK_nxe,    tr_K_nxe]),
        torch.stack([tr_K,        tr_K_gxe,      tr_K_nxe,     torch.tensor(float(N), device=device, dtype=DTYPE)])
    ]).to(device=device, dtype=DTYPE)

    condA = torch.linalg.cond(A)
    if torch.isfinite(condA) and condA > 1e8:
        ridge = max(ridge, 1e-6)
    
    if ridge > 0.0:
        A = A + ridge * torch.eye(4, device=device, dtype=DTYPE)

    b = torch.stack([yKy1, yKy2, torch.sum(y * e2 * y), y2]).to(device=device, dtype=DTYPE)
    sigma_hat = _solve_sigma_linear_or_nnls(A, b, nonneg=bool(nonneg_sigma or C.get("nonneg_sigma", False)))
    return A, b, sigma_hat

# =========================== Objective & analytic gradient ===========================
@torch.no_grad()
def obj_and_grad_alpha_manual(a_np, C, sig_tuple):
    device = C["E"].device
    E = C["E"]; y = C["y"]
    KW = C["KW"]; Kdiag = C["Kdiag"]
    Gy = C["G_y"]; Ge = C["G_e"]
    KgxeW = C["KgxeW"]; KgxeMat = C["Kgxe_mat"]
    pi = C["pair_i"]; pj = C["pair_j"]
    B = C["B"]; N = C["N"]

    sigma_g, sigma_gxe, sigma_nxe = [float(x) for x in sig_tuple]
    a = to_dev(a_np.reshape(-1), device)  # (L,)

    e  = E @ a            # (N,)
    e2 = e * e
    y2 = y * y

    w_pairs = a[pi] * a[pj]
    KgxeW_cast = KgxeW.to(w_pairs.dtype)
    KgxeW_a = torch.tensordot(w_pairs, KgxeW_cast, dims=([0],[0]))
    invB = 1.0 / float(B)
    tr_gg = invB * torch.sum(KW * KW)
    tr_ge = invB * torch.sum(KW * KgxeW_a)
    tr_ee = invB * torch.sum(KgxeW_a * KgxeW_a)

    t_gxe = a @ (Gy @ a)
    t_nxe = torch.sum(e2 * y2)
    tr_KK_nxe     = torch.sum(e2 * e2)
    tr_K_g_nxe    = torch.sum(e2 * Kdiag)
    tr_KK_g_nxe   = a @ (Ge @ a)
    tr_KK_gxe_nxe = torch.sum((e2 * e2) * Kdiag)

    obj = (1.0/float(N)) * (
        -(2.0 * sigma_gxe) * t_gxe
        + (sigma_gxe ** 2) * tr_ee
        + 2.0 * sigma_gxe * sigma_g * tr_ge
        -(2.0 * sigma_nxe) * t_nxe
        + (sigma_nxe ** 2) * tr_KK_nxe
        + 2.0 * sigma_g * sigma_nxe * tr_K_g_nxe
        + 2.0 * sigma_gxe * sigma_nxe * tr_KK_gxe_nxe
    )

    g = torch.zeros_like(a)
    g += (-4.0 * sigma_gxe) * (Gy @ a)

    KgxeW_a = torch.tensordot(w_pairs, KgxeW_cast, dims=([0],[0]))
    s_ee_vec = 2.0 * invB * torch.tensordot(KgxeW_a, KgxeW_cast, dims=([0,1],[1,2]))
    s_ge_vec =       invB * torch.tensordot(KW.to(KgxeW_cast.dtype), KgxeW_cast, dims=([0,1],[1,2]))
    coeff = (sigma_gxe**2) * s_ee_vec + (2.0 * sigma_gxe * sigma_g) * s_ge_vec

    ai = a[pi]; aj = a[pj]
    g.index_add_(0, pi, coeff * aj)
    g.index_add_(0, pj, coeff * ai)

    g += (-4.0 * sigma_nxe) * (E.T @ (y2 * e))
    g += (sigma_nxe ** 2) * (4.0 * (E.T @ (e * e * e)))
    g += (4.0 * sigma_g * sigma_nxe) * (E.T @ (Kdiag * e))
    g += (8.0 * sigma_gxe * sigma_nxe) * (E.T @ (Kdiag * e * e * e))

    g = (1.0 / float(N)) * g
    grad_np = g.detach().cpu().numpy().reshape(-1, 1)
    obj_float = float(obj.item())
    return obj_float, grad_np

# =========================== Optimizer ===========================
def optimize_alpha_no_backtracking(C, alpha0_np, iters=200, step=0.2, theta_max_deg=25.0,
                                   log_every=10, orient_guard=None,
                                   tol_angle_deg=0.05, tol_obj_rel=1e-6,
                                   patience=5, min_iters=20):

    alpha = unit_np(alpha0_np).reshape(-1, 1)
    theta_max = np.deg2rad(theta_max_deg)

    best_val = None
    prev_val = None
    quiet_streak = 0
    eps_clip = 1e-12
    tol_angle = np.deg2rad(tol_angle_deg)

    # --- L1 schedule params ---
    tau0 = float(C.get("l1_tau", 0.0))
    tau_min = float(C.get("l1_tau_min", 0.0))
    anneal = str(C.get("l1_anneal", "none")).lower()

    def _tau_eff(it_idx: int):
        if tau0 <= 0.0:
            return 0.0
        if anneal == "none":
            return tau0
        t = it_idx / float(max(1, iters))
        if anneal == "linear":
            # τ(t) = τ0 + (τ_min - τ0) * t
            return float(tau0 + (tau_min - tau0) * t)
        if anneal == "cosine":
            # τ(t) = τ_min + 0.5*(τ0 - τ_min)*(1 + cos(pi * t))  (high -> low)
            return float(tau_min + 0.5 * (tau0 - tau_min) * (1.0 + math.cos(math.pi * t)))
        # fallback
        return tau0

    history = []
    with tqdm(total=iters, desc="Optimizing α (Torch, no-autograd)", unit="iter", dynamic_ncols=True) as bar:
        for it in range(1, iters + 1):
            if _STOP_FLAGS["stop"]:
                break

            with torch.no_grad():
                a_t = to_dev(alpha.ravel(), C["E"].device)
                _, _, sigma4 = assemble_Ab_sigma_torch(a_t, C, ridge=1e-10, nonneg_sigma=C.get("nonneg_sigma", False))
                sigma_vec = sigma4.detach().cpu().numpy().tolist()
                sigma_g, sigma_gxe, sigma_nxe, sigma_e = [float(x) for x in sigma_vec]

            val, grad = obj_and_grad_alpha_manual(alpha.ravel(), C, (sigma_g, sigma_gxe, sigma_nxe))
            if best_val is None or val < best_val:
                best_val = val

            a = alpha.reshape(-1, 1)
            dot_ag = float(a.T @ grad)
            g_proj = grad - (dot_ag * a)

            # simple diag precond (safe if missing)
            try:
                Ge_diag   = np.diag(_to_numpy(C["G_e"])).reshape(-1, 1)
                Kgxe_diag = np.diag(_to_numpy(C["Kgxe_mat"])).reshape(-1, 1)
                precond = 1.0 / (1e-6 + np.sqrt(Ge_diag + Kgxe_diag))
                precond = precond / (np.linalg.norm(precond) + 1e-18)
                g_proj = g_proj * precond
            except Exception:
                pass

            alpha_candidate = geodesic_step_np(alpha, g_proj, step=step, theta_max=theta_max)

            # --- spherical soft-threshold with annealed τ ---
            tau_eff = _tau_eff(it)
            if tau_eff > 0.0:
                a_np = _soft_threshold_unit(alpha_candidate.reshape(-1), tau_eff)
                alpha_candidate = a_np.reshape(-1, 1)

            a_prev = alpha.reshape(-1)
            a_new  = alpha_candidate.reshape(-1)
            dot = float(np.clip(a_prev @ a_new, -1.0 + eps_clip, 1.0 - eps_clip))
            ang = math.acos(dot)

            rel_impr = np.inf if prev_val is None else abs(prev_val - val) / max(1.0, abs(prev_val))
            quiet_streak = (quiet_streak + 1) if (it >= min_iters and ang < tol_angle and rel_impr < tol_obj_rel) else 0

            alpha = alpha_candidate
            prev_val = val

            if (it % log_every) == 0 or it == 1 or it == iters or quiet_streak == patience:
                history.append((it, val, sigma_g, sigma_gxe, sigma_nxe, sigma_e, tau_eff))

            if C["E"].device.type == "cuda" and (it % 50 == 0):
                torch.cuda.empty_cache()

            bar.set_postfix({
                "obj": f"{val:.3e}",
                "g":   f"{sigma_g:.3e}",
                "gxe": f"{sigma_gxe:.3e}",
                "nxe": f"{sigma_nxe:.3e}",
                "e":   f"{sigma_e:.3e}",
                "τ":    f"{tau_eff:.2e}",
                "ang°": f"{np.rad2deg(ang):.3e}",
                "qs":   f"{quiet_streak}",
            }, refresh=False)
            bar.update(1)
            if quiet_streak >= patience:
                break

    return alpha, np.array([sigma_g, sigma_gxe, sigma_nxe, sigma_e], float), history

# =========================== Full precompute for α stage ===========================
def precompute_from_bed(
    bed_prefix,
    row_idx,
    E_dev,
    y_dev,
    B=64,
    W=None,
    seed=0,
    block_snps=2048,
    col_start=0,
    col_stop=None,
    log_every_blocks=25,
    device=None,
    pair_chunk_size=1,
    kgxew_fp32=True,
):
    G = open_bed(f"{bed_prefix}.bed")
    total_m = G.shape[1]
    col_stop = int(total_m if col_stop is None else min(total_m, col_stop))
    assert col_stop > col_start, "No SNPs selected"

    E = E_dev
    y = y_dev
    N = int(row_idx.size)
    L = int(E.shape[1])
    M = int(col_stop - col_start)
    device = device or E.device

    if W is None:
        rng = np.random.default_rng(seed)
        W = rng.standard_normal(size=(B, N)).astype(np.float64)
    W = to_dev(W, device)
    Wt = W.transpose(0, 1)  # (N,B)

    pair_i, pair_j = make_pair_index(L)
    pair_i = pair_i.to(device)
    pair_j = pair_j.to(device)
    P = int(pair_i.numel())

    KW_sum     = torch.zeros((N, B), device=device, dtype=DTYPE)
    _dtype_k = torch.float32 if kgxew_fp32 else DTYPE
    KgxeW_sum  = torch.zeros((P, N, B), device=device, dtype=_dtype_k)
    G_y_accum  = torch.zeros((L, L), device=device, dtype=DTYPE)
    G_e_accum  = torch.zeros((L, L), device=device, dtype=DTYPE)
    s_rows     = torch.zeros((N,),   device=device, dtype=DTYPE)
    yKy1_accum = torch.tensor(0.0, device=device, dtype=DTYPE)

    Ey = (E * y[:, None]).to(device)

    n_blocks = math.ceil(M / block_snps)
    with tqdm(total=n_blocks, desc=f"Precompute (Torch, M={M})", unit="blk", dynamic_ncols=True) as bar:
        rel = 0
        for bi, s in enumerate(range(col_start, col_stop, block_snps), start=1):
            ecol = min(s + block_snps, col_stop)
            b = ecol - s

            Xb_np = G.read(index=(row_idx, slice(s, ecol))).astype(np.float64, copy=False)
            Xb = to_dev(Xb_np, device)                         # (N,b)
            tmp  = torch.nan_to_num(Xb)
            cnt  = torch.sum(torch.isfinite(Xb), dim=0, dtype=DTYPE).clamp_min(1.0)
            sumv = torch.sum(tmp, dim=0, dtype=DTYPE)
            sum2 = torch.sum(tmp * tmp, dim=0, dtype=DTYPE)
            mean_b = sumv / cnt
            var_b  = (sum2 / cnt - mean_b.to(torch.float64)**2).clamp_min(0.0).to(DTYPE)
            std_b  = torch.where(var_b == 0, torch.ones_like(var_b), torch.sqrt(var_b))
            Xb = (tmp - mean_b) / std_b                        # (N,b)

            s_rows += torch.sum(Xb * Xb, dim=1)
            KW_sum += Xb @ (Xb.T @ Wt)

            B_b = Xb.T @ Ey
            G_y_accum += B_b.T @ B_b

            XE_b = Xb.T @ E
            G_e_accum += XE_b.T @ XE_b

            xty_b = Xb.T @ y
            yKy1_accum = yKy1_accum + (xty_b @ xty_b)

            XbT = Xb.T  # (b,N)

            P = int(pair_i.numel())
            pcs = max(1, int(pair_chunk_size))
            for p0 in range(0, P, pcs):
                p1 = min(P, p0 + pcs)
                idx_i = pair_i[p0:p1]
                idx_j = pair_j[p0:p1]

                for off in range(idx_i.numel()):
                    l = int(idx_i[off].item()); k = int(idx_j[off].item())

                    Zl = E[:, l:l+1] * Xb
                    ZkT_W = (XbT * E[:, k].unsqueeze(0)) @ Wt
                    part = Zl @ ZkT_W
                    if l != k:
                        Zk = E[:, k:k+1] * Xb
                        ZlT_W = (XbT * E[:, l].unsqueeze(0)) @ Wt
                        part = part + (Zk @ ZlT_W)

                    if KgxeW_sum.dtype != part.dtype:
                        part = part.to(KgxeW_sum.dtype)
                    KgxeW_sum[p0 + off] += part

                    del Zl, ZkT_W, part
                    if l != k:
                        del Zk, ZlT_W

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            del Xb, XbT, tmp
            if device.type == "cuda":
                torch.cuda.empty_cache()

            rel += b
            if (bi % max(1, log_every_blocks)) == 0:
                bar.set_postfix({"blk": f"{bi}/{n_blocks}"}, refresh=False)
            bar.update(1)

    invM = 1.0 / float(M)
    KW       = KW_sum * invM
    KgxeW    = KgxeW_sum * invM
    G_y      = G_y_accum * invM
    G_e      = G_e_accum * invM
    Kdiag    = s_rows * invM
    tr_K     = torch.sum(Kdiag)
    yKy1     = yKy1_accum * invM
    Kgxe_mat = E.T @ (E * Kdiag[:, None])

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "N": N, "M": M, "L": L, "B": int(W.shape[0]),
        "W": W, "KW": KW, "KgxeW": KgxeW,
        "pair_i": pair_i, "pair_j": pair_j,
        "G_y": G_y, "G_e": G_e,
        "Kdiag": Kdiag, "yKy1": yKy1, "tr_K": tr_K,
        "Kgxe_mat": Kgxe_mat,
        "E": E, "y": y
    }

# =========================== Shared precompute for linear HE ===========================
@torch.no_grad()
def precompute_shared_for_he(
    bed_prefix,
    row_idx,
    E_dev,
    y_dev,
    B=64,
    W=None,
    seed=0,
    block_snps=2048,
    col_start=0,
    col_stop=None,
    device=None,
):
    """
    Build shared pieces for linear-time HE:
      KW_avg (N,B), Kdiag (N,), yKy1 (scalar)
    """
    G = open_bed(f"{bed_prefix}.bed")
    total_m = G.shape[1]
    col_stop = int(total_m if col_stop is None else min(total_m, col_stop))
    assert col_stop > col_start, "No SNPs selected"

    device = device or E_dev.device
    N, L = int(E_dev.shape[0]), int(E_dev.shape[1])

    if W is None:
        rng = np.random.default_rng(seed)
        W = rng.standard_normal(size=(B, N)).astype(np.float64)
    W = to_dev(W, device)
    Wt = W.transpose(0, 1)  # (N,B)

    KW_sum     = torch.zeros((N, B), device=device, dtype=DTYPE)
    s_rows     = torch.zeros((N,),   device=device, dtype=DTYPE)
    yKy1_accum = torch.tensor(0.0,   device=device, dtype=DTYPE)

    y = y_dev.reshape(N)
    n_blocks = math.ceil((col_stop - col_start) / block_snps)
    with tqdm(total=n_blocks, desc="HE shared precompute", unit="blk", dynamic_ncols=True) as bar:
        for s in range(col_start, col_stop, block_snps):
            ecol = min(s + block_snps, col_stop)
            b = ecol - s

            Xb_np = G.read(index=(row_idx, slice(s, ecol))).astype(np.float64, copy=False)
            Xb = to_dev(Xb_np, device)                         # (N,b)
            tmp  = torch.nan_to_num(Xb)
            cnt  = torch.sum(torch.isfinite(Xb), dim=0, dtype=DTYPE).clamp_min(1.0)
            sumv = torch.sum(tmp, dim=0, dtype=DTYPE)
            sum2 = torch.sum(tmp * tmp, dim=0, dtype=DTYPE)
            mean_b = sumv / cnt
            var_b  = (sum2 / cnt - mean_b.to(torch.float64)**2).clamp_min(0.0).to(DTYPE)
            std_b  = torch.where(var_b == 0, torch.ones_like(var_b), torch.sqrt(var_b))
            Xb = (tmp - mean_b) / std_b                        # (N,b)

            s_rows += torch.sum(Xb * Xb, dim=1)                # Kdiag
            KW_sum += Xb @ (Xb.T @ Wt)                         # for tr_gg and tr_ge

            # yKy1
            xty_b = Xb.T @ y
            yKy1_accum = yKy1_accum + (xty_b @ xty_b)

            del Xb, tmp
            if device.type == "cuda": torch.cuda.empty_cache()
            bar.update(1)

    M = float(col_stop - col_start)
    invM = 1.0 / M
    KW_avg = KW_sum * invM
    Kdiag  = s_rows * invM
    yKy1   = yKy1_accum * invM

    return {
        "KW_avg": KW_avg,
        "Kdiag":  Kdiag,
        "yKy1":   yKy1,
        "W":      W,
        "Wt":     Wt,
        "N":      N,
        "L":      L,
        "M":      M,
        "E":      E_dev,
        "y":      y,
        "bed_prefix": bed_prefix,
        "row_idx": row_idx,
        "block_snps": block_snps,
        "col_start": col_start,
        "col_stop":  col_stop,
        "device":    device,
    }

# =========================== Asymptotic SE (fixed α) ===========================
@torch.no_grad()
def asymptotic_sigma_se_over_variant_blocks(
    bed_prefix: str,
    row_idx: np.ndarray,
    E_dev: torch.Tensor,
    y_dev: torch.Tensor,
    alpha_hat: np.ndarray,
    *,
    B: int = 64,
    W: Optional[torch.Tensor] = None,
    seed: int = 0,
    block_snps: int = 2048,
    col_start: int = 0,
    col_stop: Optional[int] = None,
    device: Optional[torch.device] = None,
    store_dtype: torch.dtype = torch.float32,
    nonneg_sigma: bool = False,
):
    """
    Asymptotic (sandwich/Godambe) SE for [σ_g, σ_gxe, σ_nxe, σ_e], holding α fixed.
    Two streamed passes: (1) build MoM pieces; (2) build S (score cov pieces).
    """
    device = device or (E_dev.device if isinstance(E_dev, torch.Tensor) else torch.device("cpu"))
    G = open_bed(f"{bed_prefix}.bed")
    total_m = G.shape[1]
    col_stop = int(total_m if col_stop is None else min(total_m, col_stop))
    assert col_stop > col_start, "No SNPs selected"

    N = int(E_dev.shape[0])
    alpha = to_dev(alpha_hat.reshape(-1), device).to(DTYPE)          # (L,)
    e  = (E_dev @ alpha).reshape(N)                                  # (N,)
    e2 = e * e
    y  = y_dev.reshape(N)

    if W is None:
        rng = np.random.default_rng(seed)
        W_np = rng.standard_normal(size=(B, N)).astype(np.float64)
        W = to_dev(W_np, device)
    Wt = W.transpose(0, 1)  # (N,B)
    invB = 1.0 / float(B)

    # ---------- first streamed pass ----------
    y2 = torch.sum(y * y)
    tr_KK_nxe = torch.sum(e2 * e2)
    tr_K_nxe  = torch.sum(e2)

    Sg  = torch.zeros((N, B), device=device, dtype=DTYPE)   # M * (K_g W^T)
    Sge = torch.zeros((N, B), device=device, dtype=DTYPE)   # M * (K_ge W^T)

    yKy1_total = torch.tensor(0.0, device=device, dtype=DTYPE)
    Gea_total  = torch.tensor(0.0, device=device, dtype=DTYPE)
    Gya_total  = torch.tensor(0.0, device=device, dtype=DTYPE)
    tK_total   = torch.tensor(0.0, device=device, dtype=DTYPE)
    s1_total   = torch.tensor(0.0, device=device, dtype=DTYPE)
    s2_total   = torch.tensor(0.0, device=device, dtype=DTYPE)
    M_total    = 0

    for s in range(int(col_start), int(col_stop), int(block_snps)):
        ecol = min(s + block_snps, int(col_stop))
        b = ecol - s
        Xb_np = G.read(index=(row_idx, slice(s, ecol))).astype(np.float64, copy=False)
        Xb = to_dev(Xb_np, device)  # (N,b)

        tmp  = torch.nan_to_num(Xb)
        cnt  = torch.sum(torch.isfinite(Xb), dim=0, dtype=DTYPE).clamp_min(1.0)
        sumv = torch.sum(tmp, dim=0, dtype=DTYPE)
        sum2 = torch.sum(tmp * tmp, dim=0, dtype=DTYPE)
        mean_b = sumv / cnt
        var_b  = (sum2 / cnt - mean_b.to(torch.float64)**2).clamp_min(0.0).to(DTYPE)
        std_b  = torch.where(var_b == 0, torch.ones_like(var_b), torch.sqrt(var_b))
        Xb = (tmp - mean_b) / std_b  # (N,b)

        XbT = Xb.T
        Xb_e = Xb * e.unsqueeze(1)                   # (N,b)

        T_g  = Xb @ (XbT @ Wt)                       # (N,B)  accumulates M * K_g W^T
        T_ge = Xb_e @ ((Xb_e.T) @ Wt)                # (N,B)  accumulates M * K_ge W^T
        Sg  += T_g
        Sge += T_ge

        xty = XbT @ y
        yKy1_total += torch.dot(xty, xty)
        t = XbT @ e
        Gea_total  += torch.dot(t, t)

        svec = XbT @ (e * y)
        Gya_total  += torch.dot(svec, svec)

        kdiag_vec = torch.sum(Xb * Xb, dim=1)         # (N,)
        tK_total   += torch.sum(kdiag_vec)
        s1_total   += torch.sum(kdiag_vec * e2)
        s2_total   += torch.sum(kdiag_vec * e2 * e2)
        M_total    += b

        del Xb, XbT, Xb_e, T_g, T_ge, xty, t, svec, kdiag_vec
        if device.type == "cuda":
            torch.cuda.empty_cache()

    KW_g  = Sg  / float(M_total)          # (N,B) == K_g W^T
    KW_ge = Sge / float(M_total)          # (N,B) == K_ge W^T
    KW_nx = e2.unsqueeze(1) * Wt          # (N,B) == K_nxe W^T

    tr_gg = invB * torch.sum(KW_g * KW_g)
    tr_ge = invB * torch.sum(KW_g * KW_ge)
    tr_ee = invB * torch.sum(KW_ge * KW_ge)

    tr_K          = tK_total  / float(M_total)
    tr_K_gxe      = s1_total  / float(M_total)
    tr_KK_gxe_nxe = s2_total  / float(M_total)
    Gea_avg       = Gea_total / float(M_total)
    yKy1          = yKy1_total / float(M_total)
    yKy2          = Gya_total  / float(M_total)

    A = torch.stack([
        torch.stack([tr_gg,       tr_ge,         Gea_avg,      tr_K]),
        torch.stack([tr_ge,       tr_ee,         tr_KK_gxe_nxe, tr_K_gxe]),
        torch.stack([Gea_avg,     tr_KK_gxe_nxe, tr_KK_nxe,     tr_K_nxe]),
        torch.stack([tr_K,        tr_K_gxe,      tr_K_nxe,      torch.tensor(float(N), device=device, dtype=DTYPE)]),
    ])
    b_vec = torch.stack([yKy1, yKy2, torch.sum(y * e2 * y), torch.sum(y * y)])
    sigma4 = _solve_sigma_linear_or_nnls(A, b_vec, nonneg=bool(nonneg_sigma))
    sigma_full = sigma4.detach().cpu().numpy()

    # ---------- second streamed pass builds S ----------
    Kg_KWg  = torch.zeros_like(KW_g)
    Kg_KWge = torch.zeros_like(KW_g)
    Kg_KWnx = torch.zeros_like(KW_g)
    Kge_KWg  = torch.zeros_like(KW_ge)
    Kge_KWge = torch.zeros_like(KW_ge)
    Kge_KWnx = torch.zeros_like(KW_ge)

    G2 = open_bed(f"{bed_prefix}.bed")
    for s in range(int(col_start), int(col_stop), int(block_snps)):
        ecol = min(s + block_snps, int(col_stop))
        b = ecol - s
        Xb_np = G2.read(index=(row_idx, slice(s, ecol))).astype(np.float64, copy=False)
        Xb = to_dev(Xb_np, device)  # (N,b)

        tmp  = torch.nan_to_num(Xb)
        cnt  = torch.sum(torch.isfinite(Xb), dim=0, dtype=DTYPE).clamp_min(1.0)
        sumv = torch.sum(tmp, dim=0, dtype=DTYPE)
        sum2 = torch.sum(tmp * tmp, dim=0, dtype=DTYPE)
        mean_b = sumv / cnt
        var_b  = (sum2 / cnt - mean_b.to(torch.float64)**2).clamp_min(0.0).to(DTYPE)
        std_b  = torch.where(var_b == 0, torch.ones_like(var_b), torch.sqrt(var_b))
        Xb = (tmp - mean_b) / std_b

        XbT  = Xb.T                                 # (b,N)
        Xb_e = Xb * e.unsqueeze(1)                  # (N,b)

        # Kg on KW_*
        R = XbT @ KW_g;   Kg_KWg  += Xb @ R
        R = XbT @ KW_ge;  Kg_KWge += Xb @ R
        R = XbT @ KW_nx;  Kg_KWnx += Xb @ R

        # Kge on KW_*
        Re = (Xb_e.T) @ KW_g;   Kge_KWg  += Xb_e @ Re
        Re = (Xb_e.T) @ KW_ge;  Kge_KWge += Xb_e @ Re
        Re = (Xb_e.T) @ KW_nx;  Kge_KWnx += Xb_e @ Re

        del Xb, XbT, Xb_e, R, Re
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scaleM = 1.0 / float(M_total)
    Kg_KWg  *= scaleM
    Kg_KWge *= scaleM
    Kg_KWnx *= scaleM
    Kge_KWg  *= scaleM
    Kge_KWge *= scaleM
    Kge_KWnx *= scaleM

    sig_g, sig_ge, sig_nx, sig_e = sigma4

    Kg_Wt  = KW_g
    Kge_Wt = KW_ge
    Knx_Wt = e2.unsqueeze(1) * Wt
    I_Wt   = Wt

    Kg_on_KWg  = Kg_KWg
    Kge_on_KWg = Kge_KWg
    Knx_on_KWg = e2.unsqueeze(1) * KW_g
    I_on_KWg   = KW_g

    Kg_on_KWge  = Kg_KWge
    Kge_on_KWge = Kge_KWge
    Knx_on_KWge = e2.unsqueeze(1) * KW_ge
    I_on_KWge   = KW_ge

    Kg_on_KWnx  = Kg_KWnx
    Kge_on_KWnx = Kge_KWnx
    Knx_on_KWnx = e2.unsqueeze(1) * KW_nx
    I_on_KWnx   = KW_nx

    def compose_KSigmaWt(Km_Wt, Km_KWg, Km_KWge, Km_KWnx):
        return sig_e * Km_Wt + sig_g * Km_KWg + sig_ge * Km_KWge + sig_nx * Km_KWnx

    Kg_SigWt  = compose_KSigmaWt(Kg_Wt,  Kg_on_KWg,  Kg_on_KWge,  Kg_on_KWnx)
    Kge_SigWt = compose_KSigmaWt(Kge_Wt, Kge_on_KWg, Kge_on_KWge, Kge_on_KWnx)
    Knx_SigWt = compose_KSigmaWt(Knx_Wt, Knx_on_KWg, Knx_on_KWge, Knx_on_KWnx)
    I_SigWt   = compose_KSigmaWt(I_Wt,   I_on_KWg,   I_on_KWge,   I_on_KWnx)

    mats = [Kg_SigWt, Kge_SigWt, Knx_SigWt, I_SigWt]
    S = torch.zeros((4, 4), device=device, dtype=DTYPE)
    for i in range(4):
        for j in range(i, 4):
            Sij = 2.0 * invB * torch.sum(mats[i] * mats[j])
            S[i, j] = Sij
            S[j, i] = Sij

    A_np = A.detach().cpu().numpy().astype(np.float64)
    S_np = S.detach().cpu().numpy().astype(np.float64)
    Ainv = np.linalg.inv(A_np)
    asymp_cov = Ainv @ S_np @ Ainv.T
    asymp_se  = np.sqrt(np.clip(np.diag(asymp_cov), 0.0, np.inf))

    return {
        "sigma_full": sigma_full,      # [sigma_g, sigma_gxe, sigma_nxe, sigma_e]
        "asymp_cov":  asymp_cov,
        "asymp_se":   asymp_se,        # (4,)
        "A":          A_np,
        "S":          S_np,
        "B":          B,
    }

# =========================== Δ-method helpers (probe-splitting) ===========================
def _sigma_for_alpha_np(C, alpha_np):
    a_t = to_dev(unit_np(alpha_np).reshape(-1), C["E"].device)
    _, _, sig = assemble_Ab_sigma_torch(a_t, C, ridge=1e-10, nonneg_sigma=C.get("nonneg_sigma", False))
    return sig.detach().cpu().numpy().reshape(-1)

def _grad_tangent_coords(C, alpha_np, sig_tuple, T):
    a = unit_np(alpha_np).reshape(-1,1)
    _, g = obj_and_grad_alpha_manual(a.ravel(), C, sig_tuple)
    g_proj = g - (a.T @ g) * a
    if T.shape[1] == 0:
        return np.zeros((0,1), float)
    return (T.T @ g_proj).reshape(-1,1)

def _hessian_tangent_numeric(C, alpha_np, sig_tuple, T, theta=1e-3, ridge=1e-8):
    d = T.shape[1]
    if d == 0:
        return np.zeros((0,0), float)
    g0 = _grad_tangent_coords(C, alpha_np, sig_tuple, T).reshape(-1)
    H = np.zeros((d,d), float)
    for i in range(d):
        a_eps = geodesic_move_np(alpha_np, T[:,i], theta)
        g_eps = _grad_tangent_coords(C, a_eps, sig_tuple, T).reshape(-1)
        H[:, i] = (g_eps - g0) / float(theta)
    lam = ridge * (np.trace(H)/max(1,d) + 1.0)
    H = H + lam*np.eye(d)
    return 0.5*(H+H.T)

def _jacobian_sigma_wrt_alpha(C, alpha_np, T, theta=1e-3):
    d = T.shape[1]
    if d == 0:
        return np.zeros((4,0), float)
    sig0 = _sigma_for_alpha_np(C, alpha_np)
    J = np.zeros((4, d), float)
    for i in range(d):
        a_eps = geodesic_move_np(alpha_np, T[:,i], theta)
        sig_eps = _sigma_for_alpha_np(C, a_eps)
        J[:, i] = (sig_eps - sig0) / float(theta)
    return J

def _apply_probe_weights_C(C, r_vec):
    device = C["E"].device
    r = to_dev(np.asarray(r_vec, float).reshape(1, -1), device)
    KW_r   = C["KW"]   * r                   # (N,B)
    KgxeW_r= C["KgxeW"]* r.reshape(1,1,-1)  # (P,N,B)
    C2 = dict(C)
    C2["KW"] = KW_r
    C2["KgxeW"] = KgxeW_r
    return C2

def _apply_probe_subset_C(C, idx):
    """Slice along B (probe) dimension."""
    idx = np.asarray(idx, int).reshape(-1)
    C2 = dict(C)
    C2["KW"]   = C["KW"][:, idx]
    C2["KgxeW"] = C["KgxeW"][:, :, idx]
    C2["B"]    = int(idx.size)
    return C2

# =========================== Δ-method (PROBE SPLITTING) ===========================
def delta_method_sigma_alpha_correction_probe_split(
    C, alpha_hat_np, ase, *,
    probe_boot_reps=128,
    theta_fd=5e-3,
    hess_ridge=1e-8,
    seed=12345,
):
    rng = np.random.default_rng(seed)

    alpha_hat = unit_np(alpha_hat_np).reshape(-1)
    L = alpha_hat.size
    T = tangent_basis_np(alpha_hat)                  # (L, d)
    d = T.shape[1]

    # σ at α̂ (tuple for gradient)
    sig_hat = _sigma_for_alpha_np(C, alpha_hat)
    sig_tuple = (float(sig_hat[0]), float(sig_hat[1]), float(sig_hat[2]))

    # Jacobian and Hessian
    J = _jacobian_sigma_wrt_alpha(C, alpha_hat, T, theta=theta_fd)   # (4, d)
    H = _hessian_tangent_numeric(C, alpha_hat, sig_tuple, T, theta=theta_fd, ridge=hess_ridge)

    # Guard H: make SPD
    try:
        np.linalg.cholesky(H + 1e-18*np.eye(d))
    except np.linalg.LinAlgError:
        tr = float(np.trace(H)) if d > 0 else 1.0
        lam = max(hess_ridge, 1e-6 * (tr / max(1, d) + 1.0))
        logging.info("[Δ-method] Increasing Hessian ridge to %.3e (base=%.3e)", lam, hess_ridge)
        H = H + lam * np.eye(d)

    if d == 0:
        cov_sigma_from_alpha = np.zeros((4,4), float)
        cov_fixed = np.asarray(ase["asymp_cov"], float)
        cov_total = cov_fixed.copy()
        se_fixed  = np.sqrt(np.clip(np.diag(cov_fixed), 0.0, np.inf))
        se_delta  = np.zeros_like(se_fixed)
        se_total  = np.sqrt(np.clip(np.diag(cov_total), 0.0, np.inf))
        return {
            "T": T,
            "J_sigma_alphaT": J,
            "H_tangent": H,
            "cov_alpha_tangent": np.zeros((0,0)),
            "cov_sigma_from_alpha": cov_sigma_from_alpha,
            "cov_sigma_fixed": cov_fixed,
            "cov_sigma_total": cov_total,
            "se_fixed": se_fixed,
            "se_delta": se_delta,
            "se_total": se_total,
        }

    # Ambient and tangent gradients at α̂
    def _ambient_grad(Cuse):
        _, g_np = obj_and_grad_alpha_manual(alpha_hat, Cuse, sig_tuple)
        return g_np.reshape(-1)
    def _proj_tan(a, g):
        return g - (a.dot(g)) * a

    g0_amb = _ambient_grad(C)
    g0_tan = T.T @ _proj_tan(alpha_hat, g0_amb)

    # ---- (i) Gaussian-multiplier bootstrap over probes (no re-stream)
    B = int(C["B"])
    G = np.zeros((d, d), float)
    cnt = 0
    if B >= 1:
        for _ in range(int(max(2, probe_boot_reps))):
            r = rng.normal(size=B)
            Cb = _apply_probe_weights_C(C, r)
            g_amb = _ambient_grad(Cb)
            g_tan = T.T @ _proj_tan(alpha_hat, g_amb)
            dg = g_tan - g0_tan
            G += np.outer(dg, dg)
            cnt += 1
    Sigma_boot = (G / max(1, cnt - 1)) if cnt >= 2 else np.zeros((d,d), float)

    # ---- (ii) Split-half across probes
    if B >= 4:
        idx_all = np.arange(B, dtype=int)
        rng.shuffle(idx_all)
        idx_A = idx_all[:B//2]
        idx_B = idx_all[B//2:]
        CA = _apply_probe_subset_C(C, idx_A)
        CB = _apply_probe_subset_C(C, idx_B)
        gA_tan = T.T @ _proj_tan(alpha_hat, _ambient_grad(CA))
        gB_tan = T.T @ _proj_tan(alpha_hat, _ambient_grad(CB))
        dg = (gA_tan - gB_tan).reshape(-1)
        # Var(g_full) ≈ Var(gA - gB)/4 for independent halves
        Sigma_split = 0.25 * np.outer(dg, dg)
    else:
        Sigma_split = np.zeros((d,d), float)

    # Combine
    Sigma_g = 0.5 * (Sigma_boot + Sigma_split)
    Sigma_g = 0.5 * (Sigma_g + Sigma_g.T)

    # Cov(α̂_T) and propagate to σ
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        lam = 1e-6 * (np.trace(H)/max(1,d) + 1.0)
        logging.info("[Δ-method] Inverting H failed; adding ridge %.3e", lam)
        H_inv = np.linalg.inv(H + lam*np.eye(d))

    cov_alpha_T = H_inv @ Sigma_g @ H_inv.T
    cov_alpha_T = 0.5 * (cov_alpha_T + cov_alpha_T.T)

    cov_sigma_from_alpha = J @ cov_alpha_T @ J.T
    cov_sigma_from_alpha = 0.5 * (cov_sigma_from_alpha + cov_sigma_from_alpha.T)

    cov_fixed = np.asarray(ase["asymp_cov"], float)
    cov_total = cov_fixed + cov_sigma_from_alpha
    se_fixed  = np.sqrt(np.clip(np.diag(cov_fixed), 0.0, np.inf))
    se_delta  = np.sqrt(np.clip(np.diag(cov_sigma_from_alpha), 0.0, np.inf))
    se_total  = np.sqrt(np.clip(np.diag(cov_total), 0.0, np.inf))

    # Summary
    try:
        condH = np.linalg.cond(H)
    except Exception:
        condH = np.nan
    # basic diagnostics
    normJ = float(np.linalg.norm(J))
    logging.info("[Δ-method probe-split] d=%d | ||J||_F=%.3e | tr(Σ_boot)=%.3e | tr(Σ_split)=%.3e | cond(H)=%.3e | "
                 "SE_fixed=[%s] | SE_Δα=[%s] | SE_total=[%s]",
                 d, normJ, float(np.trace(Sigma_boot)), float(np.trace(Sigma_split)), condH,
                 " ".join(f"{x:.4e}" for x in se_fixed),
                 " ".join(f"{x:.4e}" for x in se_delta),
                 " ".join(f"{x:.4e}" for x in se_total))

    return {
        "T": T,
        "J_sigma_alphaT": J,
        "H_tangent": H,
        "cov_alpha_tangent": cov_alpha_T,
        "cov_sigma_from_alpha": cov_sigma_from_alpha,
        "cov_sigma_fixed": cov_fixed,
        "cov_sigma_total": cov_total,
        "se_fixed": se_fixed,
        "se_delta": se_delta,
        "se_total": se_total,
        "Sigma_boot": Sigma_boot,
        "Sigma_split": Sigma_split,
    }

# =========================== Linear-time HE screen ===========================
@torch.no_grad()
def he_screen_individual_envs_linear(shared, env_cols_used, thresh: float = 0.0, topk: int = 0, save_path: Optional[str] = None):
    """
    Linear-time HE screen across environments.
    """
    KW_avg   = shared["KW_avg"]      # (N,B)
    Kdiag    = shared["Kdiag"]       # (N,)
    y        = shared["y"]           # (N,)
    E        = shared["E"]           # (N,L)
    Wt       = shared["Wt"]          # (N,B)
    bed_prefix = shared["bed_prefix"]
    row_idx    = shared["row_idx"]
    block_snps = shared["block_snps"]
    col_start  = shared["col_start"]
    col_stop   = shared["col_stop"]
    device     = shared["device"]

    invB = 1.0 / float(KW_avg.shape[1])
    N, L = int(shared["N"]), int(shared["L"])

    e_all  = E
    e2_all = e_all * e_all
    y2     = y * y
    tr_KK_nxe_all     = torch.sum(e2_all * e2_all, dim=0)
    tr_KK_gxe_nxe_all = torch.sum((e2_all * e2_all) * Kdiag[:, None], dim=0)
    tr_K_nxe_all      = torch.sum(e2_all, dim=0)
    tr_K_gxe_all      = torch.sum(Kdiag[:, None] * e2_all, dim=0)
    tr_K              = torch.sum(Kdiag)
    tr_gg             = invB * torch.sum(KW_avg * KW_avg)
    yKy1              = shared["yKy1"]

    sig_gxe = np.zeros(L, dtype=float)

    G = open_bed(f"{bed_prefix}.bed")
    with tqdm(total=L, desc="HE screen (linear in L)", unit="env", dynamic_ncols=True) as pbar:
        for l in range(L):
            e_l  = e_all[:, l]
            e2_l = e2_all[:, l]

            KgxeW_sum_ll = torch.zeros((N, KW_avg.shape[1]), device=device, dtype=DTYPE)
            Gy_ll_accum  = torch.tensor(0.0, device=device, dtype=DTYPE)
            Ge_ll_accum  = torch.tensor(0.0, device=device, dtype=DTYPE)

            for s in range(int(col_start), int(col_stop), int(block_snps)):
                ecol = min(s + block_snps, int(col_stop))
                b = ecol - s

                Xb_np = G.read(index=(row_idx, slice(s, ecol))).astype(np.float64, copy=False)
                Xb = to_dev(Xb_np, device)
                tmp  = torch.nan_to_num(Xb)
                cnt  = torch.sum(torch.isfinite(Xb), dim=0, dtype=DTYPE).clamp_min(1.0)
                sumv = torch.sum(tmp, dim=0, dtype=DTYPE)
                sum2 = torch.sum(tmp * tmp, dim=0, dtype=DTYPE)
                mean_b = sumv / cnt
                var_b  = (sum2 / cnt - mean_b.to(torch.float64)**2).clamp_min(0.0).to(DTYPE)
                std_b  = torch.where(var_b == 0, torch.ones_like(var_b), torch.sqrt(var_b))
                Xb = (tmp - mean_b) / std_b

                Xb_e = Xb * e_l[:, None]
                Z   = Xb_e
                ZtW = (Xb_e.T) @ Wt
                KgxeW_sum_ll += Z @ ZtW

                Ey_l = e_l * y
                v1   = Xb.T @ Ey_l
                Gy_ll_accum = Gy_ll_accum + (v1 @ v1)

                v2   = Xb.T @ e_l
                Ge_ll_accum = Ge_ll_accum + (v2 @ v2)

                del Xb, tmp, Xb_e, Z, ZtW, v1, v2
                if device.type == "cuda": torch.cuda.empty_cache()

            invM = 1.0 / float(shared["M"])
            KgxeW_ll_avg = KgxeW_sum_ll * invM
            Gy_ll        = Gy_ll_accum  * invM

            tr_ge = invB * torch.sum(KW_avg * KgxeW_ll_avg)
            tr_ee = invB * torch.sum(KgxeW_ll_avg * KgxeW_ll_avg)

            A = torch.stack([
                torch.stack([tr_gg,               tr_ge,                       tr_K_gxe_all[l], tr_K      ]),
                torch.stack([tr_ge,               tr_ee,                       tr_KK_gxe_nxe_all[l], tr_K_gxe_all[l]]),
                torch.stack([tr_K_gxe_all[l],     tr_KK_gxe_nxe_all[l],        tr_KK_nxe_all[l],  tr_K_nxe_all[l]]),
                torch.stack([tr_K,                tr_K_gxe_all[l],             tr_K_nxe_all[l],   torch.tensor(float(N), device=device, dtype=DTYPE)])
            ]).to(device=device, dtype=DTYPE)

            b_vec = torch.stack([
                yKy1,
                Gy_ll,
                torch.sum(y * e2_l * y),
                torch.sum(y * y)
            ]).to(device=device, dtype=DTYPE)

            sigma4 = _solve_sigma_linear_or_nnls(A, b_vec, nonneg=True)  # HE screen is safer strictly nonneg
            sig_gxe[l] = float(sigma4[1].detach().cpu().item())

            del KgxeW_sum_ll, KgxeW_ll_avg, Gy_ll_accum, Ge_ll_accum
            if device.type == "cuda": torch.cuda.empty_cache()
            pbar.update(1)

    keep_mask = sig_gxe >= float(thresh)
    idx_from_thresh = np.where(keep_mask)[0]
    idx_from_topk = np.array([], dtype=int)
    if topk and topk > 0:
        order = np.argsort(-sig_gxe)  # descending
        topk = min(topk, len(sig_gxe))
        idx_from_topk = order[:topk]
    keep_idx = np.unique(np.concatenate([idx_from_thresh, idx_from_topk])).astype(int)
    if keep_idx.size == 0:
        keep_idx = np.array([int(np.argmax(sig_gxe))], dtype=int)

    if save_path:
        df = pd.DataFrame({"env": env_cols_used, "sigma_gxe": sig_gxe})
        Path(os.path.dirname(os.path.abspath(save_path) or ".")).mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, sep="\t", index=False)
        logging.info("Saved HE screen table: %s", save_path)

    return sig_gxe, np.sort(keep_idx)


def _soft_threshold_unit(a_np, tau):
    a = np.asarray(a_np, float).copy()
    a = np.sign(a) * np.maximum(np.abs(a) - float(tau), 0.0)
    n = np.linalg.norm(a)
    if n < 1e-18:
        # keep the largest-magnitude coord to avoid degeneracy
        j = int(np.argmax(np.abs(a_np)))
        a = np.zeros_like(a); a[j] = np.sign(a_np[j]) if a_np[j] != 0 else 1.0
        return a
    return a / n


# =========================== HE resume helpers ===========================
def _select_from_sigmas(sig_gxe: np.ndarray, thresh: float, topk: int, min_keep: int) -> np.ndarray:
    sig = np.asarray(sig_gxe, float).reshape(-1)
    keep_from_thresh = np.where(sig >= float(thresh))[0]
    keep_from_topk = np.array([], dtype=int)
    if topk and topk > 0:
        order = np.argsort(-sig)  # desc
        keep_from_topk = order[:min(topk, sig.size)]
    keep_idx = np.unique(np.concatenate([keep_from_thresh, keep_from_topk])).astype(int)

    need = max(1, int(min_keep))
    if keep_idx.size < need:
        order = np.argsort(-sig)
        keep_idx = order[:need]
    return np.sort(keep_idx)

def _load_he_table(path: str, env_cols_used_full: List[str]) -> np.ndarray:
    df = pd.read_csv(path, sep="\t")
    if not {"env", "sigma_gxe"}.issubset(df.columns):
        raise ValueError(f"HE table '{path}' must have columns: env, sigma_gxe")
    m = {str(r.env): float(r.sigma_gxe) for r in df.itertuples(index=False)}
    out = np.full(len(env_cols_used_full), np.nan, dtype=float)
    for i, nm in enumerate(env_cols_used_full):
        if nm in m:
            out[i] = m[nm]
    missing = np.isnan(out)
    if np.any(missing):
        logging.warning("HE resume table missing %d/%d envs; these will be treated as -inf for selection.",
                        int(missing.sum()), len(out))
        out[missing] = -np.inf
    return out

# =========================== Variant split-half CV (A/B) ===========================
def _half_ranges(total_m: int, start: int, stop: int):
    stop = int(min(stop, total_m)) if stop is not None else int(total_m)
    start = int(start)
    assert stop > start, f"Empty SNP slice: [{start}, {stop})"
    mid = start + (stop - start) // 2
    A = (start, mid)
    B = (mid, stop)
    return A, B

@torch.no_grad()
def _fit_alpha_on_range(bed_prefix, row_idx, E_dev, y_dev,
                        B, seed, block_snps, range_pair, device,
                        pair_chunk_size=1, kgxew_fp32=True,
                        iters=200, step=0.2, l1_tau=0,
                        l1_anneal="none", l1_tau_min=0.0,
                        nonneg: bool = False):
    col_start, col_stop = range_pair
    C = precompute_from_bed(
        bed_prefix=bed_prefix,
        row_idx=row_idx,
        E_dev=E_dev,
        y_dev=y_dev,
        B=B,
        seed=seed,
        block_snps=block_snps,
        col_start=col_start,
        col_stop=col_stop,
        device=device,
        pair_chunk_size=pair_chunk_size,
        kgxew_fp32=kgxew_fp32,
    )
    C["nonneg_sigma"] = bool(nonneg)
    C["l1_tau"] = float(l1_tau)
    C["l1_anneal"] = str(l1_anneal)
    C["l1_tau_min"] = float(l1_tau_min)
    rng = np.random.default_rng(seed + 12345)
    alpha0 = unit_np(rng.standard_normal(int(E_dev.shape[1])))
    alpha_hat, sigma_hat, history = optimize_alpha_no_backtracking(
        C, alpha0, iters=iters, step=step, theta_max_deg=25.0, log_every=max(1, iters//10)
    )
    return alpha_hat.reshape(-1), sigma_hat.reshape(-1), C, history

@torch.no_grad()
def _holdout_sigma_on_range(bed_prefix, row_idx, E_dev, y_dev,
                            alpha_hat_np, B, W, seed, block_snps,
                            range_pair, device,nonneg_sigma: bool = False):
    col_start, col_stop = range_pair
    ase = asymptotic_sigma_se_over_variant_blocks(
        bed_prefix=bed_prefix,
        row_idx=row_idx,
        E_dev=E_dev,
        y_dev=y_dev,
        alpha_hat=alpha_hat_np,
        B=B,
        W=W,
        seed=seed,
        block_snps=block_snps,
        col_start=col_start,
        col_stop=col_stop,
        device=device,
        nonneg_sigma=nonneg_sigma
    )
    return ase["sigma_full"], ase["asymp_se"]

def _cv_variants_split_half(
    bed_prefix, row_idx, E_dev, y_dev,
    total_m, full_range, B, base_seed, block_snps, device,
    iters, step, pair_chunk_size, kgxew_fp32,
    z_gate=1.0, l1_tau=0, l1_anneal="none", l1_tau_min=0.0, nonneg: bool = False
):
    """SNP split-half CV: A→fit α, holdout σ on B; swap, average.
       If both held-outs say |σ_gxe|,|σ_nxe| ≤ z*SE, set to zero."""
    A, Bhalf = _half_ranges(total_m, full_range[0], full_range[1])
    # Fit on A, holdout on B
    alpha_A, sigma_in_A, C_A, _ = _fit_alpha_on_range(
        bed_prefix, row_idx, E_dev, y_dev, B, base_seed+11,
        block_snps, A, device, pair_chunk_size, kgxew_fp32, iters, step,
        l1_tau, l1_anneal, l1_tau_min, nonneg=nonneg
    )
    sig_B, se_B = _holdout_sigma_on_range(
        bed_prefix, row_idx, E_dev, y_dev, alpha_A,
        B, C_A.get("W", None), base_seed+13, block_snps, Bhalf, device, nonneg_sigma=nonneg
    )
    # Fit on B, holdout on A
    alpha_B, sigma_in_B, C_B, _ = _fit_alpha_on_range(
        bed_prefix, row_idx, E_dev, y_dev, B, base_seed+21,
        block_snps, Bhalf, device, pair_chunk_size, kgxew_fp32, iters, step,
        l1_tau, l1_anneal, l1_tau_min, nonneg=nonneg
    )
    sig_A, se_A = _holdout_sigma_on_range(
        bed_prefix, row_idx, E_dev, y_dev, alpha_B,
        B, C_B.get("W", None), base_seed+23, block_snps, A, device, nonneg_sigma=nonneg
    )

    sig_AB = 0.5 * (np.asarray(sig_A) + np.asarray(sig_B))

    def _no_interaction(sig, se, z):
        return (abs(sig[1]) <= z*se[1]) and (abs(sig[2]) <= z*se[2])

    no_gxe_AB = _no_interaction(sig_A, se_A, z_gate)
    no_gxe_BA = _no_interaction(sig_B, se_B, z_gate)

    aA = np.asarray(alpha_A).reshape(-1)
    aB = np.asarray(alpha_B).reshape(-1)
    if float(aA @ aB) < 0: aB = -aB
    alpha_final = unit_np(aA + aB)

    if no_gxe_AB and no_gxe_BA:
        sig_AB = np.asarray(sig_AB, float)
        sig_AB[1] = 0.0
        sig_AB[2] = 0.0

    debug = {
        "holdout_A_on_B_sigma": sig_B.tolist(),
        "holdout_A_on_B_se":    se_B.tolist(),
        "holdout_B_on_A_sigma": sig_A.tolist(),
        "holdout_B_on_A_se":    se_A.tolist(),
        "no_gxe_AB": bool(no_gxe_AB),
        "no_gxe_BA": bool(no_gxe_BA),
    }
    return alpha_final, sig_AB, debug

@torch.no_grad()
def _solve_sigma_linear_or_nnls(A: torch.Tensor,
                                b: torch.Tensor,
                                nonneg: bool = False,
                                max_iter: int = 200,
                                tol: float = 1e-10) -> torch.Tensor:
    """
    Solve min_x ||A x - b||_2 with optional nonnegativity x >= 0.
    Uses projected gradient on normal equations (small 4D problem).
    """
    if not nonneg:
        return torch.linalg.solve(A, b)

    AtA = A.T @ A
    Atb = A.T @ b

    # Lipschitz step via spectral norm of AtA (exact in 4D)
    L = torch.linalg.eigvalsh(AtA).max().clamp_min(torch.tensor(1e-12, dtype=A.dtype, device=A.device))
    step = 1.0 / L

    # Warm start: clip unconstrained solution
    try:
        x = torch.clamp(torch.linalg.solve(A, b), min=0.0)
    except Exception:
        x = torch.clamp(torch.zeros_like(b), min=0.0)

    for _ in range(max_iter):
        grad = AtA @ x - Atb
        x_new = torch.clamp(x - step * grad, min=0.0)
        if torch.max(torch.abs(x_new - x)) <= tol:
            break
        x = x_new
    return x


# =========================== Orchestration ===========================
def cosine(u, v, eps=1e-12):
    u = np.asarray(u).reshape(-1); v = np.asarray(v).reshape(-1)
    num = float(u @ v); den = float(np.linalg.norm(u) * np.linalg.norm(v) + eps)
    return num / den

def run_one_sim_bed(bed_prefix, env_file, num_samples, col_start, col_stop,
                    sigma_g, sigma_gxe, sigma_nxe, B, seed, iters, step, block_snps,
                    pheno_file, pheno_name, env_cols, anchor_cols, anchor_weight,
                    force_positive_feature, device,
                    pair_chunk_size=1, kgxew_fp32=True, save_prefix: str = None,
                    he_do_screen: bool = True, he_thresh: float = 0.0, he_topk: int = 0,
                    he_min_keep: int = 1, he_save_table: Optional[str] = None,
                    he_resume_table: Optional[str] = None,
                    flip_envs_where_gxe_neg: bool = False, cat_max_levels: int = 10,
                    flip_report: bool = False,
                    cv_variants: bool = False, cv_z: float = 1.0,
                    l1_tau=0.0, l1_anneal="none", l1_tau_min=0.0, nonneg=False,
                    lifestyle_envs: bool = False):

    dbg("run: start", level=1)
    (G, row_idx, E_np_fullW, bed_fids_sel, pheno_df, env_cols_used_full, Lw_inv,
     env_df_full, E_mean, E_std) = load_real_E_and_row_idx(
        bed_prefix, env_file, num_samples, env_cols,
        pheno_file, pheno_name, lifestyle_envs=lifestyle_envs
    )
    E_dev_fullW = to_dev(E_np_fullW, device)

    # Phenotype (load or simulate)
    alpha_true = None
    if pheno_file:
        y_df = pheno_df.set_index('FID')
        if y_df.index.dtype != bed_fids_sel.dtype:
            y_df.index = y_df.index.astype(str)
        y_df = y_df.reindex(bed_fids_sel)

        y_np = y_df[pheno_name].to_numpy(dtype=np.float64)
        m = np.nanmean(y_np)
        y_np = np.nan_to_num(y_np, nan=m).astype(np.float64)
        y_np = (y_np - y_np.mean()) / (y_np.std(ddof=1) + 1e-12)
        y_dev = to_dev(y_np, device)
    else:
        y_cpu, alpha_true_cpu = simulate_y_from_bed(
            bed_prefix, row_idx, E_dev_fullW, seed,
            sigma_g, sigma_gxe, sigma_nxe,
            col_start, col_stop, block_snps, device
        )
        y_np = y_cpu.astype(np.float64)
        y_dev = to_dev(y_np, device)
        alpha_true = alpha_true_cpu

    # HE screen path
    he_sigma_gxe_initial = None
    selected_idx = np.arange(E_dev_fullW.shape[1], dtype=int)

    if he_do_screen:
        if he_resume_table and os.path.exists(he_resume_table):
            logging.info("Resuming HE screen from table: %s", he_resume_table)
            he_sigma_gxe_initial = _load_he_table(he_resume_table, env_cols_used_full)
            selected_idx = _select_from_sigmas(he_sigma_gxe_initial, he_thresh, he_topk, he_min_keep)
        else:
            shared = precompute_shared_for_he(
                bed_prefix=bed_prefix,
                row_idx=row_idx,
                E_dev=E_dev_fullW,
                y_dev=y_dev,
                B=B,
                W=None,
                seed=seed,
                block_snps=block_snps,
                col_start=col_start,
                col_stop=col_stop,
                device=device,
            )
            he_sigma_gxe_initial, selected_idx = he_screen_individual_envs_linear(
                shared, env_cols_used_full, thresh=he_thresh, topk=he_topk, save_path=he_save_table
            )
            if selected_idx.size < max(1, int(he_min_keep)):
                selected_idx = _select_from_sigmas(he_sigma_gxe_initial, he_thresh, he_topk, he_min_keep)

    # Optional flipping for negative σ_gxe
    flipped_envs: List[str] = []
    flip_kinds: Dict[str, str] = {}
    if (he_do_screen and (he_resume_table is None) and (he_sigma_gxe_initial is not None) and flip_envs_where_gxe_neg):
        he_sigma = np.asarray(he_sigma_gxe_initial, float).reshape(-1)
        for i, s in enumerate(he_sigma):
            if s < 0.0:
                colname = env_cols_used_full[i]
                is_cand, kind = detect_env_is_binary_or_categorical(env_df_full[colname], max_levels=cat_max_levels)
                if is_cand:
                    flip_env_column_inplace(env_df_full, colname)
                    flipped_envs.append(colname)
                    flip_kinds[colname] = kind
        if len(flipped_envs) > 0:
            logging.info("Flipped envs due to negative HE sigma_gxe (and categorical/binary): %s",
                         ", ".join([f"{nm}({flip_kinds[nm]})" for nm in flipped_envs]))
            # Re-whiten and re-run HE screen
            E_np_fullW, Lw_inv, E_mean, E_std = _standardize_and_whiten(env_df_full, env_cols_used_full, bed_fids_sel)
            E_dev_fullW = to_dev(E_np_fullW, device)
            shared = precompute_shared_for_he(
                bed_prefix=bed_prefix,
                row_idx=row_idx,
                E_dev=E_dev_fullW,
                y_dev=y_dev,
                B=B,
                W=None,
                seed=seed + 11,
                block_snps=block_snps,
                col_start=col_start,
                col_stop=col_stop,
                device=device,
            )
            he_sigma_gxe_initial, selected_idx = he_screen_individual_envs_linear(
                shared, env_cols_used_full, thresh=he_thresh, topk=he_topk, save_path=he_save_table
            )
            if selected_idx.size < max(1, int(he_min_keep)):
                selected_idx = _select_from_sigmas(he_sigma_gxe_initial, he_thresh, he_topk, he_min_keep)

    # Subset E for α-stage
    if selected_idx.size < E_dev_fullW.shape[1]:
        idx_t = torch.as_tensor(selected_idx, device=E_dev_fullW.device, dtype=torch.long)
        E_dev = E_dev_fullW.index_select(1, idx_t)
    else:
        E_dev = E_dev_fullW
    env_cols_used = [env_cols_used_full[i] for i in selected_idx]

    # ========== Variant split-half CV path ==========
    if cv_variants:
        total_m = open_bed(f"{bed_prefix}.bed").shape[1]
        full_start = int(col_start)
        full_stop  = int(total_m if col_stop is None else min(col_stop, total_m))

        alpha_cv, sigma_cv, cv_dbg = _cv_variants_split_half(
            bed_prefix=bed_prefix,
            row_idx=row_idx,
            E_dev=to_dev(E_dev, device),
            y_dev=y_dev,
            total_m=total_m,
            full_range=(full_start, full_stop),
            B=B,
            base_seed=seed,
            block_snps=block_snps,
            device=device,
            iters=iters,
            step=step,
            pair_chunk_size=pair_chunk_size,
            kgxew_fp32=kgxew_fp32,
            z_gate=float(cv_z),
            l1_tau=l1_tau,
            l1_anneal=l1_anneal,
            l1_tau_min=l1_tau_min
        )

        # Fixed-α SE on full SNP range using α from CV
        ase_full = asymptotic_sigma_se_over_variant_blocks(
            bed_prefix=bed_prefix,
            row_idx=row_idx,
            E_dev=to_dev(E_dev, device),
            y_dev=y_dev,
            alpha_hat=np.asarray(alpha_cv).reshape(-1),
            B=B,
            W=None,
            seed=seed + 991,
            block_snps=block_snps,
            col_start=full_start,
            col_stop=full_stop,
            device=device,
            nonneg_sigma=nonneg,
        )

        out = {
            "N": row_idx.size,
            "M_used": (full_stop - full_start),
            "L": int(E_dev.shape[1]),
            "sigma_hat": np.asarray(ase_full["sigma_full"].tolist(), float),

            "sigma_hat_se_fixed_alpha": ase_full["asymp_se"].tolist(),
            "sigma_hat_se_alpha_delta": [0.0, 0.0, 0.0, 0.0],
            "sigma_hat_se_total":       ase_full["asymp_se"].tolist(),

            "sigma_hat_jackknife_full": ase_full["sigma_full"].tolist(),
            "sigma_hat_jackknife_se":   ase_full["asymp_se"].tolist(),

            "alpha_hat_full": None,   # fill below
            "alpha_hat_selected": None,
            "alpha_hat_whitened_selected": np.asarray(alpha_cv).reshape(-1).tolist(),

            "env_cols_used": [c for c in env_cols_used],
            "env_cols_used_full": env_cols_used_full,
            "selected_idx": selected_idx.tolist(),
            "selected_cols": [env_cols_used_full[i] for i in selected_idx],
            "he_sigma_gxe_per_env": he_sigma_gxe_initial.tolist() if he_sigma_gxe_initial is not None else None,
            "anchor_cols_used": [],
            "anchor_weight": 0.0,
            "force_positive_feature": None,
            "flipped_envs": flipped_envs,
            "flip_kinds": flip_kinds,
            "flip_envs_where_gxe_neg": bool(flip_envs_where_gxe_neg),
            "cat_max_levels": int(cat_max_levels),

            "cv_variants_used": True,
            "cv_z": float(cv_z),
            "cv_debug": cv_dbg,
        }

        # Map α back to original basis and all/full
        L_full = E_dev_fullW.shape[1]
        alpha_w_full = np.zeros((L_full, 1), dtype=np.float64)
        alpha_w_full[selected_idx.reshape(-1), 0] = np.asarray(alpha_cv).reshape(-1)
        alpha_orig_full = (Lw_inv @ alpha_w_full)

        out["alpha_hat_full"] = alpha_orig_full.reshape(-1).tolist()
        out["alpha_hat_selected"] = alpha_orig_full[selected_idx].reshape(-1).tolist()

        if alpha_true is not None:
            out["alpha_true"] = alpha_true.reshape(-1)
            out["cos_alpha"] = cosine(alpha_true, alpha_w_full)
            e_true = E_np_fullW @ alpha_true.reshape(-1, 1)
            e_hat  = E_np_fullW @ alpha_w_full
            out["rmse_env_score"] = float(np.sqrt(np.mean((e_true - e_hat) ** 2)))
        else:
            out["alpha_true"] = None
            out["cos_alpha"] = np.nan
            out["rmse_env_score"] = np.nan
            
            
        if isinstance(save_prefix, str) and len(save_prefix) > 0:
            try:
                e_hat_np = (E_np_fullW @ alpha_w_full).reshape(-1)  # uses alpha_w_full you computed above
                e_true_np = None
                if alpha_true is not None:
                    e_true_np = (E_np_fullW @ alpha_true.reshape(-1, 1)).reshape(-1)

                save_transformed_env(save_prefix, e_hat=e_hat_np, e_true=e_true_np)
                save_transformed_env_with_ids(prefix=save_prefix,
                                            fids=bed_fids_sel.values,
                                            iids=bed_fids_sel.values,
                                            e_hat=e_hat_np,
                                            e_true=e_true_np,
                                            col_name_hat="e_hat",
                                            col_name_true="e_true")
                save_full_transformed_env(
                    prefix=save_prefix,
                    env_df_full=env_df_full,
                    env_cols_used=env_cols_used_full,   # use ALL env columns
                    E_mean=E_mean,
                    E_std=E_std,
                    Lw_inv=Lw_inv,
                    alpha_hat_whitened=alpha_w_full.reshape(-1),
                )

                E_np_alpha_stage = _to_numpy(E_dev)  # E used for α-stage (post-screen/whitened)
                save_E_y_tsv(prefix=save_prefix,
                            bed_ids=bed_fids_sel,
                            env_cols_used=env_cols_used,
                            E_np=E_np_alpha_stage,
                            y_np=y_np,
                            pheno_name=pheno_name)
                save_E_y_with_ids(prefix=save_prefix,
                                bed_fids_sel=bed_fids_sel,
                                env_cols_used=env_cols_used,
                                E_np=E_np_alpha_stage,
                                y_np=y_np,
                                pheno_name=pheno_name)

                save_plink_filters(prefix=save_prefix,
                                G_bed=G,
                                bed_fids_sel=bed_fids_sel,
                                col_start=col_start,
                                col_stop=col_stop)
            except Exception as exc:
                logging.warning("Failed to save files at prefix '%s' (CV path): %s", save_prefix, exc)

        return out

    # ========== Standard (non-CV) α-stage ==========
    C = precompute_from_bed(
        bed_prefix=bed_prefix,
        row_idx=row_idx,
        E_dev=E_dev,
        y_dev=y_dev,
        B=B,
        seed=seed + 1,
        block_snps=block_snps,
        col_start=col_start,
        col_stop=col_stop,
        device=device,
        pair_chunk_size=pair_chunk_size,
        kgxew_fp32=kgxew_fp32,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    rng = np.random.default_rng(seed + 123)
    alpha0 = unit_np(rng.standard_normal(int(E_dev.shape[1])))

    _ = build_anchor_from_names(env_cols_used, anchor_cols)
    C["nonneg_sigma"] = nonneg
    C["l1_tau"] = float(l1_tau)
    C["l1_anneal"] = str(l1_anneal)
    C["l1_tau_min"] = float(l1_tau_min)
    alpha_hat_w_sub, sigma_hat, history = optimize_alpha_no_backtracking(
        C, alpha0, iters=iters, step=step, theta_max_deg=25.0, log_every=max(1, iters//10),
        orient_guard=None,
    )

    # Asymptotic SE (fixed α)
    ase = asymptotic_sigma_se_over_variant_blocks(
        bed_prefix=bed_prefix,
        row_idx=row_idx,
        E_dev=to_dev(E_dev, device),
        y_dev=y_dev,
        alpha_hat=alpha_hat_w_sub.reshape(-1),
        B=B,
        W=C.get("W", None),
        seed=seed + 991,
        block_snps=block_snps,
        col_start=col_start,
        col_stop=col_stop,
        device=device,
        nonneg_sigma=nonneg,
    )

    print("\n=== Asymptotic SE (Hutchinson–sandwich; α held fixed) ===")
    print("σ̂_full (g,gxe,nxe,e):", np.array2string(ase["sigma_full"], formatter={'float_kind':lambda x: f'{x:.6f}'}))
    print("SE_asymp (g,gxe,nxe,e):", np.array2string(ase["asymp_se"],   formatter={'float_kind':lambda x: f'{x:.6f}'}))

    # Δ from α-uncertainty via PROBE SPLITTING
    delta = delta_method_sigma_alpha_correction_probe_split(
        C,
        alpha_hat_w_sub.reshape(-1),
        ase,
        probe_boot_reps=128,
        theta_fd=5e-3,
        seed=seed + 2025,
        hess_ridge=1e-8,
    )

    print("\n=== Δ from α-uncertainty (probe-splitting) ===")
    print("SE_delta_α (g,gxe,nxe,e):", np.array2string(delta["se_delta"], formatter={'float_kind':lambda x: f'{x:.6f}'}))
    print("SE_total  (g,gxe,nxe,e):", np.array2string(delta["se_total"], formatter={'float_kind':lambda x: f'{x:.6f}'}))

    # Map α back (both all and selected views)
    L_full = E_dev_fullW.shape[1]
    alpha_w_full = np.zeros((L_full, 1), dtype=np.float64)
    alpha_w_full[selected_idx.reshape(-1), 0] = alpha_hat_w_sub.reshape(-1)
    alpha_orig_full = (Lw_inv @ alpha_w_full)

    alpha_orig_selected       = alpha_orig_full[selected_idx, :].reshape(-1)
    alpha_whitened_selected   = alpha_hat_w_sub.reshape(-1)

    if force_positive_feature:
        name2idx = {_norm_name(c): i for i, c in enumerate(env_cols_used_full)}
        key = _norm_name(force_positive_feature)
        idx = name2idx.get(key, None)
        if idx is None:
            logging.warning("force-positive feature '%s' not found; skipping.", force_positive_feature)
        else:
            if float(alpha_orig_full[idx]) < 0.0:
                alpha_w_full            = -alpha_w_full
                alpha_orig_full         = -alpha_orig_full
                alpha_orig_selected     = -alpha_orig_selected
                alpha_whitened_selected = -alpha_whitened_selected

    e_hat_np = (E_np_fullW @ alpha_w_full).reshape(-1)

    if isinstance(save_prefix, str) and len(save_prefix) > 0:
        try:
            e_true_np = None
            if alpha_true is not None:
                e_true_np = (E_np_fullW @ alpha_true.reshape(-1, 1)).reshape(-1)
            save_transformed_env(save_prefix, e_hat=e_hat_np, e_true=e_true_np)

            save_transformed_env_with_ids(prefix=save_prefix,
                                          fids=bed_fids_sel.values,
                                          iids=bed_fids_sel.values,
                                          e_hat=e_hat_np,
                                          e_true=e_true_np,
                                          col_name_hat="e_hat",
                                          col_name_true="e_true")
            save_full_transformed_env(
                prefix=save_prefix,
                env_df_full=env_df_full,
                env_cols_used=env_cols_used_full,   # use ALL env columns
                E_mean=E_mean,
                E_std=E_std,
                Lw_inv=Lw_inv,
                alpha_hat_whitened=alpha_w_full.reshape(-1),
            )
        except Exception as exc:
            logging.warning("Failed to save transformed environment at prefix '%s': %s", save_prefix, exc)

    if isinstance(save_prefix, str) and len(save_prefix) > 0:
        try:
            E_np_alpha_stage = _to_numpy(E_dev)
            save_E_y_tsv(prefix=save_prefix,
                         bed_ids=bed_fids_sel,
                         env_cols_used=env_cols_used,
                         E_np=E_np_alpha_stage,
                         y_np=y_np,
                         pheno_name=pheno_name)
            save_E_y_with_ids(prefix=save_prefix,
                              bed_fids_sel=bed_fids_sel,
                              env_cols_used=env_cols_used,
                              E_np=E_np_alpha_stage,
                              y_np=y_np,
                              pheno_name=pheno_name)

            save_plink_filters(prefix=save_prefix,
                               G_bed=G,
                               bed_fids_sel=bed_fids_sel,
                               col_start=col_start,
                               col_stop=col_stop)
        except Exception as exc:
            logging.warning("Failed to save E/y at prefix '%s': %s", save_prefix, exc)

    he_sigmas_final = he_sigma_gxe_initial.tolist() if he_sigma_gxe_initial is not None else None

    metrics = {
        "N": row_idx.size,
        "M_used": (col_stop if col_stop is not None else G.shape[1]) - col_start,
        "L": int(E_dev.shape[1]),
        "sigma_hat": sigma_hat,

        # --- SEs ---
        "sigma_hat_se_fixed_alpha": ase["asymp_se"].tolist(),
        "sigma_hat_se_alpha_delta": delta["se_delta"].tolist(),
        "sigma_hat_se_total":       delta["se_total"].tolist(),

        "sigma_hat_jackknife_full": ase["sigma_full"].tolist(),
        "sigma_hat_jackknife_se":   ase["asymp_se"].tolist(),

        # --- alpha fields ---
        "alpha_hat_full": alpha_orig_full.reshape(-1).tolist(),
        "alpha_hat_selected": alpha_orig_selected.tolist(),
        "alpha_hat_whitened_selected": alpha_whitened_selected.tolist(),

        "alpha_hat": alpha_orig_full.reshape(-1).tolist(),

        "env_cols_used": env_cols_used,
        "env_cols_used_full": env_cols_used_full,
        "selected_idx": selected_idx.tolist(),
        "selected_cols": [env_cols_used_full[i] for i in selected_idx],
        "he_sigma_gxe_per_env": he_sigmas_final,
        "anchor_cols_used": [c for c in (anchor_cols or []) if c in env_cols_used],
        "anchor_weight": float(anchor_weight),
        "force_positive_feature": force_positive_feature,
        "flipped_envs": flipped_envs,
        "flip_kinds": flip_kinds,
        "flip_envs_where_gxe_neg": bool(flip_envs_where_gxe_neg),
        "cat_max_levels": int(cat_max_levels),
    }
    if alpha_true is not None:
        metrics["alpha_true"] = alpha_true.reshape(-1)
        metrics["cos_alpha"] = cosine(alpha_true, alpha_w_full)
        e_true = E_np_fullW @ alpha_true.reshape(-1, 1)
        e_hat  = E_np_fullW @ alpha_w_full
        metrics["rmse_env_score"] = float(np.sqrt(np.mean((e_true - e_hat) ** 2)))
    else:
        metrics["alpha_true"] = None
        metrics["cos_alpha"] = np.nan
        metrics["rmse_env_score"] = np.nan

    if flip_report and len(flipped_envs) > 0 and he_sigmas_final is not None:
        print("\n=== Flip report (post-flip HE, same order as env_cols_used_full) ===")
        print("env\tflipped?\tkind\tHE_sigma_gxe")
        he_arr = np.asarray(he_sigmas_final)
        for i, nm in enumerate(env_cols_used_full):
            f = "yes" if nm in flipped_envs else "no"
            k = flip_kinds.get(nm, "-")
            s = he_arr[i] if i < he_arr.size else float('nan')
            print(f"{nm}\t{f}\t{k}\t{s:+.6f}")
        print("=== end flip report ===\n")

    return metrics

def _ensure_dir_for(prefix: str):
    d = os.path.dirname(os.path.abspath(prefix))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_E_y_tsv(prefix: str,
                 bed_ids: pd.Index,
                 env_cols_used: List[str],
                 E_np: np.ndarray,
                 y_np: np.ndarray,
                 pheno_name: str):
    _ensure_dir_for(prefix)
    E_path = f"{prefix}.E.tsv"
    y_path = f"{prefix}.y.tsv"
    E_temp_df = pd.DataFrame(E_np, columns=env_cols_used)
    E_temp_df.columns = [f"e{k}" for k in range(len(E_temp_df.columns))]
    E_temp_df.to_csv(E_path, sep="\t", index=False)
    pd.DataFrame({pheno_name: y_np}).to_csv(y_path, sep="\t", index=False)
    logging.info("Saved intermediates: %s , %s", E_path, y_path)

def save_E_y_with_ids(prefix: str,
                      bed_fids_sel: pd.Index,
                      env_cols_used: List[str],
                      E_np: np.ndarray,
                      y_np: np.ndarray,
                      pheno_name: str):
    _ensure_dir_for(prefix)
    e_path = f"{prefix}.E_ids.tsv"
    y_path = f"{prefix}.y_ids.tsv"

    e_cols = [f"e{k}" for k in range(E_np.shape[1])]

    ids_df = pd.DataFrame({
        "FID": bed_fids_sel.astype(str).values,
        "IID": bed_fids_sel.astype(str).values,
    })

    e_df = pd.concat([ids_df.reset_index(drop=True),
                      pd.DataFrame(E_np, columns=e_cols)], axis=1)
    e_df.to_csv(e_path, sep="\t", index=False)

    y_df = ids_df.copy()
    y_df[pheno_name] = np.asarray(y_np, float).reshape(-1)
    y_df.to_csv(y_path, sep="\t", index=False)

    logging.info("Saved E/y with IDs: %s , %s", e_path, y_path)

def save_full_transformed_env(prefix: str,
                              env_df_full: pd.DataFrame,
                              env_cols_used: List[str],
                              E_mean: np.ndarray,
                              E_std: np.ndarray,
                              Lw_inv: np.ndarray,
                              alpha_hat_whitened = None):
    _ensure_dir_for(prefix)
    E_full_path = f"{prefix}.E_full.tsv"
    ehat_full_path = f"{prefix}.e_hat_full.tsv"

    mat_full = env_df_full[env_cols_used].to_numpy(dtype=np.float64)
    Ez_full = (mat_full - E_mean) / E_std
    E_white_full = Ez_full @ Lw_inv
    E_white_full = np.nan_to_num(E_white_full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    full_cols = [f"e{k}" for k in range(E_white_full.shape[1])]
    out_full = pd.concat(
        [env_df_full[['FID','IID']].reset_index(drop=True),
         pd.DataFrame(E_white_full, columns=full_cols)],
        axis=1
    )
    out_full.to_csv(E_full_path, sep="\t", index=False)
    logging.info("Saved full transformed env (no filtering): %s", E_full_path)

    if alpha_hat_whitened is not None:
        e_hat_full = (E_white_full @ alpha_hat_whitened.reshape(-1, 1)).reshape(-1)
        pd.DataFrame({
            "FID": env_df_full['FID'].values,
            "IID": env_df_full['IID'].values,
            "e_hat": e_hat_full
        }).to_csv(ehat_full_path, sep="\t", index=False)
        logging.info("Saved full transformed env-score (no filtering): %s", ehat_full_path)

def save_plink_filters(prefix: str,
                       G_bed,
                       bed_fids_sel: pd.Index,
                       col_start: int,
                       col_stop: int):
    _ensure_dir_for(prefix)
    keep_path = f"{prefix}.keep.txt"
    extract_path = f"{prefix}.extract.txt"

    keep_df = pd.DataFrame({"FID": bed_fids_sel.astype(str).values,
                            "IID": bed_fids_sel.astype(str).values})
    keep_df.to_csv(keep_path, sep=" ", header=False, index=False)

    total_m = int(G_bed.shape[1])
    stop = total_m if (col_stop is None) else min(int(col_stop), total_m)
    start = max(0, int(col_start))
    if start >= stop:
        raise ValueError(f"Empty SNP range for extract: start={start}, stop={stop}, total_m={total_m}")

    sid = np.asarray(G_bed.sid).astype(str)
    snp_ids = sid[start:stop]
    pd.Series(snp_ids).to_csv(extract_path, index=False, header=False)

    logging.info("Saved PLINK filters: keep -> %s  | extract -> %s", keep_path, extract_path)

def save_transformed_env(prefix: str,
                         e_hat: np.ndarray,
                         e_true: np.ndarray = None,
                         col_name_hat: str = "e_hat",
                         col_name_true: str = "e_true"):
    _ensure_dir_for(prefix)
    hat_path = f"{prefix}.e_hat.tsv"
    pd.DataFrame({col_name_hat: np.asarray(e_hat, float).reshape(-1)}).to_csv(hat_path, sep="\t", index=False)
    logging.info("Saved transformed environment (hat): %s", hat_path)
    if e_true is not None:
        true_path = f"{prefix}.e_true.tsv"
        pd.DataFrame({col_name_true: np.asarray(e_true, float).reshape(-1)}).to_csv(true_path, sep="\t", index=False)
        logging.info("Saved transformed environment (true): %s", true_path)

def save_transformed_env_with_ids(prefix: str,
                                  fids: np.ndarray,
                                  iids: np.ndarray,
                                  e_hat: np.ndarray,
                                  e_true: np.ndarray = None,
                                  col_name_hat: str = "e_hat",
                                  col_name_true: str = "e_true"):
    _ensure_dir_for(prefix)
    hat_path  = f"{prefix}.e_hat_ids.tsv"
    true_path = f"{prefix}.e_true_ids.tsv"

    df_ids = pd.DataFrame({"FID": fids.astype(str), "IID": iids.astype(str)})

    df_hat = df_ids.copy()
    df_hat[col_name_hat] = np.asarray(e_hat, float).reshape(-1)
    df_hat.to_csv(hat_path, sep="\t", index=False)
    logging.info("Saved transformed environment (hat) with IDs: %s", hat_path)

    if e_true is not None:
        df_true = df_ids.copy()
        df_true[col_name_true] = np.asarray(e_true, float).reshape(-1)
        df_true.to_csv(true_path, sep="\t", index=False)
        logging.info("Saved transformed environment (true) with IDs: %s", true_path)

def _to_list(x):
    return None if x is None else np.asarray(x).ravel().tolist()

def _vec_to_dict(vec, prefix: str, names=("g","gxe","nxe","e")):
    if vec is None:
        return {f"{prefix}_{n}": np.nan for n in names}
    v = np.asarray(vec, float).reshape(-1)
    out = {}
    for i, n in enumerate(names):
        out[f"{prefix}_{n}"] = float(v[i]) if i < v.size else np.nan
    return out

def results_to_alpha_row(out, meta=None):
    row = {
        "N": out.get("N"),
        "M_used": out.get("M_used"),
        "L": out.get("L"),
        "has_alpha_true": out.get("alpha_true") is not None,
        "cos_alpha": (float(out.get("cos_alpha"))
                      if out.get("alpha_true") is not None else np.nan),
        "rmse_env_score": (float(out.get("rmse_env_score"))
                           if out.get("alpha_true") is not None else np.nan),
        "alpha_hat_full": out.get("alpha_hat_full"),
        "alpha_hat_selected": out.get("alpha_hat_selected"),
        "alpha_hat_whitened_selected": out.get("alpha_hat_whitened_selected"),
        "alpha_dim": (len(out["alpha_hat_full"]) if out.get("alpha_hat_full") is not None else
                      (len(out["alpha_true"]) if out.get("alpha_true") is not None else np.nan)),
        "alpha_true": _to_list(out.get("alpha_true")),
        "env_cols_used": list(out.get("env_cols_used", [])),
        "selected_idx": list(out.get("selected_idx", [])),
        "selected_cols": list(out.get("selected_cols", [])),
        "anchor_cols_used": list(out.get("anchor_cols_used", [])),
        "anchor_weight": out.get("anchor_weight"),
        "force_positive_feature": out.get("force_positive_feature"),
        "flipped_envs": list(out.get("flipped_envs", [])),
        "flip_envs_where_gxe_neg": out.get("flip_envs_where_gxe_neg"),
        "cat_max_levels": out.get("cat_max_levels"),
        "he_sigma_gxe_per_env": _to_list(out.get("he_sigma_gxe_per_env")),
    }
    if meta:
        row.update(meta)
    return row

def results_to_sigma_row(out, meta=None):
    sigma = out.get("sigma_hat")
    row = {
        "N": out.get("N"),
        "M_used": out.get("M_used"),
        "L": out.get("L"),
    }
    row.update(_vec_to_dict(sigma, "sigma_hat"))

    jk_full = out.get("sigma_hat_jackknife_full")
    jk_se   = out.get("sigma_hat_jackknife_se")
    row.update(_vec_to_dict(jk_full, "sigma_hat_jk_full"))
    row.update(_vec_to_dict(jk_se,   "sigma_hat_jk_se"))

    row.update(_vec_to_dict(out.get("sigma_hat_se_fixed_alpha"), "sigma_hat_se_fixed_alpha"))
    row.update(_vec_to_dict(out.get("sigma_hat_se_alpha_delta"), "sigma_hat_se_alpha_delta"))
    row.update(_vec_to_dict(out.get("sigma_hat_se_total"),       "sigma_hat_se_total"))

    if meta:
        row.update(meta)
    return row

def append_row_to_csv(row, csv_path):
    df = pd.DataFrame([row])
    Path(os.path.dirname(os.path.abspath(csv_path))).mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, mode="w", index=False, header=True)

def expand_alpha_cols(row, key="alpha_hat"):
    alpha = row.get(key)
    if alpha is None:
        return {}
    return {f"{key}_{i}": float(v) for i, v in enumerate(alpha)}

# =========================== Main ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed-prefix", required=True, type=str)
    ap.add_argument("--env-file",   required=True, type=str)
    ap.add_argument("--pheno-file", type=str, default=None)
    ap.add_argument("--pheno-name", type=str, default="PHENO")

    ap.add_argument("--env-cols", type=str, default=None)
    ap.add_argument("--anchor-cols", type=str, default=None)
    ap.add_argument("--anchor-weight", type=float, default=0.0)
    ap.add_argument("--force-positive-feature", type=str, default=None)

    ap.add_argument("--num-samples", type=int, default=5000)
    ap.add_argument("--col-start", type=int, default=0)
    ap.add_argument("--col-stop",  type=int, default=200000, help="-1 ==> use all SNPs")

    ap.add_argument("--sigma-g",   type=float, default=0.5)
    ap.add_argument("--sigma-gxe", type=float, default=0.05)
    ap.add_argument("--sigma-nxe", type=float, default=0.05)

    ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--block-snps", type=int, default=2048)
    ap.add_argument("--pair-chunk-size", type=int, default=1, help="pairs processed at once (memory/speed tradeoff)")
    ap.add_argument("--kgxew-fp32", action="store_true", help="store KgxeW in float32 to save memory")
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--save-files", type=str, default=None,
                    help="Prefix to save standardized E/y and PLINK filters. "
                         "Writes <prefix>.E.tsv, <prefix>.y.tsv, <prefix>.keep.txt, <prefix>.extract.txt")

    # --- HE screen options ---
    ap.add_argument("--he-screen", action="store_true",
                    help="Run HE regression per environment and subset by threshold/top-K before α optimization.")
    ap.add_argument("--he-thresh", type=float, default=0.0,
                    help="Keep envs with per-env sigma_gxe >= this threshold (after HE).")
    ap.add_argument("--he-topk", type=int, default=0,
                    help="Additionally keep the top-K envs by sigma_gxe (K=0 disables).")
    ap.add_argument("--he-min-keep", type=int, default=1,
                    help="Ensure at least this many envs are kept (falls back to best).")
    ap.add_argument("--he-save-table", type=str, default=None,
                    help="Save per-env HE sigma_gxe as TSV here.")
    ap.add_argument("--he-resume-table", type=str, default=None,
                    help="Resume HE screen from this TSV (columns: env, sigma_gxe). Skips recomputing HE and disables flipping.")

    # --- Flipping controls ---
    ap.add_argument("--flip-envs-where-gxe-neg", action="store_true",
                    help="Flip raw env coding for envs with negative per-env HE sigma_gxe, if binary/small categorical.")
    ap.add_argument("--cat-max-levels", type=int, default=10,
                    help="Max unique integer-coded levels to consider a variable categorical for flipping.")
    ap.add_argument("--flip-report", action="store_true",
                    help="Print a concise flip report after post-flip HE screen.")

    # --- Variant split-half CV ---
    ap.add_argument("--cv-variants", action="store_true",
                    help="Enable variant split-half CV: fit α on A, holdout σ on B; swap, average; gate g×e/n×e by z.")
    ap.add_argument("--cv-z", type=float, default=1.0,
                    help="z-score gate on held-out σ_gxe and σ_nxe; if BOTH holds say |σ| ≤ z*SE, set to zero.")

    # Debugging flags
    ap.add_argument("--debug", type=str, default="off",
                    choices=["off","light","med","heavy"],
                    help="Print intermediate values (off/light/med/heavy)")
    ap.add_argument("--dump-prefix", type=str, default=None,
                    help="If set and --debug=heavy, dump key arrays to <prefix>.*.npy")

    # --- L1 spherical soft-threshold options ---
    ap.add_argument("--l1-tau", type=float, default=0.0, help="Spherical soft-threshold base strength τ.")
    ap.add_argument("--l1-anneal", type=str, default="none",
                    choices=["none", "linear", "cosine"],
                    help="Anneal schedule for τ across iterations (default: none).")
    ap.add_argument("--l1-tau-min", type=float, default=0.0,
                    help="Final/smallest τ when using annealing (default: 0.0).")
    ap.add_argument("--nonneg-sigma", action="store_true",
                help="Enforce σ_g, σ_gxe, σ_nxe, σ_e ≥ 0 via NNLS in all solves.")
    ap.add_argument(
            "--lifestyle-envs",
            action="store_true",
            help="Use predefined lifestyle_names (Age, Walked, ..., TDI) as the env set; "
                "otherwise use all env-file columns except FID/IID."
        )

    args = ap.parse_args()

    lvl_map = {"off":0, "light":1, "med":2, "heavy":3}
    set_debug(lvl_map.get(args.debug, 0), dump_prefix=args.dump_prefix)
    nonneg = bool(args.nonneg_sigma)

    log_level = logging.DEBUG if DEBUG_LEVEL > 0 else getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    device = get_device()
    logging.info("Torch device: %s", device.type)
    if device.type == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.backends.cudnn.benchmark = True

    col_stop = None if int(args.col_stop) == -1 else int(args.col_stop)
    env_cols = _parse_csv_list(args.env_cols)
    anchor_cols = _parse_csv_list(args.anchor_cols)

    out = run_one_sim_bed(
        bed_prefix=args.bed_prefix,
        env_file=args.env_file,
        num_samples=args.num_samples,
        col_start=args.col_start,
        col_stop=col_stop,
        sigma_g=args.sigma_g,
        sigma_gxe=args.sigma_gxe,
        sigma_nxe=args.sigma_nxe,
        B=args.B,
        seed=args.seed,
        iters=args.iters,
        step=args.step,
        block_snps=args.block_snps,
        pheno_file=args.pheno_file,
        pheno_name=args.pheno_name,
        env_cols=env_cols,
        anchor_cols=anchor_cols,
        anchor_weight=args.anchor_weight,
        force_positive_feature=args.force_positive_feature,
        device=device,
        pair_chunk_size=max(1, int(args.pair_chunk_size)),
        kgxew_fp32=bool(args.kgxew_fp32),
        save_prefix=args.save_files,
        he_do_screen=bool(args.he_screen),
        he_thresh=float(args.he_thresh),
        he_topk=int(args.he_topk),
        he_min_keep=max(1, int(args.he_min_keep)),
        he_save_table=args.he_save_table,
        he_resume_table=args.he_resume_table,
        flip_envs_where_gxe_neg=bool(args.flip_envs_where_gxe_neg),
        cat_max_levels=int(args.cat_max_levels),
        flip_report=bool(args.flip_report),
        cv_variants=bool(args.cv_variants),
        cv_z=float(args.cv_z),
        l1_tau=args.l1_tau,
        l1_anneal=args.l1_anneal,
        l1_tau_min=args.l1_tau_min,
        nonneg=nonneg,
        lifestyle_envs=bool(args.lifestyle_envs),
    )

    print("=== Results ===")
    print(f"N={out['N']}, M_used={out['M_used']}, L={out['L']}")
    if out.get("alpha_true") is not None:
        print("cos(true α, est α)   :", f"{out['cos_alpha']:.4f}")
        print("RMSE(env score)      :", f"{out['rmse_env_score']:.4e}")
        print("α_true (first 8)     :", np.array2string(np.asarray(out['alpha_true'])[:8],
              precision=4, suppress_small=True))
    else:
        print("cos(true α, est α)   : (real phenotype) N/A")
        print("RMSE(env score)      : (real phenotype) N/A")
        print("α_true               : (real phenotype) N/A")

    print("σ̂ (g, gxe, nxe, e)  :", np.array2string(out["sigma_hat"],
          formatter={'float_kind':lambda x: f'{float(x):.4f}'}))

    print("SE_fixed α           :", np.array2string(np.asarray(out["sigma_hat_se_fixed_alpha"]),
          formatter={'float_kind':lambda x: f'{float(x):.4f}'}))
    print("SE_Δα                :", np.array2string(np.asarray(out["sigma_hat_se_alpha_delta"]),
          formatter={'float_kind':lambda x: f'{float(x):.4f}'}))
    print("SE_total             :", np.array2string(np.asarray(out["sigma_hat_se_total"]),
          formatter={'float_kind':lambda x: f'{float(x):.4f}'}))

    ah_sel = np.asarray(out.get("alpha_hat_selected")) if out.get("alpha_hat_selected") is not None else np.array([])
    if ah_sel.size > 0:
        print("α_hat (selected; original basis, first 8):",
              np.array2string(ah_sel[:8], precision=4, suppress_small=True))

    ah_full = np.asarray(out.get("alpha_hat_full")) if out.get("alpha_hat_full") is not None else np.array([])
    if ah_full.size > 0:
        print("α_hat (all; original basis, first 8)     :",
              np.array2string(ah_full[:8], precision=4, suppress_small=True))

    print("env_cols_used (after screen) :", ", ".join(out["env_cols_used"]))
    if out.get("selected_cols") is not None:
        print("selected_cols (HE)          :", ", ".join(out["selected_cols"]))
    if out.get("he_sigma_gxe_per_env") is not None:
        print("he_sigma_gxe_per_env (len)  :", len(out["he_sigma_gxe_per_env"]))
    if out.get("cv_variants_used"):
        print(f"(Variant CV) z-gate={out.get('cv_z'):.2f}")
    if out.get("flip_envs_where_gxe_neg"):
        print("flipped_envs                :", ", ".join(out.get("flipped_envs") or []))

    if args.save_files:
        alpha_row = results_to_alpha_row(out)
        if alpha_row.get("alpha_hat_selected") is not None:
            alpha_row.update(expand_alpha_cols({"alpha_hat_selected": alpha_row.get("alpha_hat_selected")}, "alpha_hat_selected"))
        if alpha_row.get("alpha_hat_full") is not None:
            alpha_row.update(expand_alpha_cols({"alpha_hat_full": alpha_row.get("alpha_hat_full")}, "alpha_hat_full"))
        append_row_to_csv(alpha_row, f"{args.save_files}.alpha_summary.csv")

        sigma_row = results_to_sigma_row(out)
        append_row_to_csv(sigma_row, f"{args.save_files}.sigma_summary.csv")

if __name__ == "__main__":
    main()
