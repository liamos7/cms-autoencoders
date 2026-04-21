# Lasso paths and R² analysis of CICADA latent features.
# R² = 1 - SS_res/SS_tot: coefficient of determination for predicting a target
# (teacher_score / total_et / nPV) from the 80-dimensional latent space via OLS.
# Baseline R²=0 corresponds to always predicting the sample mean (no model).
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
from sklearn.linear_model import lasso_path, LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'axes.titlepad': 10,
    'axes.labelpad': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# extraction

def load_from_hdf5(filepath, n_events=None):
    """
    Load latent vectors and observables from an HDF5 file.

    Keys used: teacher_latent, teacher_score, total_et, nPV,
               first_jet_eta, ht
    """
    sl = slice(None, n_events)
    with h5py.File(filepath, 'r') as f:
        Z             = f['teacher_latent'][sl]
        teacher_score = f['teacher_score'][sl]
        total_et      = f['total_et'][sl]
        nPV           = f['nPV'][sl]
        first_jet_eta = f['first_jet_eta'][sl]
        ht            = f['ht'][sl]

    return (Z.astype(np.float32),
            teacher_score.astype(np.float32),
            total_et.astype(np.float32),
            nPV.astype(np.float32),
            first_jet_eta.astype(np.float32),
            ht.astype(np.float32))

# path fitting

def fit_lasso_path(Z, y, n_alphas=100, eps=1e-4):
    """
    Standardize Z and y, then compute the full lasso path.
    Returns:
        alphas : (n_alphas,)  regularization values, descending
        coefs  : (latent_dim, n_alphas)  coefficient paths
        scaler_Z, scaler_y : fitted StandardScalers
    """
    scaler_Z = StandardScaler().fit(Z)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))

    Z_s = scaler_Z.transform(Z)
    y_s = scaler_y.transform(y.reshape(-1, 1)).ravel()

    alphas, coefs, _ = lasso_path(Z_s, y_s, n_alphas=n_alphas, eps=eps)
    return alphas, coefs, scaler_Z, scaler_y


def fit_lasso_cv(Z, y, cv=5):
    """
    Cross-validated lasso to find optimal α and the active set.
    Returns a dict with summary statistics.
    """
    scaler_Z = StandardScaler().fit(Z)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))

    Z_s = scaler_Z.transform(Z)
    y_s = scaler_y.transform(y.reshape(-1, 1)).ravel()

    model = LassoCV(cv=cv, n_alphas=100, eps=1e-4, max_iter=10000, n_jobs=1)
    model.fit(Z_s, y_s)

    active_mask = model.coef_ != 0
    active_dims = np.where(active_mask)[0]

    # R² = 1 - SS_res/SS_tot: fraction of target variance explained by the
    # active latent dims; R²=0 means predicting the mean for every event.
    return {
        'alpha_cv': model.alpha_,
        'r2': model.score(Z_s, y_s),
        'n_active': active_mask.sum(),
        'active_dims': active_dims,
        'coefs': model.coef_,
    }


def get_entry_order(coefs, alphas):
    """
    From a lasso path, return dimensions ordered by when they first enter
    (largest α at which coefficient becomes nonzero).
    """
    latent_dim = coefs.shape[0]
    entry_alphas = []
    for d in range(latent_dim):
        nonzero = np.where(np.abs(coefs[d, :]) > 1e-12)[0]
        if len(nonzero) > 0:
            entry_alphas.append((d, alphas[nonzero[0]]))
        else:
            entry_alphas.append((d, 0.0))
    entry_alphas.sort(key=lambda x: -x[1])
    return [d for d, _ in entry_alphas]


# R^2 values

def single_dim_r2(Z, y):
    """
    For each latent dimension, compute R² of a univariate linear regression.
    Returns array of shape (latent_dim,).
    Vectorized: computes all dims at once via closed-form Pearson r².
    """
    # Standardize once across all dims
    Z_s = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-12)
    y_s = (y - y.mean()) / (y.std() + 1e-12)
    # r² for univariate OLS = Pearson r² = (Z_sᵀ y_s / n)²
    r = (Z_s * y_s[:, None]).mean(axis=0)
    return r ** 2


def get_coef_magnitude_order(cv_result):
    """
    Return latent dimensions ordered by |β| from a LassoCV result dict.
    """
    return np.argsort(-np.abs(cv_result['coefs'])).tolist()


def cumulative_r2(Z, y, dim_order, do_cv=True):
    """
    Compute R² using OLS with the first k dimensions (in the given order),
    for k = 1, 2, ..., len(dim_order).
    Optionally computes 5-fold CV R² at selected checkpoints.

    Train R² is computed via QR decomposition: reuse the previous QR factorization
    by appending one column at a time (Gram-Schmidt), avoiding a fresh fit each step.
    """
    scaler_Z = StandardScaler().fit(Z)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    Z_s = scaler_Z.transform(Z)
    y_s = scaler_y.transform(y.reshape(-1, 1)).ravel()

    ss_tot = np.dot(y_s, y_s)  # y_s is zero-mean after StandardScaler
    n = len(y_s)

    r2_train = []
    r2_cv = []

    # Incrementally build orthonormal basis Q via Gram-Schmidt
    Q = np.empty((n, 0), dtype=np.float64)
    for k, d in enumerate(dim_order, 1):
        col = Z_s[:, d].astype(np.float64)
        # Orthogonalize against existing Q columns
        col = col - Q @ (Q.T @ col)
        norm = np.linalg.norm(col)
        if norm > 1e-12:
            col /= norm
            Q = np.column_stack([Q, col]) if Q.shape[1] > 0 else col[:, None]

        # Train R²: ||Q Qᵀ y||² / ||y||²
        proj = Q @ (Q.T @ y_s)
        r2_train.append(float(np.dot(proj, proj) / ss_tot))

        if do_cv and (k <= 20 or k % 5 == 0 or k == len(dim_order)):
            dims = dim_order[:k]
            cv_scores = cross_val_score(LinearRegression(), Z_s[:, dims], y_s,
                                        cv=5, scoring='r2')
            r2_cv.append((k, cv_scores.mean()))

    return r2_train, r2_cv


# plotting lasso

def plot_lasso_paths(alphas, coefs, target_name, out_dir, top_k=5,
                     alpha_cv=None, cv_coefs=None):
    """Coefficient paths vs log(α).

    Labels the first top_k entries.  If alpha_cv and cv_coefs are supplied:
      - draws a vertical line at the CV-optimal α
      - highlights active dims (nonzero at CV α) with solid coloured lines;
        all truly-zero dims stay grey
      - annotates active dim indices at the right edge of the plot
      - prints the selected set in a text box on the figure
    """
    latent_dim = coefs.shape[0]
    entry_order = get_entry_order(coefs, alphas)
    first_entries = entry_order[:top_k]

    # Determine which dims are active at the CV-optimal α
    active_at_cv = set()
    if cv_coefs is not None:
        active_at_cv = set(int(d) for d in np.where(np.abs(cv_coefs) > 1e-12)[0])

    # Use a qualitative colormap for active dims
    n_active_total = len(active_at_cv)
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(max(n_active_total, 1))
    active_color = {d: cmap(i) for i, d in enumerate(sorted(active_at_cv))}

    fig, ax = plt.subplots(figsize=(13, 7))
    log_alphas = np.log10(alphas)
    log_alpha_cv = np.log10(alpha_cv) if alpha_cv is not None else None

    # Grey inactive lines first
    for d in range(latent_dim):
        if d not in active_at_cv:
            ax.plot(log_alphas, coefs[d, :], color='#cccccc', alpha=0.5,
                    linewidth=0.7, zorder=1)

    # Colored active lines on top; annotate top_k at right edge only
    for rank, d in enumerate(first_entries):
        if d not in active_at_cv:
            continue
        color = active_color[d]
        ax.plot(log_alphas, coefs[d, :], color=color, linewidth=2.2,
                alpha=0.95, zorder=3)
        ax.annotate(f'$z_{{{d}}}$ (#{rank+1})',
                    xy=(log_alphas[-1], coefs[d, -1]),
                    xytext=(log_alphas[-1] + 0.05, coefs[d, -1]),
                    fontsize=9, color=color, va='center', clip_on=False)

    for d in sorted(active_at_cv):
        if d in first_entries:
            continue
        color = active_color[d]
        ax.plot(log_alphas, coefs[d, :], color=color, linewidth=1.6,
                alpha=0.75, zorder=2)

    # CV-optimal α line
    if log_alpha_cv is not None:
        ax.axvline(log_alpha_cv, color='black', linestyle='--', linewidth=1.8,
                   label=f'CV-optimal $\\alpha$ = {alpha_cv:.2e}', zorder=4)
        ax.legend(framealpha=0.85, edgecolor='0.7')

    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel(r'$\log_{10}(\alpha)$  [larger $\alpha$ $\to$ more sparsity]')
    ax.set_ylabel('Standardised Lasso Coefficient')
    ax.set_title(
        f'Lasso Regularisation Paths — predicting {target_name}\n'
        r'Coloured: $\beta \neq 0$ at CV-optimal $\alpha$;  grey: exactly zero')
    ax.grid(True, alpha=0.2, linestyle='--')

    # Compact text box
    if active_at_cv:
        sorted_active = sorted(active_at_cv)
        n_active = len(sorted_active)
        box_text = f'CV-selected: {n_active} / {latent_dim} dims\n{sorted_active}'
        ax.text(0.02, 0.02, box_text, transform=ax.transAxes,
                fontsize=8, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffde7',
                          edgecolor='0.7', alpha=0.9))

    fig.tight_layout(pad=1.5)
    path = os.path.join(out_dir, f'lasso_path_{target_name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close(fig)


def plot_active_set(cv_result, target_name, out_dir):
    """Bar chart of Lasso coefficients at the CV-optimal α.

    Clearly distinguishes exactly-zero dims (grey, left panel) from the
    nonzero selected set (coloured, right panel), so sparsity is visually
    unambiguous.
    """
    coefs     = cv_result['coefs']          # shape (latent_dim,)
    active    = cv_result['active_dims']    # indices with |β| > 0
    latent_dim = len(coefs)

    active_set = sorted(active.tolist())
    zero_set   = [d for d in range(latent_dim) if d not in set(active_set)]

    n_active_dims = len(active_set)
    fig_h = max(6, 0.30 * n_active_dims + 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_h),
                                   gridspec_kw={'width_ratios': [3, 1]})

    # Left: all dims — magnitude-colored active, grey zero
    abs_max = np.abs(coefs[active_set]).max() if active_set else 1.0
    norm = mcolors.Normalize(vmin=0, vmax=abs_max)
    bar_colors_all = [
        plt.cm.Reds(norm(abs(coefs[d]))) if d in set(active_set) else '#e0e0e0'
        for d in range(latent_dim)
    ]
    ax1.bar(range(latent_dim), coefs, color=bar_colors_all, width=1.0, edgecolor='none')
    ax1.axhline(0, color='k', linewidth=0.6)
    ax1.set_xlabel('Latent Dimension Index')
    ax1.set_ylabel(r'Lasso $\hat{\beta}$ at CV-optimal $\alpha$')
    ax1.set_title(
        f'Lasso Coefficients — predicting {target_name}\n'
        f'CV $\\alpha$ = {cv_result["alpha_cv"]:.2e}  |  '
        f'{len(active_set)} / {latent_dim} selected,  '
        f'{len(zero_set)} exactly zero')
    ax1.set_xlim(-0.5, latent_dim - 0.5)
    ax1.grid(True, alpha=0.2, linestyle='--', axis='y')

    # Right: active dims sorted by |β|
    if active_set:
        order = np.argsort(-np.abs(coefs[active_set]))
        ordered_dims  = [active_set[i] for i in order]
        ordered_coefs = coefs[ordered_dims]
        bar_colors = ['#e6194b' if c > 0 else '#4363d8' for c in ordered_coefs]
        bars = ax2.barh(range(len(ordered_dims)), ordered_coefs,
                        color=bar_colors, edgecolor='none')
        ax2.set_yticks(range(len(ordered_dims)))
        ax2.set_yticklabels([f'$z_{{{d}}}$' for d in ordered_dims], fontsize=9)
        ax2.axvline(0, color='k', linewidth=0.6)
        ax2.set_xlabel(r'$\hat{\beta}$')
        ax2.set_title(f'Active Set ({len(active_set)} dims)\nSorted by $|\\beta|$')
        ax2.grid(True, alpha=0.2, linestyle='--', axis='x')
        x_range = np.abs(ordered_coefs).max() if len(ordered_coefs) else 1
        for bar, val in zip(bars, ordered_coefs):
            xpos = bar.get_width() + (0.02 * x_range if val >= 0 else -0.02 * x_range)
            ha   = 'left' if val >= 0 else 'right'
            ax2.text(xpos, bar.get_y() + bar.get_height() / 2,
                     f'{val:+.3f}', va='center', ha=ha, fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No active dims', transform=ax2.transAxes,
                 ha='center', va='center', fontsize=12)
        ax2.set_title('Active Set')

    fig.tight_layout(pad=1.5)
    path = os.path.join(out_dir, f'active_set_{target_name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close(fig)


def plot_n_active(alphas, coefs, target_name, out_dir, alpha_cv=None):
    """Number of active (nonzero) dimensions vs log(α)."""
    n_active = (np.abs(coefs) > 1e-12).sum(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    log_alphas = np.log10(alphas)
    ax.plot(log_alphas, n_active, linewidth=2, color='#4363d8')

    if alpha_cv is not None:
        ax.axvline(np.log10(alpha_cv), color='#e6194b', linestyle='--', linewidth=1.8,
                   label=f'CV optimal ($\\alpha={alpha_cv:.2e}$)')
        ax.legend(framealpha=0.85, edgecolor='0.7')

    ax.set_xlabel(r'$\log_{10}(\alpha)$')
    ax.set_ylabel('Number of Active Latent Dimensions')
    ax.set_title(f'Latent Sparsity — predicting {target_name}')
    ax.grid(True, alpha=0.2, linestyle='--')

    fig.tight_layout(pad=1.5)
    path = os.path.join(out_dir, f'n_active_{target_name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close(fig)


# plotting R^2

def plot_single_dim_r2(r2s, target_name, out_dir, top_k=10):
    """Bar chart of per-dimension R²."""
    latent_dim = len(r2s)
    order = np.argsort(-r2s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    norm = mcolors.Normalize(vmin=r2s.min(), vmax=r2s.max())
    bar_colors_all = [
        plt.cm.Reds(norm(r2s[d])) if d in order[:top_k] else '#d0d8e8'
        for d in range(latent_dim)
    ]
    ax1.bar(range(latent_dim), r2s, color=bar_colors_all, width=1.0, edgecolor='none')
    ax1.set_xlabel('Latent Dimension Index')
    ax1.set_ylabel(r'$R^2$ (single-dim OLS)')
    ax1.set_title(
        f'Per-dimension $R^2$ — predicting {target_name}\n'
        r'Univariate OLS; top-' + str(top_k) + r' highlighted; baseline $R^2=0$ = mean prediction')
    ax1.set_xlim(-0.5, latent_dim - 0.5)
    ax1.grid(True, alpha=0.2, linestyle='--', axis='y')

    top_dims = order[:top_k]
    top_r2 = r2s[top_dims]
    norm2 = mcolors.Normalize(vmin=top_r2.min(), vmax=top_r2.max())
    top_colors = [plt.cm.Oranges(norm2(v)) for v in top_r2]
    bars = ax2.barh(range(top_k-1, -1, -1), top_r2, color=top_colors, edgecolor='none')
    ax2.set_yticks(range(top_k-1, -1, -1))
    ax2.set_yticklabels([f'$z_{{{d}}}$' for d in top_dims], fontsize=11)
    ax2.set_xlabel(r'$R^2$')
    ax2.set_title(f'Top {top_k} Single-dim Predictors\nof {target_name}')
    ax2.grid(True, alpha=0.2, linestyle='--', axis='x')
    x_range = top_r2.max() if len(top_r2) else 1
    for bar, r2 in zip(bars, top_r2):
        ax2.text(bar.get_width() + 0.01 * x_range,
                 bar.get_y() + bar.get_height()/2,
                 f'{r2:.4f}', va='center', fontsize=9)

    fig.tight_layout(pad=1.5)
    path = os.path.join(out_dir, f'single_dim_r2_{target_name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close(fig)


def plot_cumulative_r2(r2_train_entry, r2_cv_entry, r2_train_coef, r2_cv_coef,
                       target_name, out_dir, full_r2=None):
    """Cumulative R² as dimensions are added (Lasso entry order only)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    n = len(r2_train_entry)
    ks = np.arange(1, n + 1)

    ax.plot(ks, r2_train_entry, linewidth=2, color='#4363d8', label='Train $R^2$')
    if r2_cv_entry:
        cv_ks, cv_vals = zip(*r2_cv_entry)
        ax.plot(cv_ks, cv_vals, 'o-', markersize=4, linewidth=2,
                color='#e6194b', label='5-fold CV $R^2$')
    if full_r2 is not None:
        ax.axhline(full_r2, color='0.5', linestyle='--', linewidth=1.5,
                   label=f'All-80-dim $R^2$ = {full_r2:.4f}')
    ax.set_xlabel('Latent Dimensions Added\n(Lasso Entry Order)')
    ax.set_ylabel(r'$R^2 = 1 - SS_\mathrm{res}/SS_\mathrm{tot}$')
    ax.set_title(f'Cumulative $R^2$ — {target_name}\nLasso Entry Order')
    ax.legend(framealpha=0.85, edgecolor='0.7')
    ax.set_xlim(1, n)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2, linestyle='--')

    fig.tight_layout(pad=1.5)
    path = os.path.join(out_dir, f'cumulative_r2_{target_name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close(fig)


# summaries

def print_active_set(cv_result, target_name):
    """Print the exact active set (nonzero β) at the CV-optimal α, sorted by |β|."""
    coefs      = cv_result['coefs']
    active     = sorted(cv_result['active_dims'].tolist())
    n_total    = len(coefs)
    n_active   = len(active)
    zero_dims  = [d for d in range(n_total) if d not in set(active)]

    print(f'\n  --- Sparsity check: active set at CV-optimal α '
          f'(target: {target_name}) ---')
    print(f'  {n_active} / {n_total} dims have β ≠ 0  '
          f'({n_total - n_active} are EXACTLY zero)')
    print(f'  Active dims (sorted by index):  {active}')
    print(f'  Zero dims  ({len(zero_dims)} total):  {zero_dims}')
    print()
    print(f'  {"Dim":>6}  {"β":>10}  {"|β|":>10}  {"rank by |β|":>12}')
    print(f'  {"-"*6}  {"-"*10}  {"-"*10}  {"-"*12}')
    order = sorted(active, key=lambda d: -abs(coefs[d]))
    for rank, d in enumerate(order, 1):
        print(f'  {f"[{d}]":>6}  {coefs[d]:>+10.4f}  {abs(coefs[d]):>10.4f}  {rank:>12}')
    print()


def print_cv_summary(cv_result, target_name):
    """Print the LassoCV summary."""
    print(f'\n{"="*60}')
    print(f'  LassoCV summary — target: {target_name}')
    print(f'{"="*60}')
    print(f'  Optimal α                       : {cv_result["alpha_cv"]:.4e}')
    print(f'  R² = 1 - SS_res/SS_tot          : {cv_result["r2"]:.4f}')
    print(f'  (predicting {target_name} from all active latent dims;')
    print(f'   baseline R²=0 is always predicting the mean)')
    print(f'  Active dimensions               : {cv_result["n_active"]} / {len(cv_result["coefs"])}')
    print(f'  Active entries                  : {cv_result["active_dims"].tolist()}')

    ranked = np.argsort(-np.abs(cv_result['coefs']))
    top = ranked[:min(10, cv_result['n_active'])]
    print(f'  Top dims by |β|   : {top.tolist()}')
    for d in top:
        if cv_result['coefs'][d] != 0:
            print(f'    latent[{d:2d}]  β = {cv_result["coefs"][d]:+.4f}')
    print()


def print_r2_summary(target_name, r2_single, entry_order, coef_order,
                     r2_train_entry, r2_train_coef, full_r2):
    """Print the R² breakdown table.

    All R² values are the coefficient of determination
    R² = 1 - SS_res/SS_tot for predicting the target from latent dims via OLS.
    Baseline (R²=0) is predicting the sample mean for every event.
    """
    print(f'\n{"="*70}')
    print(f'  R² BREAKDOWN — predicting {target_name} from 80 latent dims (OLS)')
    print(f'  R² = 1 - SS_res/SS_tot;  baseline (R²=0) = always predict mean')
    print(f'{"="*70}')
    print(f'  R² using all active LassoCV dims: {full_r2:.4f}')
    print()

    top_single = np.argsort(-r2_single)[:10]
    print(f'  Top 10 single-dimension R² (univariate OLS for {target_name}):')
    for rank, d in enumerate(top_single, 1):
        print(f'    #{rank:2d}  latent[{d:2d}]  R² = {r2_single[d]:.4f}')

    print(f'\n  Dims needed to reach cumulative R² threshold'
          f'  (entry order / |β| order):')
    for threshold in [0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]:
        if full_r2 < threshold:
            print(f'    R² ≥ {threshold:.2f}:  '
                  f'(full model R² = {full_r2:.4f}, cannot reach)')
            continue

        k_entry = next((k for k, r2 in enumerate(r2_train_entry, 1)
                        if r2 >= threshold), None)
        k_coef  = next((k for k, r2 in enumerate(r2_train_coef, 1)
                        if r2 >= threshold), None)
        e_str = f'{k_entry:3d}' if k_entry else ' N/A'
        c_str = f'{k_coef:3d}' if k_coef else ' N/A'
        print(f'    R² ≥ {threshold:.2f}:  {e_str} / {c_str}')

    print(f'\n  First 10 dims (lasso entry):  {entry_order[:10]}')
    print(f'  First 10 dims (|β| rank)  :  {coef_order[:10]}')
    print()


# main

def main():
    parser = argparse.ArgumentParser(
        description='Lasso path + R² breakdown for CICADA latent space')
    parser.add_argument('--data_dir', type=str,
                        default='/scratch/network/lo8603/thesis/fast-ad/data/h5_files/',
                        help='Directory containing HDF5 data files')
    parser.add_argument('--zb_file', type=str, default='zb.h5',
                        help='Zero Bias HDF5 filename')
    parser.add_argument('--out_dir', type=str, default='plots/lasso',
                        help='Output directory for all plots')
    parser.add_argument('--n_events', type=int, default=200000,
                        help='Max events to load (default 200k). '
                             'Lasso paths use all loaded; R² CV uses --n_events_r2.')
    parser.add_argument('--n_events_r2', type=int, default=200000,
                        help='Events for R² breakdown (CV is slow on >200k)')
    parser.add_argument('--n_alphas', type=int, default=100,
                        help='Number of alpha values in lasso path')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top entries to label in path plot')
    parser.add_argument('--skip_cumulative_cv', action='store_true',
                        help='Skip cross-validated R² in cumulative plots '
                             '(much faster)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    zb_path = os.path.join(args.data_dir, args.zb_file)
    print(f'Loading from {zb_path}...')
    Z, teacher_scores, total_et, nPV, first_jet_eta, ht = load_from_hdf5(
        zb_path, n_events=args.n_events)
    print(f'  Events: {len(Z)}  |  Latent dim: {Z.shape[1]}')

    targets = {
        'teacher_score': teacher_scores,
        'total_et':      total_et,
        'nPV':           nPV,
        'first_jet_eta': first_jet_eta,
        'ht':            ht,
    }

    # ── Subsample for R² breakdown if needed ─────────────────────────────────
    n_r2 = min(args.n_events_r2, len(Z))
    rng = np.random.RandomState(42)
    r2_idx = rng.choice(len(Z), n_r2, replace=False)
    Z_r2 = Z[r2_idx]
    targets_r2 = {name: y[r2_idx] for name, y in targets.items()}
    print(f'  R² breakdown will use {n_r2} events (subsampled)')


    print('\n' + '='*70)
    print('  PART 1: LASSO PATHS')
    print('='*70)

    cv_results = {}
    path_results = {}

    for name, y in targets.items():
        print(f'\n--- Fitting lasso path for target: {name} ---')

        mask = np.isfinite(y)
        Z_c, y_c = Z[mask], y[mask]

        # Fit path
        alphas, coefs, _, _ = fit_lasso_path(Z_c, y_c, n_alphas=args.n_alphas)
        path_results[name] = (alphas, coefs)

        # Fit CV model
        cv_result = fit_lasso_cv(Z_c, y_c)
        cv_results[name] = cv_result

        # Print summary
        print_cv_summary(cv_result, name)
        print_active_set(cv_result, name)

        # Plot
        plot_lasso_paths(alphas, coefs, name, args.out_dir, top_k=args.top_k,
                         alpha_cv=cv_result['alpha_cv'],
                         cv_coefs=cv_result['coefs'])
        plot_active_set(cv_result, name, args.out_dir)
        plot_n_active(alphas, coefs, name, args.out_dir,
                      alpha_cv=cv_result['alpha_cv'])

    # Entry-order summary across targets
    print('\n' + '='*60)
    print('  Entry order summary (which dims enter first per target)')
    print('='*60)
    for name, (alphas, coefs) in path_results.items():
        entry_order = get_entry_order(coefs, alphas)
        print(f'  {name:20s} → first 5 entries: {entry_order[:5]}')


    print('\n' + '='*70)
    print(f'  PART 2: R² BREAKDOWN  (N = {n_r2})')
    print('='*70)

    do_cv = not args.skip_cumulative_cv

    for name, y in targets_r2.items():
        mask = np.isfinite(y)
        Z_c, y_c = Z_r2[mask], y[mask]

        print(f'\n{"#"*70}')
        print(f'  Analyzing target: {name}  (N={len(y_c)})')
        print(f'{"#"*70}')

        # Single-dim R²
        print('  Computing single-dimension R² ...')
        r2_single = single_dim_r2(Z_c, y_c)
        plot_single_dim_r2(r2_single, name, args.out_dir)

        # Entry order (from full-data path results)
        alphas, coefs = path_results[name]
        entry_order = get_entry_order(coefs, alphas)

        # |β| rank order (from full-data CV results)
        coef_order = get_coef_magnitude_order(cv_results[name])
        full_r2 = cv_results[name]['r2']

        # Cumulative R²
        print(f'  Computing cumulative R² (entry order, cv={do_cv}) ...')
        r2_train_entry, r2_cv_entry = cumulative_r2(Z_c, y_c, entry_order,
                                                     do_cv=do_cv)

        print(f'  Computing cumulative R² (|β| order, cv={do_cv}) ...')
        r2_train_coef, r2_cv_coef = cumulative_r2(Z_c, y_c, coef_order,
                                                    do_cv=do_cv)

        # Plot
        plot_cumulative_r2(r2_train_entry, r2_cv_entry,
                           r2_train_coef, r2_cv_coef,
                           name, args.out_dir, full_r2=full_r2)

        # Summary
        print_r2_summary(name, r2_single, entry_order, coef_order,
                         r2_train_entry, r2_train_coef, full_r2)

    print(f'\nAll plots saved to {args.out_dir}/')


if __name__ == '__main__':
    main()