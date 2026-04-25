"""
gmm_em.py

EM algorithm for a Gaussian Mixture Model with isotropic covariance.
Designed for clustering GPT-2 hidden states for unsupervised POS induction.

Isotropic covariance means each component k has a single scalar variance:
    Sigma_k = sigma_k^2 * I

So the Gaussian density simplifies to:
    log N(h | mu_k, sigma_k^2 I) = -d/2 * log(2*pi)
                                  - d/2 * log(sigma_k^2)
                                  - 1/(2*sigma_k^2) * ||h - mu_k||^2

Regularization adds epsilon to sigma_k^2 after each M-step to prevent
variance from collapsing to zero when a component captures very few tokens.

Usage:
    python gmm_em.py --states states.npy --k 15 --output gmm.npz

Requirements:
    pip install numpy tqdm
"""

import argparse
import numpy as np
from tqdm import tqdm
import gzip

# ── Constants ─────────────────────────────────────────────────────────────────

LOG_2PI   = np.log(2.0 * np.pi)
EPS       = 1e-6    # covariance regularization floor
MIN_SIGMA = 1e-3    # hard minimum on sigma^2 to prevent degeneracy


# ── Initialisation ────────────────────────────────────────────────────────────

def _init_kmeans_plusplus(
    states : np.ndarray,   # (N, d)
    K      : int,
    rng    : np.random.Generator,
) -> np.ndarray:
    """
    K-means++ initialisation for component means.

    Algorithm:
      1. Pick the first centre uniformly at random.
      2. For each subsequent centre, pick a point with probability
         proportional to its squared distance to the nearest existing centre.

    This spreads the initial centres out, reducing the chance of two
    components starting on top of each other and then collapsing.
    """
    N, d = states.shape
    centres = np.empty((K, d), dtype=np.float64)

    # Step 1 — first centre chosen uniformly
    centres[0] = states[rng.integers(N)]

    for k in range(1, K):
        # Squared distance from each point to its nearest existing centre
        # Shape: (N, k) -> min over k -> (N,)
        diffs   = states[:, None, :] - centres[:k][None, :, :]  # (N, k, d)
        sq_dist = (diffs ** 2).sum(axis=2)                       # (N, k)
        min_sq  = sq_dist.min(axis=1)                            # (N,)

        # Sample proportional to squared distance
        probs      = min_sq / min_sq.sum()
        centres[k] = states[rng.choice(N, p=probs)]

    return centres


def initialise(
    states : np.ndarray,   # (N, d)
    K      : int,
    rng    : np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialise GMM parameters using K-means++ for means.

    Returns:
        pi     : (K,)   mixing weights, uniform at start
        mu     : (K, d) component means from K-means++
        sigma2 : (K,)   per-component variances, initialised to global variance
    """
    N, d = states.shape

    mu     = _init_kmeans_plusplus(states, K, rng)           # (K, d)
    pi     = np.full(K, 1.0 / K)                            # (K,)
    sigma2 = np.full(K, states.var() + EPS)                 # (K,)

    return pi, mu, sigma2


# ── Log-likelihood helpers ────────────────────────────────────────────────────

def _log_gaussian_isotropic(
    states : np.ndarray,   # (N, d)
    mu     : np.ndarray,   # (K, d)
    sigma2 : np.ndarray,   # (K,)
) -> np.ndarray:           # (N, K)
    """
    Compute log N(h_t | mu_k, sigma_k^2 * I) for all t and k.

    log N(h | mu, s^2 I) = -d/2 * log(2*pi)
                          - d/2 * log(s^2)
                          - ||h - mu||^2 / (2 * s^2)

    Computed efficiently without forming (N, K, d) intermediate arrays
    by expanding the squared norm:
        ||h - mu||^2 = ||h||^2 - 2 h·mu + ||mu||^2
    giving shapes (N,1) - (N,K) + (1,K) = (N,K).
    """
    N, d = states.shape
    K    = mu.shape[0]

    # ||h_t||^2  shape (N, 1)
    h_sq = (states ** 2).sum(axis=1, keepdims=True)

    # -2 * h_t · mu_k  shape (N, K)
    cross = -2.0 * states @ mu.T

    # ||mu_k||^2  shape (1, K)
    mu_sq = (mu ** 2).sum(axis=1, keepdims=True).T

    # squared Euclidean distances  (N, K)
    sq_dist = h_sq + cross + mu_sq

    # per-component log-density  (N, K)
    log_norm = -0.5 * d * LOG_2PI \
               - 0.5 * d * np.log(sigma2)[None, :] \
               - 0.5 * sq_dist / sigma2[None, :]

    return log_norm                                         # (N, K)


def _log_responsibilities(
    states : np.ndarray,   # (N, d)
    pi     : np.ndarray,   # (K,)
    mu     : np.ndarray,   # (K, d)
    sigma2 : np.ndarray,   # (K,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    E-step: compute log responsibilities and per-token log-likelihood.

    log gamma_t(k) = log pi_k + log N(h_t | mu_k, sigma_k^2 I)
                     - logsumexp_j [log pi_j + log N(h_t | mu_j, sigma_j^2 I)]

    Uses the log-sum-exp trick for numerical stability:
        logsumexp(a) = max(a) + log sum(exp(a - max(a)))

    Returns:
        log_gamma : (N, K)  log responsibilities (log-normalised)
        log_lik   : (N,)    per-token log p(h_t | theta)
    """
    log_pi    = np.log(pi + 1e-300)                        # (K,)
    log_gauss = _log_gaussian_isotropic(states, mu, sigma2) # (N, K)

    # unnormalised log joint  log [pi_k * N(h_t | mu_k, sigma_k^2)]
    log_joint = log_pi[None, :] + log_gauss                # (N, K)

    # logsumexp over components for numerical stability
    log_joint_max = log_joint.max(axis=1, keepdims=True)   # (N, 1)
    log_sum       = log_joint_max[:, 0] + np.log(
                        np.exp(log_joint - log_joint_max).sum(axis=1)
                    )                                       # (N,)

    log_gamma = log_joint - log_sum[:, None]               # (N, K)

    return log_gamma, log_sum


# ── EM steps ──────────────────────────────────────────────────────────────────

def e_step(
    states : np.ndarray,   # (N, d)
    pi     : np.ndarray,   # (K,)
    mu     : np.ndarray,   # (K, d)
    sigma2 : np.ndarray,   # (K,)
) -> tuple[np.ndarray, float]:
    """
    E-step: compute soft assignments (responsibilities) for each token.

    gamma_t(k) = p(z_t = k | h_t, theta)
               = pi_k * N(h_t | mu_k, sigma_k^2 I)
                 / sum_j pi_j * N(h_t | mu_j, sigma_j^2 I)

    Returns:
        gamma    : (N, K)  responsibilities (probabilities, sum to 1 over k)
        total_ll : float   total log-likelihood sum_t log p(h_t | theta)
    """
    log_gamma, log_lik = _log_responsibilities(states, pi, mu, sigma2)
    gamma    = np.exp(log_gamma)                           # (N, K)
    total_ll = log_lik.sum()
    return gamma, total_ll


def m_step(
    states : np.ndarray,   # (N, d)
    gamma  : np.ndarray,   # (N, K)
    eps    : float = EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    M-step: update parameters given the soft assignments.

    For isotropic covariance Sigma_k = sigma_k^2 * I the closed-form
    updates are:

        N_k      = sum_t gamma_t(k)                     effective count
        pi_k     = N_k / N                              mixing weight
        mu_k     = (1/N_k) sum_t gamma_t(k) h_t         weighted mean
        sigma_k^2 = (1/(N_k*d)) sum_t gamma_t(k)||h_t - mu_k||^2
                    + eps                               regularized variance

    The eps floor prevents sigma_k^2 collapsing to zero when a component
    has very few tokens assigned to it.

    Returns:
        pi     : (K,)   updated mixing weights
        mu     : (K, d) updated means
        sigma2 : (K,)   updated variances (regularised)
    """
    N, d = states.shape
    K    = gamma.shape[1]

    # Effective count per component  (K,)
    Nk = gamma.sum(axis=0)                                 # (K,)
    # Guard against empty components
    Nk = np.maximum(Nk, 1e-300)

    # Mixing weights  (K,)
    pi = Nk / N

    # Weighted means  (K, d)
    # mu_k = (1/N_k) * sum_t gamma_t(k) * h_t
    mu = (gamma.T @ states) / Nk[:, None]                  # (K, d)

    # Isotropic variance  (K,)
    # sigma_k^2 = (1/(N_k*d)) * sum_t gamma_t(k) * ||h_t - mu_k||^2
    #
    # Expand: ||h_t - mu_k||^2 = ||h_t||^2 - 2 h_t·mu_k + ||mu_k||^2
    h_sq   = (states ** 2).sum(axis=1)                     # (N,)
    cross  = (states @ mu.T)                               # (N, K)
    mu_sq  = (mu ** 2).sum(axis=1)                         # (K,)

    # weighted sum of ||h_t - mu_k||^2 per component  (K,)
    w_sq_dist = (
        (gamma * h_sq[:, None]).sum(axis=0)                # sum_t gamma_t(k)||h_t||^2
        - 2.0 * (gamma * cross).sum(axis=0)                # -2 sum_t gamma_t(k) h_t·mu_k
        + Nk * mu_sq                                       # N_k * ||mu_k||^2
    )

    sigma2 = w_sq_dist / (Nk * d)

    # ── Regularisation ────────────────────────────────────────────────────
    # Add eps to every component's variance to prevent singular covariances.
    # Also enforce a hard minimum to catch any numerical underflow.
    sigma2 = sigma2 + eps
    sigma2 = np.maximum(sigma2, MIN_SIGMA)

    return pi, mu, sigma2


# ── Collapse detection and recovery ──────────────────────────────────────────

def detect_collapsed_components(
    gamma   : np.ndarray,   # (N, K)
    min_mass: float = 1.0,  # minimum effective count to be considered alive
) -> list[int]:
    """
    Return indices of components whose effective count fell below min_mass.
    A collapsed component has attracted essentially no tokens and will
    produce degenerate parameters on subsequent iterations.
    """
    Nk = gamma.sum(axis=0)                                 # (K,)
    return [k for k in range(len(Nk)) if Nk[k] < min_mass]


def reinitialise_collapsed(
    states   : np.ndarray,  # (N, d)
    mu       : np.ndarray,  # (K, d)
    sigma2   : np.ndarray,  # (K,)
    gamma    : np.ndarray,  # (N, K)
    collapsed: list[int],
    rng      : np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reinitialise collapsed components by splitting the largest component.

    Strategy: find the component with the highest effective count (the
    'richest'), perturb its mean slightly to create two distinct centres,
    and assign the collapsed component one of those perturbed centres.
    This avoids restarting from scratch while breaking the symmetry.
    """
    if not collapsed:
        return mu, sigma2

    Nk      = gamma.sum(axis=0)                            # (K,)
    richest = int(Nk.argmax())

    for k in collapsed:
        noise      = np.random.default_rng().normal(
                         scale=np.sqrt(sigma2[richest]) * 0.1,
                         size=mu.shape[1]
                     )
        mu[k]      = mu[richest] + noise
        sigma2[k]  = sigma2[richest]
        print(f"  [reinit] component {k} collapsed → "
              f"split from component {richest}")

    return mu, sigma2


# ── Full EM loop ──────────────────────────────────────────────────────────────

def fit_gmm(
    states     : np.ndarray,        # (N, d)
    K          : int,
    max_iter   : int   = 100,
    tol        : float = 1e-4,      # convergence threshold on log-likelihood
    eps        : float = EPS,       # covariance regularisation
    n_init     : int   = 5,         # number of random restarts
    min_mass   : float = 1.0,       # min effective count before reinit
    seed       : int   = 42,
    verbose    : bool  = True,
) -> dict:
    """
    Fit a GMM with isotropic covariance using EM with multiple restarts.

    Multiple restarts (n_init) are run and the solution with the highest
    final log-likelihood is returned, reducing sensitivity to initialisation.

    Args:
        states   : Hidden state matrix (N, d).
        K        : Number of mixture components (= number of POS tags).
        max_iter : Maximum EM iterations per restart.
        tol      : Stop early if |delta log-likelihood| < tol per token.
        eps      : Regularisation added to sigma^2 each M-step.
        n_init   : Number of independent random restarts.
        min_mass : Components with effective count below this are reinitialised.
        seed     : Base random seed (each restart uses seed + restart_index).
        verbose  : Print progress.

    Returns dict with keys:
        pi       : (K,)   final mixing weights
        mu       : (K, d) final component means
        sigma2   : (K,)   final per-component variances
        gamma    : (N, K) final responsibilities
        log_liks : list of per-iteration total log-likelihoods
        best_ll  : float  final total log-likelihood of the best run
    """
    N, d   = states.shape
    best   = None

    for restart in range(n_init):
        rng = np.random.default_rng(seed + restart)
        if verbose:
            print(f"\n── Restart {restart + 1}/{n_init} ──────────────────")

        # Initialise parameters
        if verbose:
            print(" - initialise")
        pi, mu, sigma2 = initialise(states[:200_000], K, rng)
        log_liks       = []
        prev_ll        = -np.inf

        pbar = tqdm(range(max_iter), desc="EM", disable=not verbose)
        for it in pbar:

            # ── E-step ────────────────────────────────────────────────────
            gamma, total_ll = e_step(states, pi, mu, sigma2)
            log_liks.append(total_ll)

            # ── Collapse check ────────────────────────────────────────────
            collapsed = detect_collapsed_components(gamma, min_mass)
            if collapsed:
                mu, sigma2 = reinitialise_collapsed(
                    states, mu, sigma2, gamma, collapsed, rng
                )
                # Recompute responsibilities after reinit
                gamma, total_ll = e_step(states, pi, mu, sigma2)

            # ── M-step ────────────────────────────────────────────────────
            pi, mu, sigma2 = m_step(states, gamma, eps)

            # ── Convergence check ─────────────────────────────────────────
            ll_per_token = total_ll / N
            delta        = abs(total_ll - prev_ll) / N
            pbar.set_postfix({
                "ll/tok": f"{ll_per_token:.4f}",
                "delta" : f"{delta:.2e}",
            })

            if it > 0 and delta < tol:
                if verbose:
                    print(f"  Converged at iteration {it + 1}  "
                          f"(delta={delta:.2e} < tol={tol})")
                break

            prev_ll = total_ll

        # Keep the best restart
        if best is None or total_ll > best["best_ll"]:
            best = {
                "pi"      : pi,
                "mu"      : mu,
                "sigma2"  : sigma2,
                "gamma"   : gamma,
                "log_liks": log_liks,
                "best_ll" : total_ll,
            }
            if verbose:
                print(f"  New best log-likelihood: {total_ll:.2f}")

    if verbose:
        print(f"\nBest total log-likelihood: {best['best_ll']:.2f}")
        print(f"Best ll/token:             {best['best_ll'] / N:.4f}")

    return best


# ── Decoding ──────────────────────────────────────────────────────────────────

def decode(gamma: np.ndarray) -> np.ndarray:
    """
    Hard tag assignment: z_t = argmax_k gamma_t(k).

    Returns:
        tags : (N,) integer array of component indices
    """
    return gamma.argmax(axis=1)


def cluster_stats(
    gamma  : np.ndarray,   # (N, K)
    sigma2 : np.ndarray,   # (K,)
    tokens : list[str] | None = None,
    top_n  : int = 10,
) -> None:
    """
    Print per-component statistics for inspection.
    If tokens are provided, also print the most common tokens per cluster.
    """
    K  = gamma.shape[1]
    Nk = gamma.sum(axis=0)
    H  = -(gamma * np.log(gamma + 1e-300)).sum(axis=1).mean()

    print(f"\n{'─'*60}")
    print(f"{'Component':>10} {'Eff.count':>10} {'Weight':>8} {'Sigma^2':>10}")
    print(f"{'─'*60}")
    for k in range(K):
        print(f"{k:>10} {Nk[k]:>10.1f} {Nk[k]/Nk.sum():>8.4f} "
              f"{sigma2[k]:>10.6f}")

    print(f"\nMean posterior entropy H(gamma_t): {H:.4f}  "
          f"(max={np.log(K):.4f} = uniform)")

    if tokens is not None:
        tags = decode(gamma)
        print(f"\nTop-{top_n} tokens per component:")
        for k in range(K):
            idx     = np.where(tags == k)[0]
            toks    = [tokens[i] for i in idx]
            from collections import Counter
            common  = Counter(toks).most_common(top_n)
            preview = " | ".join(f"{t!r}:{c}" for t, c in common)
            print(f"  [{k:2d}] {preview}")


# ── Persistence ───────────────────────────────────────────────────────────────

def save_gmm(path: str, result: dict) -> None:
    """Save GMM parameters and responsibilities to a .npz file."""
    np.savez(
        path,
        pi      = result["pi"],
        mu      = result["mu"],
        sigma2  = result["sigma2"],
        gamma   = result["gamma"],
        log_liks= np.array(result["log_liks"]),
    )
    print(f"Saved GMM → '{path}'")


def load_gmm(path: str) -> dict:
    """Load GMM parameters from a .npz file."""
    data = np.load(path)
    return {
        "pi"      : data["pi"],
        "mu"      : data["mu"],
        "sigma2"  : data["sigma2"],
        "gamma"   : data["gamma"],
        "log_liks": data["log_liks"].tolist(),
        "best_ll" : float(data["log_liks"][-1]),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EM for isotropic GMM on GPT-2 hidden states"
    )
    p.add_argument("--states",   required=True,
                   help="Path to hidden states .npy file (N, d)")
    p.add_argument("--tokens",   default=None,
                   help="Path to tokens .txt file for cluster inspection")
    p.add_argument("--k",        type=int, default=15,
                   help="Number of mixture components (default: 15)")
    p.add_argument("--max-iter", type=int, default=100,
                   help="Max EM iterations per restart (default: 100)")
    p.add_argument("--tol",      type=float, default=1e-4,
                   help="Convergence tolerance per token (default: 1e-4)")
    p.add_argument("--eps",      type=float, default=EPS,
                   help=f"Covariance regularisation (default: {EPS})")
    p.add_argument("--n-init",   type=int, default=5,
                   help="Number of random restarts (default: 5)")
    p.add_argument("--seed",     type=int, default=42,
                   help="Base random seed (default: 42)")
    p.add_argument("--output",   default="gmm.npz",
                   help="Output path for GMM parameters (default: gmm.npz)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading states from '{args.states}' ...")
    states = np.load(args.states).astype(np.float32)
    print(f"States shape: {states.shape}")

    tokens = None
    if args.tokens:
        with open(args.tokens, "r", encoding="utf-8") as f:
            tokens = f.read().splitlines()
        print(f"Loaded {len(tokens):,} token strings")

    result = fit_gmm(
        states   = states,
        K        = args.k,
        max_iter = args.max_iter,
        tol      = args.tol,
        eps      = args.eps,
        n_init   = args.n_init,
        seed     = args.seed,
        verbose  = True,
    )

    cluster_stats(result["gamma"], result["sigma2"], tokens)
    save_gmm(args.output, result)


if __name__ == "__main__":
    main()


"""

class Args:
    pass

args = Args()
args.states = "/media/HD2/RASOOL/OUTPUTS/pos-pred-gpt-em/states.npy"
args.tokens = "/media/HD2/RASOOL/OUTPUTS/pos-pred-gpt-em/tokens.txt"
args.n_init = 5
K = 15

states = np.load(args.states).astype(np.float32)
with open(args.tokens, "r", encoding="utf-8") as f:
    tokens = f.read().splitlines()

    
print(len(tokens))
print(len(states))


states = states[:100_000]
tokens = tokens[:100_000]
## Run Em
seed = 1
restart = 0
rng = np.random.default_rng(seed + restart)

# Initialise parameters
pi, mu, sigma2 = initialise(states, K, rng)
print(len(pi), len(mu), len(sigma2))


log_liks       = []
prev_ll        = -np.inf

# ── E-step ────────────────────────────────────────────────────
gamma, total_ll = e_step(states, pi, mu, sigma2)


"""