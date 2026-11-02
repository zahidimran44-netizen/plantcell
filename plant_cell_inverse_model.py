"""
Reverse-engineering plant cell mechanical properties: reduced mechanical model + Bayesian inversion.

Forward model (single cell, thin spherical shell, small strain)
  k_wall   = c * E * h^2,   c = 4 / (R * sqrt(3 (1 - nu^2)))      (Reissner point-load stiffness)
  k_turg   = alpha * pi * P * R                                    (phenomenological, probe-dependent alpha)
  k1 (turgid)      = k_wall + k_turg
  k2 (plasmolysed) = k_wall
  eps (osmotic)    = (1 - nu) * P * R / (2 * E * h)                (equibiaxial Hooke law + Laplace law)

Unknowns   x = (ln E, ln P, ln h, ln R, nu, ln alpha)
Data/priors are combined as weighted log-residuals; posterior covariance from the Gauss-Newton
information matrix is checked against repeated synthetic experiments and MCMC.

Units: SI (Pa, m, N/m).  Run:  python plant_cell_inverse_model.py
"""
import numpy as np
from scipy.optimize import least_squares

TRUE = dict(E=20e6, P=0.30e6, h=0.40e-6, R=15e-6, nu=0.30, alpha=0.10)
SIG = dict(k=0.05, eps=0.05, h=0.10, R=0.05, nu=0.05, alpha=0.10)   # 1-sigma (relative, nu absolute)
NAMES = ["lnE", "lnP", "lnh", "lnR", "nu", "lnalpha"]


def forward(x):
    """x = (lnE, lnP, lnh, lnR, nu, lnalpha) -> (k1, k2, eps)."""
    E, P, h, R = np.exp(x[0]), np.exp(x[1]), np.exp(x[2]), np.exp(x[3])
    nu, al = x[4], np.exp(x[5])
    c = 4.0 / (R * np.sqrt(3.0 * (1.0 - nu**2)))
    k2 = c * E * h**2
    k1 = k2 + al * np.pi * P * R
    eps = (1.0 - nu) * P * R / (2.0 * E * h)
    return np.array([k1, k2, eps])


def x_true():
    t = TRUE
    return np.array([np.log(t["E"]), np.log(t["P"]), np.log(t["h"]), np.log(t["R"]), t["nu"], np.log(t["alpha"])])


def residuals(x, obs, scenario):
    """Weighted residual vector. scenario: 'S' single state, 'A' two states, 'B' two states + osmotic strain."""
    k1, k2, eps = forward(x)
    r = [(np.log(k1) - np.log(obs["k1"])) / SIG["k"]]
    if scenario in ("A", "B"):
        r.append((np.log(k2) - np.log(obs["k2"])) / SIG["k"])
    if scenario == "B":
        r.append((np.log(eps) - np.log(obs["eps"])) / SIG["eps"])
    r += [(x[2] - np.log(obs["h"])) / SIG["h"], (x[3] - np.log(obs["R"])) / SIG["R"],
          (x[4] - obs["nu"]) / SIG["nu"], (x[5] - np.log(obs["alpha"])) / SIG["alpha"]]
    return np.array(r)


def jacobian(x, obs, scenario, step=1e-6):
    f0 = residuals(x, obs, scenario)
    J = np.zeros((f0.size, x.size))
    for j in range(x.size):
        xp = x.copy(); xp[j] += step
        J[:, j] = (residuals(xp, obs, scenario) - f0) / step
    return J


def noiseless_obs():
    k1, k2, eps = forward(x_true())
    return dict(k1=k1, k2=k2, eps=eps, h=TRUE["h"], R=TRUE["R"], nu=TRUE["nu"], alpha=TRUE["alpha"])


def posterior_cov(scenario):
    """Laplace/Fisher covariance at the true parameters (exact data)."""
    J = jacobian(x_true(), noiseless_obs(), scenario)
    F = J.T @ J
    return F, np.linalg.pinv(F)


def noisy_obs(rng):
    o = noiseless_obs()
    return dict(k1=o["k1"] * np.exp(SIG["k"] * rng.normal()), k2=o["k2"] * np.exp(SIG["k"] * rng.normal()),
                eps=o["eps"] * np.exp(SIG["eps"] * rng.normal()),
                h=o["h"] * np.exp(SIG["h"] * rng.normal()), R=o["R"] * np.exp(SIG["R"] * rng.normal()),
                nu=o["nu"] + SIG["nu"] * rng.normal(), alpha=o["alpha"] * np.exp(SIG["alpha"] * rng.normal()))


def fit(obs, scenario, x0=None):
    x0 = x_true() + 0.05 if x0 is None else x0
    return least_squares(residuals, x0, args=(obs, scenario), method="trf", xtol=1e-12, ftol=1e-12).x


def closed_form_two_state(k1, k2, R, h=None, nu=0.3, alpha=0.1, eps=None):
    """Algebraic inversion. With h known: E = k2/(c h^2).  With eps instead of h: h = k2/(c s), E = c s^2/k2, s = E h."""
    c = 4.0 / (R * np.sqrt(3.0 * (1.0 - nu**2)))
    P = (k1 - k2) / (alpha * np.pi * R)
    if h is not None:
        return dict(E=k2 / (c * h**2), P=P, h=h)
    s = (1.0 - nu) * (k1 - k2) / (2.0 * alpha * np.pi * eps)
    hh = k2 / (c * s)
    return dict(E=c * s**2 / k2, P=P, h=hh)


def log_post(x, obs, scenario):
    """Unnormalised log posterior with log-uniform bounds on E and P."""
    if not (np.log(1e5) < x[0] < np.log(1e10) and np.log(1e3) < x[1] < np.log(1e7)):
        return -np.inf
    if not (0.0 < x[4] < 0.49):
        return -np.inf
    r = residuals(x, obs, scenario)
    return -0.5 * float(r @ r)


def adaptive_metropolis(obs, scenario, n=120000, seed=1):
    rng = np.random.default_rng(seed)
    x = fit(obs, scenario); lp = log_post(x, obs, scenario)
    d = x.size; cov = np.diag([0.2, 0.2, 0.05, 0.03, 0.03, 0.05]) ** 2
    chain = np.empty((n, d)); acc = 0
    for i in range(n):
        if i > 2000 and i % 500 == 0:
            cov = np.cov(chain[max(0, i - 20000):i].T) * (2.38**2 / d) + 1e-10 * np.eye(d)
        prop = x + rng.multivariate_normal(np.zeros(d), cov)
        lpp = log_post(prop, obs, scenario)
        if np.log(rng.uniform()) < lpp - lp:
            x, lp = prop, lpp; acc += 1
        chain[i] = x
    return chain[n // 4:], acc / n


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    out = {}
    o = noiseless_obs()
    print(f"Noise-free data: k1={o['k1']:.3f} N/m  k2={o['k2']:.3f} N/m  eps={o['eps']:.3f}")

    # 1. Identifiability: eigen-structure of the information matrix on (lnE, lnP) with nuisance parameters free
    for sc in ("S", "A", "B"):
        F, C = posterior_cov(sc)
        ev = np.linalg.eigvalsh(F)
        sd = np.sqrt(np.diag(C))
        print(f"\nScenario {sc}: eig(F) min={ev.min():.3e} max={ev.max():.3e} rank={np.linalg.matrix_rank(F, tol=1e-6*ev.max())}/6")
        if sc == "S":
            w, V = np.linalg.eigh(F)
            print("  NOT identifiable. Null direction (lnE, lnP, lnh, lnR, nu, lnalpha) =", np.round(V[:, 0], 3))
        else:
            print("  predicted rel. s.d. (E,P,h) =", [f"{s*100:.1f}%" for s in sd[:3]])
        out[sc] = sd

    # 2. Closed-form propagation check, scenario A (h from imaging)
    s_lnE = np.sqrt(SIG["k"]**2 + (2 * SIG["h"])**2 + SIG["R"]**2 + (TRUE["nu"] / (1 - TRUE["nu"]**2) * SIG["nu"])**2)
    print(f"\nFirst-order closed form, ln E s.d. (A, k2/h/R/nu only) = {s_lnE*100:.1f}%")

    # 3. Repeated synthetic experiments: do actual estimator errors match predicted covariance?
    for sc in ("A", "B"):
        est = []
        for _ in range(400):
            ob = noisy_obs(rng)
            est.append(fit(ob, sc))
        est = np.array(est)
        bias = est.mean(0) - x_true()
        emp = est.std(0)
        pred = np.sqrt(np.diag(posterior_cov(sc)[1]))
        print(f"\nScenario {sc}, 400 repeats: empirical vs predicted s.d. (lnE, lnP, lnh)")
        for j in range(3):
            print(f"  {NAMES[j]:5s} empirical {emp[j]*100:5.1f}%  predicted {pred[j]*100:5.1f}%  bias {bias[j]*100:+.1f}%")

    # 4. Posterior sampling on one dataset
    ob = noisy_obs(np.random.default_rng(7))
    for sc in ("S", "A", "B"):
        ch, ar = adaptive_metropolis(ob, sc, n=60000)
        E = np.exp(ch[:, 0]) / 1e6; P = np.exp(ch[:, 1]) / 1e6; h = np.exp(ch[:, 2]) * 1e6
        q = lambda v: np.percentile(v, [2.5, 50, 97.5])
        print(f"\nMCMC scenario {sc} (accept {ar:.2f}):  E {q(E).round(1)} MPa   P {q(P).round(3)} MPa   h {q(h).round(3)} um")
