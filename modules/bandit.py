#!/usr/bin/env python3
"""
bandit.py – Contextual Model-Aware Thompson Bandit
───────────────────────────────────────────────────
2025-06-03  • CVaR-TS  • Online scalar calibration  • ε-greedy

Key updates (vs. 2025-06-02):
1. **Online 1-D calibrator** – rescales every GN-IMC latency prediction
   so that the *predicted* baseline tracks the *measured* baseline.
   → Works on any new machine / Postgres instance without retraining.
2. Default arm is *always* safe, so the “safe=N/…“ counter never shows 0.
3. Bug fix: `_risk_adjusted_draw()` used the wrong variable name.
4. Code is still pickle-compatible with previous versions.

Public API (`choose()` / `update()`) is unchanged.
"""

from __future__ import annotations

import math, pickle, random, time
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.stats import t          # only for CVaR quantile

__all__ = ["ThompsonBandit"]

# ---------------------------------------------------------------------
# Small helper: smooth scalar y_true / y_pred ratio   (EWMA)
# ---------------------------------------------------------------------
class _ScalarCalibrator:
    def __init__(self, alpha: float = 0.02):
        self.gamma = 1.0           # multiplicative scale
        self.alpha = alpha         # EWMA step size
    def update(self, y_pred: float, y_true: float):
        if y_pred > 0 and y_true > 0:
            ratio       = y_true / y_pred
            self.gamma  = (1 - self.alpha) * self.gamma + self.alpha * ratio
    def __call__(self, y_pred: float | np.ndarray) -> float | np.ndarray:
        return y_pred * self.gamma

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
ArmStats = Tuple[int, float, float, float]   # n, μ̂, α, β   (NIΓ posterior)

# ---------------------------------------------------------------------
# Thompson-sampling bandit
# ---------------------------------------------------------------------
class ThompsonBandit:
    # ----- safety / catastrophe --------------------------------------
    N_BAD           = 200       # virtual bad pulls on catastrophe
    R_CAT_FACTOR    = 0.30      # reward ≤ 0.30 ⇒ catastrophe
    CLAMP_MIN_FRACT = 0.10      # latency ≥ 0.1 × baseline
    CLAMP_MAX_FRACT = 4.00      # latency ≤ 4   × baseline

    # ----- exploration knobs -----------------------------------------
    EPSILON         = 0.40      # ε-greedy probability
    CVAR_Q          = 0.50      # risk-quantile for CVaR-TS
    GATE_K0         = 2.0       # initial k in µ + kσ veto
    GATE_K_MIN      = 0.5       # floor for k
    GATE_HALFLIFE   = 50_000     # queries for k to halve
    TARGET_SPEEDUP  = 1.2       # the 1.2× speedup goal
    MONITOR_WINDOW  = 15       # how many recent speedups to track
    MIN_EPSILON     = 0.20      # floor for ε
    MAX_EPSILON     = 0.80      # ceiling for ε

    # ----- Bayesian prior (Normal-Inverse-Gamma) ---------------------
    K0, A0, B0      = 1.0, 3.0, 10.0

    def __init__(
        self,
        persist: str | Path = "~/models/bandit_ts.pkl",
        save_every: int = 25,
        bootstrap_k: int = 5,
        knn_penalty: float = 1.0,
    ):
        self.persist       = Path(persist).expanduser()
        self.save_every    = save_every
        self.bootstrap_k   = bootstrap_k
        self._arm_penalty  = {"knn": knn_penalty}
        self._recent_speedups = deque(maxlen=self.MONITOR_WINDOW)

        # Bandit state  (all pickle-able)
        self.stats: Dict[str, Dict[str, ArmStats]] = defaultdict(dict)
        self.contexts: Dict[str, np.ndarray]       = {}   # qkey → flat embedding

        # Book-keeping
        self._calls   = 0
        self._gate_k  = self.GATE_K0
        self._cal     = _ScalarCalibrator()

        if self.persist.exists():
            self._load()

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    @staticmethod
    def _post_var(n: int, α: float, β: float) -> float:
        """Posterior variance of Student-t predictive."""
        return (β / (α - 1)) * (1 + 1 / max(n, 1)) if α > 1.1 else 1e6

    def _risk_adjusted_draw(self, n: int, μ: float, α: float, β: float) -> float:
        """CVaR-Thompson: draw the Q-quantile (risk-averse)."""
        if n == 0:                         # no data – use posterior mean
            return μ
        df   = 2 * α                       # degrees of freedom
        z_q  = t.ppf(self.CVAR_Q, df=df)   # negative number
        return μ + z_q * math.sqrt(self._post_var(n, α, β))

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def choose(
        self,
        qkey: str,
        arms: Dict[str, Tuple[int, float]],          # {arm: (hint_idx, pred_ms)}
        base_pred_ms: float,
        mu_sigma: Dict[str, Tuple[float, float]],    # {arm: (μ_pred, σ_pred)}
        actual_base_ms: Optional[float] = None,      # measured baseline (if any)
        z: Optional[np.ndarray] = None,              # embedding vector
    ) -> str:
        """Return the arm label to execute for this query."""
        if self._recent_speedups:
            avg_s = sum(self._recent_speedups) / len(self._recent_speedups)
            deficit = max(self.TARGET_SPEEDUP - avg_s, 0)
            frac = min(deficit / self.TARGET_SPEEDUP, 1.0)
            self.EPSILON = self.MIN_EPSILON + frac * (self.MAX_EPSILON - self.MIN_EPSILON)
        # else leave EPSILON at its default startup value

        # ---- update call count & µ + kσ gate -------------------------
        self._calls += 1
        decay  = 0.5 ** (self._calls / self.GATE_HALFLIFE)
        self._gate_k = max(self.GATE_K_MIN, self.GATE_K0 * decay)

        # ---- 1-D calibration (only if we know the real baseline) -----
        if actual_base_ms is not None:
            self._cal.update(base_pred_ms, actual_base_ms)
        base_pred_ms = self._cal(base_pred_ms)

        # scale entire μ/σ dict
        mu_sigma = {
            a: (self._cal(mu), self._cal(sig))
            for a, (mu, sig) in mu_sigma.items()
        }

        # ---- flatten context for k-NN warm-start ---------------------
        if z is not None:
            z = z.flatten()

        # ---- baseline & clamping limits ------------------------------
        baseline_ms = actual_base_ms if actual_base_ms is not None else base_pred_ms
        min_lat = baseline_ms * self.CLAMP_MIN_FRACT
        max_lat = baseline_ms * self.CLAMP_MAX_FRACT

        # ---- pessimistic latency (µ + kσ) ----------------------------
        pessimistic = {}
        for arm, (idx, raw_pred) in arms.items():
            μ_pred, σ_pred = mu_sigma.get(arm, (raw_pred, 0.0))
            worst = μ_pred + self._gate_k * σ_pred
            pessimistic[arm] = (idx, min(max(worst, min_lat), max_lat))

        # ---- safety gate  (default is *always* safe) -----------------
        safe_arms = {
            a: v for a, v in pessimistic.items()
            if (a == "default") or (v[1] < baseline_ms * 2.0)   # allow ≤40 % slower
        }
        print(f"[bandit] {qkey}: gate_k={self._gate_k:.2f}  "
              f"baseline={baseline_ms:.0f}  safe={len(safe_arms)}/{len(arms)}")

        # ---- make sure we always have at least 'default' -------------
        if not safe_arms:
            return "default"

        # ---- pseudo-obs warm-start -----------------------------------
        if qkey not in self.stats:
            # only give default arm a “no-data” prior
            self.stats[qkey] = {
                "default": (0, 1.0, self.A0, self.B0)
            }

        # ---- contextual bootstrap ------------------------------------
        if z is not None:
            self.contexts[qkey] = z
            if len(self.contexts) >= self.bootstrap_k:
                keys, mats = zip(*self.contexts.items())
                D = np.linalg.norm(np.stack(mats) - z, axis=1)
                nearest = np.argsort(D)[: self.bootstrap_k]
                for arm in safe_arms:
                    priors = [
                        self.stats[k].get(arm, (0, 0, 0, 0))[1]
                        for k in (keys[i] for i in nearest) if arm in self.stats[k]
                    ]
                    if priors and arm not in self.stats[qkey]:
                        μ0 = float(sum(priors) / len(priors))
                        self.stats[qkey][arm] = (1, μ0, self.A0, self.B0)

        # ---- ε-greedy exploration ------------------------------------
        if random.random() < self.EPSILON:
            return random.choice(list(safe_arms.keys()))

        # ---- CVaR-Thompson sampling ----------------------------------
        best_arm, best_val = None, -math.inf
        for arm, (_, lat_ms) in safe_arms.items():
            reward_est = baseline_ms / max(lat_ms, 1e-3)
            n, μ, α, β = self.stats[qkey].get(
                arm, (0, reward_est, self.A0, self.B0)
            )
            draw = self._risk_adjusted_draw(n, μ, α, β)
            draw *= self._arm_penalty.get(arm, 1.0)
            if draw > best_val:
                best_arm, best_val = arm, draw

        return best_arm or "default"

    # -----------------------------------------------------------------
    # Bayesian update
    # -----------------------------------------------------------------
    def _bayes_update(self, qkey: str, arm: str, reward: float) -> int:
        n, μ, α, β = self.stats[qkey][arm]
        n1   = n + 1
        k_n  = self.K0 + n
        μ1   = (self.K0 * μ + n * reward) / k_n if n else reward
        α1   = α + 0.5
        β1   = β + 0.5 * (reward - μ) ** 2 * n / k_n if n else β
        self.stats[qkey][arm] = (n1, μ1, α1, β1)
        return n1

    def update(
        self,
        qkey: str,
        arm: str,
        latency_ms: float,
        base_ms: float,
        z: Optional[np.ndarray] = None,
    ):
        """Update posteriors after observing actual latency."""
        if z is not None:
            self.contexts[qkey] = z.flatten()

        speedup = base_ms / max(latency_ms, 1e-3)
        self._recent_speedups.append(speedup)

        raw_reward = speedup
        reward     = raw_reward * self._arm_penalty.get(arm, 1.0)

        # ensure entry exists
        if arm not in self.stats[qkey]:
            self.stats[qkey][arm] = (0, reward, self.A0, self.B0)

        if reward <= self.R_CAT_FACTOR:
            # catastrophe → many “bad” virtual pulls
            for _ in range(self.N_BAD):
                self._bayes_update(qkey, arm, raw_reward)
        else:
            n1 = self._bayes_update(qkey, arm, reward)
            if n1 % self.save_every == 0:
                self._dump()

    # -----------------------xx------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    def _dump(self):
        with open(self.persist, "wb") as f:
            pickle.dump(dict(self.stats), f)

    def _load(self):
        self.stats = defaultdict(dict, pickle.load(open(self.persist, "rb")))
