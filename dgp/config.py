"""True parameters of the synthetic DGP.

All ground-truth values live here. `ChannelConfig` holds per-channel parameters
(adstock, saturation, random-walk coefficient dynamics, spend generation).
`DGPConfig` holds global parameters (horizon, baseline dynamics, noise, seed).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChannelConfig:
    """Per-channel ground-truth parameters.

    Coefficient dynamics:
        β_{c,t+1} = β_{c,t} + beta_drift + η_t,   η_t ~ N(0, beta_innovation_std^2)

    Transforms applied (in order) to raw weekly spend s_{c,t}:
        adstocked:   a*_{c,t} = s_{c,t} + adstock_decay · a*_{c,t-1}
        saturated:   a_{c,t}  = Hill(a*_{c,t}; alpha=hill_alpha, gamma=hill_gamma)

    Spend generation:
        s_{c,t} ~ LogNormal(mu=log(spend_mean) - 0.5·spend_log_sigma^2,
                            sigma=spend_log_sigma)
        which has E[s] = spend_mean.
    """

    name: str

    # Coefficient random-walk-with-drift.
    beta_0: float
    beta_drift: float
    beta_innovation_std: float

    # Adstock (geometric).
    adstock_decay: float  # λ ∈ [0, 1)

    # Hill saturation.  Input is normalized by hill_scale before the Hill
    # function, so gamma lives on a [0, ~1] scale regardless of spend units.
    hill_alpha: float     # shape; >0. Larger ⇒ sharper S-curve.
    hill_gamma: float     # half-saturation point on the normalized scale.
    hill_scale: float     # divisor applied to adstocked spend prior to Hill.

    # Spend generation (log-normal).
    spend_mean: float
    spend_log_sigma: float


@dataclass(frozen=True)
class DGPConfig:
    """Global DGP parameters.

    The observation equation is:
        y_t = α_t + Σ_c β_{c,t} · a_{c,t} + ε_t,   ε_t ~ N(0, noise_std^2)

    Baseline α_t follows a deterministic trend plus annual seasonality:
        α_t = baseline_mean
              + baseline_trend · t
              + seasonality_amplitude · sin(2π · t / seasonality_period)
    """

    n_weeks: int = 156
    channels: tuple[ChannelConfig, ...] = field(
        default_factory=lambda: default_channels()
    )

    # Baseline dynamics.
    baseline_mean: float = 10.0
    baseline_trend: float = 0.01
    seasonality_amplitude: float = 1.5
    seasonality_period: float = 52.0  # weeks

    # Observation noise.
    noise_std: float = 0.35

    # RNG seed.  None ⇒ non-reproducible.
    seed: int | None = 42

    def channel_by_name(self, name: str) -> ChannelConfig:
        for c in self.channels:
            if c.name == name:
                return c
        raise KeyError(f"No channel named {name!r}")


def default_channels() -> tuple[ChannelConfig, ...]:
    """Three channels encoding the narrative: TV declining, Digital rising, Search stable.

    Spend / Hill parameters are tuned so each channel's adstocked-and-saturated
    feature lands in the *sensitive* region of the Hill curve (mean ≈ 0.5–0.7
    on the normalized scale, CoV ≳ 10%).  That keeps β_{c,t} identifiable
    from y over 156 weeks — earlier defaults used saturated features
    (mean ≈ 0.85, CoV ≈ 3%) and β_tv was in a near-nullspace direction.

    Adstock decays differ by channel (TV 0.50, Digital 0.25, Search 0.10) to
    preserve the editorial flavor "TV has the longest memory, Search almost
    none" while keeping feature dynamics distinct enough that β_c is
    individually recoverable.
    """
    tv = ChannelConfig(
        name="tv",
        beta_0=4.0,
        beta_drift=-0.015,          # ~−2.3 over 156 weeks
        beta_innovation_std=0.020,
        adstock_decay=0.50,
        hill_alpha=1.5,
        hill_gamma=0.5,
        hill_scale=45.0,
        spend_mean=30.0,
        spend_log_sigma=0.55,
    )
    digital = ChannelConfig(
        name="digital",
        beta_0=1.5,
        beta_drift=0.020,           # ~+3.1 over 156 weeks
        beta_innovation_std=0.015,
        adstock_decay=0.25,
        hill_alpha=1.5,
        hill_gamma=0.5,
        hill_scale=45.0,
        spend_mean=30.0,
        spend_log_sigma=0.55,
    )
    search = ChannelConfig(
        name="search",
        beta_0=3.0,
        beta_drift=0.0,
        beta_innovation_std=0.010,
        adstock_decay=0.10,
        hill_alpha=1.5,
        hill_gamma=0.5,
        hill_scale=45.0,
        spend_mean=30.0,
        spend_log_sigma=0.55,
    )
    return (tv, digital, search)
