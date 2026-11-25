"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                CLARITY ART v4 FOR FORGE NEO (EXTENSION)                      ║
║      Universal Paired Sampler & Native-Calibrated Spectral Scheduler         ║
║                                                                              ║
║  Components:                                                                 ║
║    • Sampler:   Clarity Art (sample_clarity_art)                             ║
║    • Scheduler: Clarity Art (clarity_art)                                    ║
║                                                                              ║
║  Core Engineering:                                                           ║
║    • Float32 internal latent accumulation (zero bfloat16 drift on shift=16)  ║
║    • Tapered Ancestral Rectified Flow: vibrant anatomy/skin + crisp finish   ║
║    • Auto 2nd-order log-SNR exponential step on 16+ steps (1st-order Turbo)  ║
║    • Exact native shift reading for Krea 2, Qwen 2.1, MOSS, FLUX.3, Klein    ║
║    • Safe multi-stage & segmented sampling handoff with shape-safe noise     ║
║    • Exact prediction_type detection (SDXL/SD1.5 img2img & Hires Fix safe)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
from tqdm.auto import trange

import k_diffusion.sampling
from modules import (
    sd_samplers,
    sd_samplers_common,
    sd_samplers_kdiffusion,
    sd_schedulers,
    shared,
    script_callbacks,
)

_LOGGED_REGISTRATION = False


# ============================================================================
# 0. EXTENSION SETTINGS (Settings -> Clarity Art)
# ============================================================================

def _opt(name: str, default: Any) -> Any:
    """Safely reads a Clarity Art setting from shared.opts with fallback."""
    try:
        opts = getattr(shared, "opts", None)
        if opts is not None and hasattr(opts, name):
            val = getattr(opts, name)
            if val is not None:
                return val
    except Exception:
        pass
    return default


def _register_settings():
    """Registers optional fine-tuning knobs under Settings -> Clarity Art."""
    try:
        import gradio as gr
    except Exception:
        return

    section = ("clarity_art", "Clarity Art")

    def _add(key: str, info: Any):
        if key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)

    _add(
        "clarity_eta_taper_sigma",
        shared.OptionInfo(
            0.35,
            "[Clarity Art] Ancestral Eta taper sigma",
            gr.Slider,
            {"minimum": 0.0, "maximum": 0.60, "step": 0.01},
            section=section,
        ).info(
            "Below this sigma, ancestral noise smoothly tapers toward 0 for razor-sharp eyes, "
            "fine textures, and MOSS/FLUX.3 typography. Set 0.0 for constant Euler-a-style noise."
        ),
    )
    _add(
        "clarity_second_order",
        shared.OptionInfo(
            True,
            "[Clarity Art] Log-SNR 2nd-order exponential refinement (16+ steps)",
            section=section,
        ).info(
            "Uses cosine-gated log-SNR 2nd-order refinement on multi-step runs (>= 16 steps). "
            "Automatically stays pure 1st-order on few-step Turbo models (< 16 steps)."
        ),
    )
    _add(
        "clarity_detail_curve",
        shared.OptionInfo(
            0.25,
            "[Clarity Art] Multi-step low-sigma detail curve strength",
            gr.Slider,
            {"minimum": 0.0, "maximum": 0.50, "step": 0.05},
            section=section,
        ).info(
            "Gently softens the final sigma landing cliff on non-Turbo multi-step runs (>= 16 steps, "
            "shift < 10) while keeping high-noise composition zones intact. 0.0 = pure Mobius."
        ),
    )
    _add(
        "clarity_segment_handoff",
        shared.OptionInfo(
            0.50,
            "[Clarity Art] Multi-stage segment boundary eta factor",
            gr.Slider,
            {"minimum": 0.0, "maximum": 1.0, "step": 0.05},
            section=section,
        ).info(
            "Multiplier on ancestral eta during the final handoff step of an intermediate sampling "
            "stage (when sigma_next > 0). 0.50 = cleaner transitions; 1.0 = unchanged."
        ),
    )


# ============================================================================
# 1. RUNTIME CONTEXT, ARCHITECTURE & HIGH-RESOLUTION RESOLVER
# ============================================================================

def _inspect_caller_processing() -> Any:
    """
    Inspects up to 24 stack frames to extract the active StableDiffusionProcessing 'p',
    surviving nested wrappers (ADetailer, ControlNet, X/Y Plot, Hires Fix, multi-stage pipelines).
    """
    curr = inspect.currentframe()
    try:
        for _ in range(24):
            curr = curr.f_back
            if curr is None:
                break
            locs = curr.f_locals
            p = locs.get("p")
            if p is not None and hasattr(p, "width") and hasattr(p, "height"):
                return p
            self_obj = locs.get("self")
            if self_obj is not None:
                sp = getattr(self_obj, "p", None)
                if sp is not None and hasattr(sp, "width") and hasattr(sp, "height"):
                    return sp
    finally:
        del curr
    return None


def _get_active_diffusion_engine() -> Any:
    """Returns the active ForgeDiffusionEngine instance if loaded and valid."""
    try:
        model = getattr(shared, "sd_model", None)
        if model is not None and type(model).__name__ != "FakeInitialModel":
            return model
    except Exception:
        pass

    p = _inspect_caller_processing()
    if p is not None and hasattr(p, "sd_model"):
        model = getattr(p, "sd_model", None)
        if model is not None and type(model).__name__ != "FakeInitialModel":
            return model

    return None


def _extract_predictor(obj: Any = None) -> Any:
    """
    Robustly extracts predictor from CFGDenoiser, ForgeScheduleLinker,
    UnetPatcher, multi-stage wrappers, or active ForgeDiffusionEngine.
    """
    if obj is not None:
        if getattr(obj, "prediction_type", None) is not None:
            return obj

        for attr in ("predictor", "model_wrap"):
            sub = getattr(obj, attr, None)
            if sub is not None:
                if hasattr(sub, "predictor") and sub.predictor is not None:
                    return sub.predictor
                if getattr(sub, "prediction_type", None) is not None:
                    return sub

        seen: set[int] = set()
        curr = obj
        while curr is not None and id(curr) not in seen:
            seen.add(id(curr))
            nxt = getattr(curr, "inner_model", None)
            if nxt is None:
                break
            curr = nxt
            pred = getattr(curr, "predictor", None)
            if pred is not None:
                return pred
            if getattr(curr, "prediction_type", None) is not None:
                return curr
            if hasattr(curr, "forge_objects") and hasattr(curr.forge_objects, "unet"):
                try:
                    return curr.forge_objects.unet.model.predictor
                except Exception:
                    pass

    engine = _get_active_diffusion_engine()
    if engine is not None and hasattr(engine, "forge_objects") and hasattr(engine.forge_objects, "unet"):
        try:
            return engine.forge_objects.unet.model.predictor
        except Exception:
            pass

    return None


def _is_const_flow(
    predictor: Any,
    inner_model: Any = None,
    sigma_max: float | None = None,
) -> bool:
    """
    Accurately detects Flow Matching / Rectified Flow vs Classic Diffusion (SDXL/SD1.5).

    Priority:
      1. predictor.prediction_type (100% exact even on low-denoise SDXL img2img / Hires Fix!)
      2. active ForgeDiffusionEngine class flags (is_webui_legacy_model / is_sd1 / is_sdxl)
      3. sigma_max heuristic fallback (only when running standalone without predictor/engine)
    """
    if predictor is None and inner_model is not None:
        predictor = _extract_predictor(inner_model)

    if predictor is not None:
        pred_type = getattr(predictor, "prediction_type", None)
        if pred_type == "const":
            return True
        if pred_type in ("epsilon", "v_prediction"):
            return False

    engine = _get_active_diffusion_engine()
    if engine is not None:
        if hasattr(engine, "is_webui_legacy_model") and engine.is_webui_legacy_model():
            return False
        if getattr(engine, "is_sd1", False) or getattr(engine, "is_sdxl", False):
            return False
        return True

    if sigma_max is not None:
        if float(sigma_max) > 1.5:
            return False
        if float(sigma_max) <= 1.05:
            return True

    return True


def _is_16x_packed_engine(engine: Any) -> bool:
    """Returns True for 16x-spatial-compression VAEs (FLUX.2, FLUX.3, MOSS-Image)."""
    if engine is None:
        return False
    if getattr(engine, "is_flux3", False) or getattr(engine, "is_moss_image", False):
        return True
    name = type(engine).__name__.lower()
    if any(k in name for k in ("flux2", "flux3", "moss", "ernie")):
        return True
    try:
        vae = getattr(getattr(engine, "forge_objects", None), "vae", None)
        if vae is not None and getattr(vae, "is_flux2", False):
            return True
    except Exception:
        pass
    try:
        lf = getattr(getattr(engine, "model_config", None), "latent_format", None)
        if lf is not None and getattr(lf, "latent_channels", 0) >= 64:
            return True
    except Exception:
        pass
    return False


def _resolve_active_dimensions(
    width: int = 1024,
    height: int = 1024,
    engine: Any = None,
) -> tuple[int, int]:
    """
    Resolves active generation width & height, accounting for txt2img,
    Hires Fix (both txt2img and img2img Hires Fix), 16x packed VAEs, and 3-5+ MP.
    """
    p = _inspect_caller_processing()
    if p is not None:
        is_hr = getattr(p, "is_hr_pass", False) or getattr(p, "is_i2i_hr_pass", False)
        if is_hr:
            w = getattr(p, "hr_upscale_to_x", None) or getattr(p, "width", None)
            h = getattr(p, "hr_upscale_to_y", None) or getattr(p, "height", None)
        else:
            w = getattr(p, "width", None)
            h = getattr(p, "height", None)
        if w and h:
            return int(w), int(h)

    try:
        job_data = getattr(shared.state, "current_latent", None)
        if isinstance(job_data, torch.Tensor) and job_data.ndim in (4, 5):
            lh, lw = int(job_data.shape[-2]), int(job_data.shape[-1])
            # Skip packed 1D still-image latents like [B, 1, 1, N]
            if lh >= 8 and lw >= 8:
                if engine is None:
                    engine = _get_active_diffusion_engine()
                is_16x = _is_16x_packed_engine(engine) or (int(job_data.shape[1]) >= 64)
                scale = 16 if is_16x else 8
                return lw * scale, lh * scale
    except Exception:
        pass

    return int(width), int(height)


def _infer_shift_from_sigmas(inner_model: Any, predictor: Any = None) -> float | None:
    """
    Reads back the exact linear Mobius shift s from the model's precomputed sigma table
    using the midpoint identity: sigma(t) = s*t / (1 + (s - 1)*t) -> s = sigma*(1-t)/(t*(1-sigma)).
    Works identically for both PredictionFlux (exp(mu)) and PredictionDiscreteFlow (raw shift).
    """
    for holder in (inner_model, predictor):
        if holder is None:
            continue
        sigmas = getattr(holder, "sigmas", None)
        if sigmas is None or not isinstance(sigmas, torch.Tensor) or len(sigmas) < 4:
            continue
        i = len(sigmas) // 2 - 1
        t = (i + 1) / len(sigmas)
        s = float(sigmas[i].item())
        if 0.0 < s < 1.0:
            alpha = s * (1.0 - t) / (t * (1.0 - s))
            if math.isfinite(alpha) and alpha >= 0.5:
                return float(alpha)
    return None


def _get_ckpt_name(engine: Any) -> str:
    """Returns lowercase checkpoint name from the active engine (only falling back to opts if absent)."""
    try:
        ckpt_info = getattr(engine, "sd_checkpoint_info", None)
        if ckpt_info is None:
            sd_model = getattr(shared, "sd_model", None)
            if sd_model is not None and type(sd_model).__name__ != "FakeInitialModel":
                ckpt_info = getattr(sd_model, "sd_checkpoint_info", None)
        if ckpt_info is not None:
            name = (
                str(getattr(ckpt_info, "model_name", "") or "")
                + " "
                + str(getattr(ckpt_info, "filename", "") or "")
            ).strip().lower()
            if name:
                return name
        opts_ckpt = getattr(getattr(shared, "opts", None), "sd_model_checkpoint", "")
        return str(opts_ckpt or "").lower()
    except Exception:
        return ""


def _is_turbo_or_distilled(engine_name: str, ckpt_name: str, num_steps: int) -> bool:
    """Detects few-step distilled models where uniform native time spacing must stay exact."""
    if num_steps <= 12:
        return True
    markers = ("turbo", "schnell", "lcm", "lightning", "hyper", "distill", "4step", "8step")
    combined = engine_name + " " + ckpt_name
    return any(m in combined for m in markers)


# ============================================================================
# 2. NATIVE-CALIBRATED SHIFT RESOLVER (KREA 2, QWEN 2.1, MOSS, FLUX 3, KLEIN)
# ============================================================================

def _compute_empirical_flux2_mu(width: int, height: int, num_steps: int) -> float:
    """Official empirical log-SNR shift mu for FLUX.2 / Klein architectures."""
    seq_len = max(64.0, (width * height) / 256.0)
    if seq_len > 4300:
        return float(0.00016927 * seq_len + 0.45666666)
    m_200 = 0.00016927 * seq_len + 0.45666666
    m_10 = 8.73809524e-05 * seq_len + 1.89833333
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def _detect_optimal_shift(
    predictor: Any,
    width: int = 1024,
    height: int = 1024,
    num_steps: int = 20,
    inner_model: Any = None,
) -> float:
    """
    Resolves the exact native Mobius shift factor s >= 1.0 for every Flow Matching model:
      • FLUX.3:        Reads native predictor shift (16.0 default, or checkpoint override)
      • MOSS-Image:    Reads native predictor shift (5.0 default, respects Forge Shift slider)
      • Qwen 2.1:      Exact native dynamic resolution formula exp(0.5 + 0.4*(seq_len-256)/7936)
      • Krea 2:        Exact native exp(mu) = exp(1.15) = 3.158193 (Turbo & Raw aligned)
      • FLUX.2 Klein:  Full empirical Flux2 mu formula without artificial 4.5 clamping
      • All others:    Reads exact predictor shift (PredictionFlux -> exp(mu); DiscreteFlow -> shift)
    """
    engine = _get_active_diffusion_engine()
    width, height = _resolve_active_dimensions(width, height, engine=engine)
    engine_name = type(engine).__name__.lower() if engine is not None else ""
    ckpt_name = _get_ckpt_name(engine)
    pred_cls = type(predictor).__name__ if predictor is not None else ""

    is_krea = False
    is_moss = False
    is_qwen21 = False
    is_klein = False
    try:
        from backend.args import dynamic_args
        is_krea = bool(getattr(dynamic_args, "krea2", False))
        is_moss = bool(getattr(dynamic_args, "moss", False))
        is_qwen21 = bool(getattr(dynamic_args, "qwen21", False))
        is_klein = bool(getattr(dynamic_args, "klein", False))
    except Exception:
        pass

    seq_len = max(256.0, (width * height) / 256.0)

    def _read_discrete_shift(default_shift: float) -> float:
        if predictor is not None and hasattr(predictor, "shift") and predictor.shift is not None:
            val = float(predictor.shift)
            if pred_cls == "PredictionFlux":
                return float(max(1.0, math.exp(val)))
            if val >= 1.0:
                return float(val)
        inferred = _infer_shift_from_sigmas(inner_model, predictor)
        return float(inferred) if inferred is not None else default_shift

    def _read_krea_shift() -> float:
        mu = 1.15
        if predictor is not None and pred_cls == "PredictionFlux":
            try:
                mu = float(predictor.shift)
            except Exception:
                pass
        return float(math.exp(mu))

    def _read_qwen21_shift() -> float:
        mu = 0.5 + (0.9 - 0.5) * (seq_len - 256.0) / (8192.0 - 256.0)
        mu = max(0.35, min(1.85, mu))
        return float(math.exp(mu))

    # --- Tier A: Authoritative engine class / loader flags (never fooled by checkpoint filenames) ---
    if getattr(engine, "is_flux3", False) or "flux3" in engine_name:
        return _read_discrete_shift(16.0)

    if getattr(engine, "is_moss_image", False) or "moss" in engine_name or is_moss:
        return _read_discrete_shift(5.0)

    if engine_name == "qwenimage21" or is_qwen21:
        return _read_qwen21_shift()

    if engine_name == "krea2" or is_krea:
        return _read_krea_shift()

    if engine_name == "flux2" or "klein" in engine_name or is_klein:
        mu = _compute_empirical_flux2_mu(width, height, num_steps)
        return float(min(32.0, math.exp(max(0.1, mu))))

    # If the active engine is another known Forge engine (Flux, QwenImage, Wan, ZImage, Anima,
    # ErnieImage, Lumina2, PiD...), read its true native predictor shift directly without
    # guessing from checkpoint filenames.
    if predictor is not None and hasattr(predictor, "shift") and predictor.shift is not None:
        val = float(predictor.shift)
        if pred_cls == "PredictionFlux":
            return float(max(1.0, math.exp(val)))
        if pred_cls == "PredictionDiscreteFlow":
            return float(max(1.0, val))
        if 0.05 < val <= 2.0:
            return float(math.exp(val))
        return float(max(1.0, val))

    # --- Tier B: Checkpoint filename fallback (only when engine & predictor are unavailable) ---
    if any(tag in ckpt_name for tag in ("flux3", "flux-3", "flux_3")):
        return _read_discrete_shift(16.0)

    if "moss" in ckpt_name:
        return _read_discrete_shift(5.0)

    if any(tag in ckpt_name for tag in ("qwen2.1", "qwen21", "qwen-2.1", "qwen_2.1", "qwen_image21", "2512")):
        return _read_qwen21_shift()

    if "krea" in ckpt_name:
        return _read_krea_shift()

    if "flux2" in ckpt_name or "klein" in ckpt_name:
        mu = _compute_empirical_flux2_mu(width, height, num_steps)
        return float(min(32.0, math.exp(max(0.1, mu))))

    # --- Tier C: Infer exact shift from inner_model.sigmas if available ---
    inferred = _infer_shift_from_sigmas(inner_model, predictor)
    if inferred is not None:
        return float(max(1.0, inferred))

    # --- Tier D: Safe Flow Matching default ---
    return 3.158193


# ============================================================================
# 3. CLARITY ART SCHEDULER
# ============================================================================

def clarity_art_scheduler(
    n: int,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    inner_model: Any = None,
    device: str | torch.device = "cpu",
    width: int = 1024,
    height: int = 1024,
    **kwargs,
) -> torch.Tensor:
    """
    Clarity Art Scheduler:
    - Exact (steps + 1) strictly monotonic float64 -> float32 grid.
    - Reads each model's true native shift (Krea 2, Qwen 2.1, MOSS, FLUX.3, Klein, Wan, Z-Image).
    - Keeps pure native Mobius grid for Turbo / few-step (<= 12) and high-shift (FLUX.3) models.
    - Applies a gentle C-infinity cubic detail tail on multi-step runs (>= 16 steps, shift < 10)
      that reduces the final landing cliff by ~20-30% while leaving high-noise composition zones intact.
    - Honors explicit sigma_min / sigma_max overrides and handles steps == 1 cleanly on SDXL/SD1.5.
    """
    steps = max(1, int(n))
    engine = _get_active_diffusion_engine()
    width, height = _resolve_active_dimensions(width, height, engine=engine)
    predictor = _extract_predictor(inner_model)
    is_flow = _is_const_flow(predictor, inner_model=inner_model, sigma_max=sigma_max)

    if is_flow:
        # Custom piecewise-distilled grid models (e.g. MiniMax-H3-Image PDD heads)
        if predictor is not None and hasattr(predictor, "_grid") and callable(getattr(predictor, "sigma", None)):
            mult = float(getattr(predictor, "multiplier", 1000.0))
            ts = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32, device=device)[:-1] * mult
            sigs = predictor.sigma(ts).to(device=device, dtype=torch.float32)
            return torch.cat([sigs, torch.zeros(1, dtype=torch.float32, device=device)])

        u = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64, device=device)
        shift = _detect_optimal_shift(
            predictor, width=width, height=height, num_steps=steps, inner_model=inner_model
        )

        engine_name = type(engine).__name__.lower() if engine is not None else ""
        ckpt_name = _get_ckpt_name(engine)
        is_turbo = _is_turbo_or_distilled(engine_name, ckpt_name, steps)

        # Gentle cubic detail tail: u_adj = u + gamma * u^3 * (1 - u)
        # - Strictly monotonic for all gamma in [0, 0.99] (min derivative = 1 - gamma at u=1)
        # - Virtually zero displacement in high-noise composition zone (u <= 0.3 -> < 0.5% change)
        # - Softens the final sigma cliff on 16-50 step non-Turbo models for crisper micro-details
        gamma = float(_opt("clarity_detail_curve", 0.25))
        gamma = max(0.0, min(0.50, gamma))
        if steps >= 16 and not is_turbo and shift < 10.0 and gamma > 0.0:
            u = u + gamma * (u ** 3) * (1.0 - u)
            u[0] = 0.0
            u[-1] = 1.0

        raw_sigma = 1.0 - u
        sigmas = (shift * raw_sigma) / (1.0 + (shift - 1.0) * raw_sigma)
        sigmas[0] = 1.0
        sigmas[-1] = 0.0

        # Detect model's own boundary sigmas so Forge's default (m_sigma_min, m_sigma_max)
        # in KDiffusionSampler.get_sigmas is NOT mistaken for a manual user override
        # (critical for high-shift DiscreteFlow models like FLUX.3 where m_sigma_min = 0.0158!).
        m_smin, m_smax = 0.0, 1.0
        for holder in (inner_model, predictor):
            tbl = getattr(holder, "sigmas", None) if holder is not None else None
            if isinstance(tbl, torch.Tensor) and len(tbl) >= 2:
                m_smin = float(tbl[0].item())
                m_smax = float(tbl[-1].item())
                break

        s_hi = 1.0
        if (
            sigma_max is not None
            and abs(float(sigma_max) - m_smax) > 1e-4
            and 0.05 < float(sigma_max) < 0.999
        ):
            s_hi = float(sigma_max)

        s_lo = 0.0
        if (
            sigma_min is not None
            and abs(float(sigma_min) - m_smin) > 1e-4
            and 0.001 < float(sigma_min) < s_hi - 1e-3
        ):
            s_lo = float(sigma_min)

        if s_hi < 0.999 or s_lo > 0.0:
            active = sigmas[:-1]
            if len(active) > 1:
                a_min = float(active[-1].item())
                a_max = float(active[0].item())
                span = max(a_max - a_min, 1e-6)
                normed = (active - a_min) / span
                rescaled = s_lo + normed * (s_hi - s_lo)
                sigmas = torch.cat([rescaled, torch.zeros(1, dtype=torch.float64, device=device)])
            else:
                sigmas[0] = s_hi

        return sigmas.to(torch.float32)

    else:
        # Classic Diffusion (SDXL / SD 1.5 / V-Pred)
        s_max = 14.6146
        s_min = 0.0292

        if predictor is not None:
            if hasattr(predictor, "sigma_max") and predictor.sigma_max is not None:
                s_max = float(predictor.sigma_max)
            if hasattr(predictor, "sigma_min") and predictor.sigma_min is not None:
                s_min = max(1e-4, float(predictor.sigma_min))

        if sigma_max is not None and float(sigma_max) > 0.0:
            s_max = float(sigma_max)
        if sigma_min is not None and float(sigma_min) > 0.0:
            s_min = float(sigma_min)
        if s_min >= s_max:
            s_min = max(1e-4, s_max * 0.01)

        if steps == 1:
            return torch.tensor([s_max, 0.0], dtype=torch.float32, device=device)

        t = torch.linspace(0.0, 1.0, steps, dtype=torch.float64, device=device)
        alpha_b, beta_b = 1.08, 1.02
        tc = t.clamp(1e-8, 1.0 - 1e-8)
        pow_a = tc.pow(alpha_b)
        tw = pow_a / (pow_a + (1.0 - tc).pow(beta_b))
        tw[0], tw[-1] = 0.0, 1.0

        rho = 7.0
        s_max_inv = s_max ** (1.0 / rho)
        s_min_inv = s_min ** (1.0 / rho)
        sigmas = (s_max_inv + tw * (s_min_inv - s_max_inv)) ** rho

        return torch.cat([sigmas.to(torch.float32), torch.zeros(1, dtype=torch.float32, device=device)])


# ============================================================================
# 4. UNIVERSAL SAMPLER (FLOAT32 TAPERED ANCESTRAL RF + 2M LOG-SNR + CLASSIC SDE)
# ============================================================================

def _cosine_directional_gate(
    v_cur: torch.Tensor,
    v_prev: torch.Tensor,
    threshold: float = 0.90,
) -> torch.Tensor:
    """
    Per-sample cosine similarity gate in [0.0, 1.0].
    Zeroes out 2nd-order correction whenever consecutive denoiser predictions diverge.
    Works for 3D, 4D image, and 5D video latents.
    """
    b = v_cur.shape[0]
    vc = v_cur.reshape(b, -1).to(torch.float32)
    vp = v_prev.reshape(b, -1).to(torch.float32)

    dot = (vc * vp).sum(dim=-1)
    norm = vc.norm(dim=-1).clamp_min(1e-6) * vp.norm(dim=-1).clamp_min(1e-6)
    cos_sim = torch.nan_to_num(dot / norm, nan=1.0).clamp(-1.0, 1.0)

    denom = max(1.0 - threshold, 1e-6)
    gate = ((cos_sim - threshold) / denom).clamp(0.0, 1.0)
    view_shape = [b] + [1] * (v_cur.ndim - 1)
    return gate.to(torch.float32).view(*view_shape)


def _rms_clamp_delta(delta: torch.Tensor, max_rms_mult: float = 1.8) -> torch.Tensor:
    """Clamps difference vectors relative to their per-sample RMS magnitude in float32."""
    b = delta.shape[0]
    flat = delta.reshape(b, -1).to(torch.float32)
    rms = torch.sqrt(torch.mean(flat * flat, dim=-1, keepdim=True) + 1e-8)
    view_shape = [b] + [1] * (delta.ndim - 1)
    limit = (rms * max_rms_mult).to(delta.dtype).view(*view_shape)
    return torch.clamp(delta, -limit, limit)


def _smoothstep_taper(sigma_next: float, taper_sigma: float) -> float:
    """
    Smooth cubic Hermite taper in [0.0, 1.0]:
      - 1.0 when sigma_next >= taper_sigma (full ancestral liveliness)
      - smoothly drops to 0.0 as sigma_next -> 0 (crisp, non-jittery detail lock-in)
    """
    if taper_sigma <= 1e-6 or sigma_next >= taper_sigma:
        return 1.0
    if sigma_next <= 0.0:
        return 0.0
    u = sigma_next / taper_sigma
    return u * u * (3.0 - 2.0 * u)


def _flow_noise(x: torch.Tensor, sigma, sigma_next, noise_sampler=None) -> torch.Tensor:
    """
    Fresh Gaussian noise for ancestral / SDE steps, returned in float32 on x's device.

    Priority:
      1. Explicit noise_sampler (e.g. Brownian) if shape matches x;
      2. k_diffusion.sampling.torch.randn_like -> Forge's TorchHijack (per-seed RNG,
         shape-safe across multi-resolution stages);
      3. Plain torch.randn_like fallback.
    """
    if noise_sampler is not None:
        try:
            n = noise_sampler(sigma, sigma_next)
            if isinstance(n, torch.Tensor) and n.shape == x.shape:
                return n.to(device=x.device, dtype=torch.float32)
        except Exception:
            pass
    try:
        n = k_diffusion.sampling.torch.randn_like(x)
        if isinstance(n, torch.Tensor) and n.shape == x.shape:
            return n.to(device=x.device, dtype=torch.float32)
    except Exception:
        pass
    return torch.randn(x.shape, device=x.device, dtype=torch.float32)


def _flow_half_log_snr(sigma: float) -> float:
    """Exact half-log-SNR lambda = log((1 - sigma) / sigma) for Rectified Flow."""
    s = max(1e-5, min(1.0 - 1e-5, sigma))
    return math.log((1.0 - s) / s)


def _resolve_total_steps(num_steps: int) -> int:
    """Gets total scheduled steps (handles both single-pass and segmented multi-stage runs)."""
    p = _inspect_caller_processing()
    if p is not None:
        is_hr = getattr(p, "is_hr_pass", False) or getattr(p, "is_i2i_hr_pass", False)
        if is_hr:
            hr_s = getattr(p, "hr_second_pass_steps", 0) or getattr(p, "steps", num_steps)
            return max(num_steps, int(hr_s))
        st = getattr(p, "steps", None)
        if st:
            return max(num_steps, int(st))
    return num_steps


@torch.no_grad()
def sample_clarity_art(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict | None = None,
    callback: Any = None,
    disable: bool | None = None,
    eta: float | None = 1.0,
    s_noise: float | None = 1.0,
    noise_sampler: Any = None,
    **kwargs,
) -> torch.Tensor:
    """
    Clarity Art v4 Universal Sampler:
    - Exact model type detection via predictor.prediction_type (fixes SDXL img2img / Hires Fix!).
    - Float32 internal latent accumulation (zero bfloat16 rounding loss on high-shift FLUX.3/MOSS).
    - FLOW MATCHING / RECTIFIED FLOW (Krea 2, Qwen 2.1, FLUX.3, MOSS, FLUX.1/2, Wan 2.1):
        * Tapered Ancestral RF: vibrant anatomy, hands & skin in high/mid noise, smoothly
          converging to crisp eyes, eyelashes, and razor-sharp typography near sigma -> 0.
        * Multi-step (>= 16 steps): cosine-gated, RMS-clamped log-SNR 2nd-order exponential
          refinement (automatically 1st-order on few-step Turbo models < 16 steps).
        * Multi-stage safe: shape-safe noise, clean segment handoff, terminal landing ONLY at sigma == 0.
    - CLASSIC DIFFUSION (SDXL, SD 1.5, V-Pred):
        * True DPM++ 2M SDE when eta > 0 (with smooth low-sigma taper) and exact DPM++ 2M when eta == 0.
    """
    extra_args = {} if extra_args is None else extra_args
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    s_in = x.new_ones([x.shape[0]], dtype=torch.float32)
    num_steps = len(sigmas) - 1

    if num_steps <= 0:
        return x

    orig_dtype = x.dtype
    x_f32 = x.to(torch.float32)

    eta_base = 1.0 if eta is None else max(0.0, min(1.0, float(eta)))
    s_noise_val = 1.0 if s_noise is None else float(s_noise)

    taper_sigma = max(0.0, min(0.60, float(_opt("clarity_eta_taper_sigma", 0.35))))
    use_second_order_opt = bool(_opt("clarity_second_order", True))
    segment_handoff_mult = max(0.0, min(1.0, float(_opt("clarity_segment_handoff", 0.50))))

    predictor = _extract_predictor(model)
    is_flow = _is_const_flow(predictor, inner_model=model, sigma_max=float(sigmas[0].item()))

    if is_flow:
        # ====================================================================
        # RECTIFIED FLOW / FLOW MATCHING INTEGRATOR (Float32 Accumulator)
        # ====================================================================
        total_steps = _resolve_total_steps(num_steps)
        allow_2m = use_second_order_opt and (total_steps >= 16)

        old_denoised: torch.Tensor | None = None
        h_last: float | None = None

        for i in trange(num_steps, disable=disable):
            s_cur = float(sigmas[i].item())
            s_next = float(sigmas[i + 1].item())

            # 1. Model evaluation in original dtype -> promote denoised to float32
            x_model_in = x_f32.to(orig_dtype)
            denoised_raw = model(x_model_in, sigmas[i] * s_in, **extra_args)
            denoised = torch.nan_to_num(denoised_raw.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)

            if callback is not None:
                callback({
                    "x": x_model_in,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised.to(orig_dtype),
                })

            # 2. Terminal landing ONLY when sigma_next reaches exact zero
            #    Never collapse at intermediate multi-stage segment boundaries (sigma_next > 0)!
            if s_next <= 1e-6:
                x_f32 = denoised
                break

            # Guard against degenerate duplicate sigmas
            if abs(s_cur - s_next) <= 1e-7:
                continue

            s_cur_c = max(s_cur, 1e-5)

            # 3. Compute effective tapered ancestral eta for this step
            is_segment_handoff = (i == num_steps - 1) and (s_next > 1e-6)
            eta_step = eta_base * _smoothstep_taper(s_next, taper_sigma)
            if is_segment_handoff:
                eta_step *= segment_handoff_mult

            # 4. Ancestral step down target (clamped for monotonic safety)
            if eta_step <= 1e-6:
                eta_step = 0.0
                sigma_down = s_next
            else:
                sigma_down = s_next * (1.0 + (s_next / s_cur_c - 1.0) * eta_step)
                sigma_down = max(0.0, min(s_next, sigma_down))

            # 5. Optional log-SNR 2nd-order exponential refinement (16+ step runs only)
            #    Exact identity in RF: -alpha_down * expm1(-h) == (1 - sigma_down / s_cur)
            h_cur: float | None = None
            if 1e-4 < s_cur_c < 1.0 - 1e-4 and 1e-4 < sigma_down < 1.0 - 1e-4:
                h_cur = _flow_half_log_snr(sigma_down) - _flow_half_log_snr(s_cur_c)

            d_eff = denoised
            if (
                allow_2m
                and i > 0
                and old_denoised is not None
                and old_denoised.shape == denoised.shape
                and h_cur is not None
                and h_last is not None
                and abs(h_cur) > 1e-4
                and abs(h_last) > 1e-4
                and s_cur_c <= 0.94
                and s_next >= 0.02
            ):
                r_h = max(0.35, min(3.0, h_last / h_cur))
                # Dampen 2nd-order term when high stochastic noise was injected on the previous step
                stoch_damp = 1.0 - 0.65 * eta_step
                w_2m = min(0.30, 0.5 / r_h) * stoch_damp
                if w_2m > 1e-3:
                    gate = _cosine_directional_gate(denoised, old_denoised, threshold=0.90)
                    diff = _rms_clamp_delta(denoised - old_denoised, max_rms_mult=1.8)
                    d_eff = denoised + (w_2m * gate) * diff

            # 6. Float32 exact convex / exponential step down to sigma_down
            r_step = sigma_down / s_cur_c
            x_f32 = r_step * x_f32 + (1.0 - r_step) * d_eff

            # 7. Ancestral re-noising back to exact sigma_next marginal
            if eta_step > 0.0:
                alpha_next = 1.0 - s_next
                alpha_down = max(1.0 - sigma_down, 1e-6)
                renoise_sq = s_next ** 2 - (sigma_down ** 2) * (alpha_next ** 2) / (alpha_down ** 2)
                renoise = math.sqrt(max(0.0, renoise_sq))
                x_f32 = (alpha_next / alpha_down) * x_f32
                if renoise > 0.0 and s_noise_val != 0.0:
                    noise = _flow_noise(x_f32, sigmas[i], sigmas[i + 1], noise_sampler)
                    x_f32 = x_f32 + noise * (s_noise_val * renoise)

            x_f32 = torch.nan_to_num(x_f32, nan=0.0, posinf=0.0, neginf=0.0)

            old_denoised = denoised
            h_last = h_cur

        return x_f32.to(orig_dtype)

    else:
        # ====================================================================
        # CLASSIC DIFFUSION INTEGRATOR (SDXL / SD 1.5 - Float32 DPM++ 2M / SDE)
        # ====================================================================
        old_denoised = None
        h_last = None
        # Classic sigma scale is ~0.03..14.6; map taper threshold proportionally (~0.5)
        classic_taper_sigma = taper_sigma * 2.0

        for i in trange(num_steps, disable=disable):
            s_cur = float(sigmas[i].item())
            s_next = float(sigmas[i + 1].item())

            x_model_in = x_f32.to(orig_dtype)
            denoised_raw = model(x_model_in, sigmas[i] * s_in, **extra_args)
            denoised = torch.nan_to_num(denoised_raw.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)

            if callback is not None:
                callback({
                    "x": x_model_in,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised.to(orig_dtype),
                })

            if s_next <= 1e-6:
                x_f32 = denoised
                break

            if abs(s_cur - s_next) <= 1e-7:
                continue

            s_cur_c = max(s_cur, 1e-6)
            lambda_cur = -math.log(s_cur_c)
            lambda_next = -math.log(max(s_next, 1e-6))
            h = lambda_next - lambda_cur

            if abs(h) <= 1e-6:
                continue

            eta_step = eta_base * _smoothstep_taper(s_next, classic_taper_sigma)
            if eta_step <= 1e-6:
                eta_step = 0.0

            if (
                old_denoised is None
                or h_last is None
                or abs(h_last) <= 1e-6
                or old_denoised.shape != denoised.shape
                or i == 0
            ):
                d_eff = denoised
            else:
                r = max(0.25, min(4.0, h_last / h))
                diff = _rms_clamp_delta(denoised - old_denoised, max_rms_mult=2.2)
                gate = _cosine_directional_gate(denoised, old_denoised, threshold=0.85)
                d_eff = denoised + (gate / (2.0 * r)) * diff

            # DPM++ 2M SDE (reduces identically to deterministic DPM++ 2M when eta_step == 0)
            h_eta = h * (eta_step + 1.0)
            x_f32 = (s_next / s_cur_c) * math.exp(-h * eta_step) * x_f32 - math.expm1(-h_eta) * d_eff

            if eta_step > 0.0 and s_noise_val != 0.0:
                noise_std = s_next * math.sqrt(max(0.0, -math.expm1(-2.0 * h * eta_step)))
                if noise_std > 0.0:
                    noise = _flow_noise(x_f32, sigmas[i], sigmas[i + 1], noise_sampler)
                    x_f32 = x_f32 + noise * (s_noise_val * noise_std)

            x_f32 = torch.nan_to_num(x_f32, nan=0.0, posinf=0.0, neginf=0.0)

            old_denoised = denoised
            h_last = h

        return x_f32.to(orig_dtype)


# ============================================================================
# 5. REGISTRATION OF SAMPLER, SCHEDULER & SETTINGS IN FORGE NEO
# ============================================================================

def _register_clarity_art():
    """Registers Clarity Art Scheduler and Sampler cleanly and idempotently in Forge NEO."""
    global _LOGGED_REGISTRATION

    sched_name = "clarity_art"
    sched_label = "Clarity Art"
    sched_aliases = ["Clarity", "clarity", "ClarityArt"]

    # --- A. Register Scheduler (single shared Scheduler instance) ---
    sched_obj = next((s for s in sd_schedulers.all_schedulers if s.name == sched_name), None)
    if sched_obj is None:
        sched_obj = sd_schedulers.Scheduler(
            name=sched_name,
            label=sched_label,
            function=clarity_art_scheduler,
            need_inner_model=True,
            aliases=sched_aliases,
        )
        sd_schedulers.all_schedulers.append(sched_obj)
    else:
        sched_obj.function = clarity_art_scheduler
        sched_obj.need_inner_model = True

    hidden_scheds = set(getattr(getattr(shared, "opts", None), "hide_schedulers", []) or [])
    if sched_label not in hidden_scheds and not any(s.name == sched_name for s in sd_schedulers.schedulers):
        sd_schedulers.schedulers.append(sched_obj)

    sd_schedulers.schedulers_map[sched_name] = sched_obj
    sd_schedulers.schedulers_map[sched_label] = sched_obj
    for alias in sched_aliases:
        sd_schedulers.schedulers_map[alias] = sched_obj

    if hasattr(sd_samplers_kdiffusion, "k_diffusion_scheduler"):
        sd_samplers_kdiffusion.k_diffusion_scheduler[sched_name] = clarity_art_scheduler

    # --- B. Register Sampler (Paired with Clarity Art Scheduler by default) ---
    setattr(k_diffusion.sampling, "sample_clarity_art", sample_clarity_art)
    sd_samplers_kdiffusion.sampler_extra_params["sample_clarity_art"] = ["eta", "s_noise"]

    sampler_label = "Clarity Art"
    sampler_aliases = ["clarity_art", "ClarityArt", "clarity"]
    sampler_options = {
        "scheduler": sched_name,
        "default_scheduler": sched_name,
        "second_order": False,
        "uses_ensd": True,
    }

    existing_sampler = next((s for s in sd_samplers.all_samplers if s.name == sampler_label), None)
    if existing_sampler is None:
        sampler_data = sd_samplers_common.SamplerData(
            sampler_label,
            lambda model: sd_samplers_kdiffusion.KDiffusionSampler("sample_clarity_art", model, sampler_options),
            sampler_aliases,
            sampler_options,
        )
        sd_samplers.all_samplers.append(sampler_data)
    else:
        sampler_data = existing_sampler

    sd_samplers.all_samplers_map[sampler_label] = sampler_data
    for alias in sampler_aliases:
        sd_samplers.all_samplers_map[alias] = sampler_data

    sd_samplers.set_samplers()

    if hasattr(sd_samplers.get_sampler_and_scheduler, "cache_clear"):
        sd_samplers.get_sampler_and_scheduler.cache_clear()

    if not _LOGGED_REGISTRATION:
        _LOGGED_REGISTRATION = True
        print("[Clarity Art] v4 Extension Paired Sampler & Scheduler registered successfully")


try:
    _register_clarity_art()
    script_callbacks.on_before_ui(_register_clarity_art)
    script_callbacks.on_ui_settings(_register_settings)
except Exception as e:
    print(f"[Clarity Art] Registration failed: {e}")
