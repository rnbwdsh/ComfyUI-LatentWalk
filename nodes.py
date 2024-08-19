import torch

from helper import unpack_conditioning, NoiseWrapper, random_walk
from .blend_modes import BLEND_MODES
from .travel_modes import reflect_values, TRAVEL_MODES

REQUIRED_TYPES = {
    "start": ("LATENT",),
    "end": ("LATENT",),
    "steps": ("INT", {"default": 9, "min": 3, "max": 10000, "step": 1}),
    "factor": ("FLOAT", {"default": 0.5}),
    "blend": (list(BLEND_MODES),),
    "travel": (list(TRAVEL_MODES),),
    "reflect": ("BOOLEAN", {"default": False}),
}


class LatentWalkBase:
    CATEGORY = "latentWalk"

    def latent_walk(self, start, end, steps, factor, blend, travel, reflect):
        if hasattr(start, "shape") and hasattr(end, "shape"):
            if start.shape != end.shape:
                raise ValueError(f"Start and end latents must have the same shape. {start.shape} != {end.shape}")

        try:
            blend = BLEND_MODES[blend]
        except KeyError:
            raise ValueError(f"Unsupported blending mode {blend}. Please choose from {list(BLEND_MODES.keys())}")

        # Get cutpoints based on travel mode
        try:
            cut_points = TRAVEL_MODES[travel](steps, factor)
            if reflect:
                cut_points = reflect_values(cut_points)
        except KeyError:
            raise ValueError(f"Unsupported travel mode {travel}. Please choose from {list(TRAVEL_MODES.keys())}")

        # Blend latents using travel cutpoints and blend mode
        return [blend(end, start, t) for t in cut_points]


class LatentWalkVae(LatentWalkBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {**REQUIRED_TYPES}, "optional": {"vae": ("VAE",)}}

    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("latents", "images",)
    FUNCTION = "latent_walk_vae_decoding"

    def latent_walk_vae_decoding(self, start, end, steps, factor, blend, travel, reflect, vae):
        start, end = start["samples"], end["samples"]
        latents = super().latent_walk(start, end, steps, factor, blend, travel, reflect)
        samples = torch.cat(latents, 0)
        decoded = vae.decode(samples) if vae is not None else None
        return {'samples': samples}, decoded


class LatentWalkNoise(LatentWalkBase):
    @classmethod
    def INPUT_TYPES(cls):
        r = {**REQUIRED_TYPES, "start": ("NOISE",), "end": ("NOISE",)}
        # hardcode add as blend-mode, as other blend-modes don't work easily with noise
        r.pop("blend")
        return {"required": r}

    RETURN_TYPES = ("ACCUMULATION", "NOISE")
    RETURN_NAMES = ("ACCUMULATION", "NOISE_BATCH")
    FUNCTION = "latent_walk_noise"

    def latent_walk_noise(self, start, end, steps, factor, travel, reflect):
        start, end = NoiseWrapper(start), NoiseWrapper(end)
        out_noise = super().latent_walk(start, end, steps, factor, "add", travel, reflect)
        return {'accum': out_noise}, out_noise


class LatentWalkConditional(LatentWalkBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {**REQUIRED_TYPES, "start": ("CONDITIONING",), "end": ("CONDITIONING",)}}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("CONDITIONINGS",)
    FUNCTION = "latent_walk_conditional"

    def latent_walk_conditional(self, start, end, steps, factor, blend, travel, reflect):
        sc, sp = unpack_conditioning(start)
        ec, ep = unpack_conditioning(end)
        assert sc.shape == ec.shape, f"Conditioning shapes don't match: {sc.shape} != {ec.shape}"
        assert sp.shape == ep.shape, f"Pooled shapes don't match: {sp.shape} != {ep.shape}"
        rc = super().latent_walk(sc, ec, steps, factor, blend, travel, reflect)
        rp = super().latent_walk(sp, ep, steps, factor, blend, travel, reflect)
        return ([[torch.cat(rc, 0), {"pooled_output": torch.cat(rp, 0)}]],)


class LatentWalkConditionalRandom:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "start": ("CONDITIONING",),
            "dist_mult": ("FLOAT", {"default": 0.5, "min": 0.001, "max": 1000.0, "step": 0.01}),
            "momentum": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "steps": ("INT", {"default": 9, "max": 10000, "step": 1}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("CONDITIONINGS",)
    FUNCTION = "latent_walk_conditional_random"

    def latent_walk_conditional_random(self, start, dist_mult, momentum, steps):
        sc, sp = unpack_conditioning(start)
        rc = torch.cat(list(random_walk(sc, dist_mult, momentum, steps)), 0)
        rp = torch.cat(list(random_walk(sp, dist_mult, momentum, steps)), 0)
        return ([[rc, {"pooled_output": rp}]],)


classes = [LatentWalkVae, LatentWalkNoise, LatentWalkConditional, LatentWalkConditionalRandom]
NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in classes}
NODE_DISPLAY_NAME_MAPPINGS = {cls.__name__: cls.__name__ for cls in classes}
