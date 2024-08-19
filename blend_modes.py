import torch


def slerp(B, A, factor):
    # from  https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475
    dims = A.shape
    A = A.reshape(dims[0], -1)  # flatten to batches
    B = B.reshape(dims[0], -1)
    low_norm = A / torch.norm(A, dim=1, keepdim=True)
    high_norm = B / torch.norm(B, dim=1, keepdim=True)
    low_norm[low_norm != low_norm] = 0.0  # in case we divide by zero
    high_norm[high_norm != high_norm] = 0.0
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - factor) * omega) / so).unsqueeze(1) * A + (
            torch.sin(factor * omega) / so).unsqueeze(1) * B
    return res.reshape(dims)


def overlay_blend(A, B, factor):
    low = 2 * A * B
    high = 1 - 2 * (1 - A) * (1 - B)
    blended_latent = (A * factor) * low + (B * factor) * high
    return blended_latent


def soft_light_blend(A, B, factor):
    low = 2 * A * B + A ** 2 - 2 * A * B * A
    high = 2 * A * (1 - B) + torch.sqrt(A) * (2 * B - 1)
    blended_latent = (A * factor) * low + (B * factor) * high
    return blended_latent


def random_noise(A, B, factor):
    noise1 = torch.randn_like(A)
    noise2 = torch.randn_like(B)
    noise1 = (noise1 - noise1.min()) / (noise1.max() - noise1.min())
    noise2 = (noise2 - noise2.min()) / (noise2.max() - noise2.min())
    blended_noise = (A * factor) * noise1 + (B * factor) * noise2
    blended_noise = torch.clamp(blended_noise, 0, 1)
    return blended_noise


BLEND_MODES = {
    'add': lambda A, B, factor: (A * factor) + (B * (1 - factor)),
    'multiply': lambda A, B, factor: (A * factor) * (B * (1 - factor)),
    'divide': lambda A, B, factor: (A * factor) / (B * (1 - factor)),
    'subtract': lambda A, B, factor: (A * factor) - (B * (1 - factor)),
    'screen': lambda A, B, factor: 1 - ((1 - A) * (1 - B) * (1 - factor)),
    'difference': lambda A, B, factor: abs(A - B) * factor,
    'exclusion': lambda A, B, factor: (A + B - 2 * A * B) * factor,
    'hard_light': lambda A, B, factor: torch.where(B < 0.5, 2 * A * B, 1 - 2 * (1 - A) * (1 - B)) * factor,
    'linear_dodge': lambda A, B, factor: torch.clamp(A + B, 0, 1) * factor,
    'lerp': lambda A, B, factor: torch.lerp(A, B, factor),
    'slerp': slerp,
    'overlay': overlay_blend,
    'soft_light': soft_light_blend,
    'random': random_noise,
}
