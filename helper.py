import torch


def unpack_conditioning(cond):
    return cond[0][0], cond[0][1]["pooled_output"]


class NoiseWrapper:
    """ Helper class around a duck-typed Noise_RandomNoise object from nodes_custom_sampler with
    a generate_noise function """
    def __init__(self, noise, factor=None, other=None):
        self.noise = noise
        self.factor = factor
        self.other = other

    @property
    def seed(self):
        return self.noise.seed

    def generate_noise(self, input_latent):
        if self.other is None:
            return self.noise.generate_noise(input_latent) * self.factor
        else:
            expected_norm = self.noise.noise.generate_noise(input_latent).norm()
            unscaled = self.noise.generate_noise(input_latent) + self.other.generate_noise(input_latent)
            # without rescaling, the overall noise becomes too small -> sampling doesn't work as expected
            return unscaled * expected_norm / unscaled.norm()

    def __mul__(self, factor: float):
        assert isinstance(factor, float)
        return NoiseWrapper(self.noise, factor, None)

    def __add__(self, other):
        assert isinstance(other, NoiseWrapper)
        return NoiseWrapper(self, None, other)

    def __repr__(self):
        return f"NoiseWrapper({self.noise}, {self.factor}, {self.other})"


def random_walk(start, dist_mult, momentum, steps):
    curr = start.clone()
    target_norm = start.norm()
    yield curr.clone()
    prev = curr.clone()
    for _ in range(steps - 1):
        rand_add = torch.randn_like(start)
        rand_add = rand_add / rand_add.norm() * target_norm * dist_mult
        curr = curr * (1 - momentum) + (prev + rand_add) * momentum
        curr = curr / curr.norm() * target_norm
        prev = curr.clone()
        yield curr.clone()