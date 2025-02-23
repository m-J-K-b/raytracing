import numpy as np

from src.util import (
    Vec3,
    cosine_hemisphere_pdf,
    fresnel_conductor,
    fresnel_dielectric,
    sample_cosine_hemisphere,
)


class BxDF:
    """Class for the physically based reflection and reflection of an interaction. Holds only the mathematical equations required to simulate interaction, but no properties of the object it self, such as color"""

    def __init__(self) -> None:
        pass

    def reflection_coefficient(
        self, incoming_ray_direction: Vec3, outgoing_ray_direction: Vec3
    ) -> Vec3:
        raise NotImplementedError

    def sample_direction(self, incoming_ray_direction: Vec3) -> Vec3:
        raise NotImplementedError

    def pdf(self, incoming_ray_direction: Vec3, outgoing_ray_direction: Vec3) -> float:
        raise NotImplementedError


class DiffuseBRDF(BxDF):
    def __init__(self):
        return

    def reflection_coefficient(
        self, incoming_ray_direction: Vec3, outgoing_ray_direction: Vec3
    ) -> Vec3:
        """Calculate the reflection coefficient for each part of the light spectrum

        Args:
            incoming_ray_direction (Vec3): incomgin ray direction in normal aligned space, same hemisphere as normal
            outgoing_ray_direction (Vec3): outgoing ray direction in normal aligned space, same hemisphere as normal

        Returns:
            Vec3: reflection coefficient for each part of the spectrum
        """
        if incoming_ray_direction.y * outgoing_ray_direction.y < 0:
            return Vec3(0)
        return Vec3(1 / np.pi)

    def sample_direction(self, incoming_ray_direction: Vec3) -> Vec3:
        """Sample direction based on the incoming ray direction

        Args:
            incoming_ray_direction (Vec3): incomgin ray direction in normal aligned space, same hemisphere as normal
        Returns:
            Vec3: outgoing ray direction in normal aligned space, same hemisphere as normal
        """
        sample = sample_cosine_hemisphere()
        return sample

    def pdf(self, incoming_ray_direction: Vec3, outgoing_ray_direction: Vec3) -> float:
        """Calculate probability density function of the brdf with respect to the incoming and outgoing ray direction

        Args:
            incoming_ray_direction (Vec3): incomgin ray direction in normal aligned space, same hemisphere as normal
            outgoing_ray_direction (Vec3): outgoing ray direction in normal aligned space, same hemisphere as normal

        Returns:
            float: Probability of the incoming and outoing direction occuring as a pair.
        """
        return cosine_hemisphere_pdf(incoming_ray_direction)


class ConductorBRDF(BxDF):
    def __init__(
        self,
        eta: float,
        k: float,
        distribution: "TrowbridgeReitzDistribution | None" = None,
    ) -> None:
        """init conductor brdf

        Args:
            eta (Vec3): Real part of the objects ior
            k (Vec3): Imaginary part of the objects ior
        """
        super().__init__()
        self.eta = eta
        self.k = k
        self.distribution = distribution

    def reflection_coefficient(
        self, incoming_ray_direction: Vec3, outgoing_ray_direction: Vec3
    ) -> Vec3:
        return Vec3(
            fresnel_conductor(incoming_ray_direction, complex(self.eta, self.k))
            / incoming_ray_direction.y
        )

    def sample_direction(self, incoming_ray_direction: Vec3) -> Vec3:
        return Vec3(
            -incoming_ray_direction.x,
            incoming_ray_direction.y,
            -incoming_ray_direction.z,
        )

    def pdf(self, incoming_ray_direction: Vec3, outgoing_ray_direction: Vec3) -> float:
        return (
            incoming_ray_direction.x
            + outgoing_ray_direction.x
            + incoming_ray_direction.z
            + outgoing_ray_direction.z
            == 0
        )
