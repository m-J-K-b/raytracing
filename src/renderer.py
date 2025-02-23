from threading import Lock, Thread
from typing import List

import numpy as np

from src.frame import Frame
from src.ray import Ray
from src.render_result import RenderResult
from src.render_settings import RenderSettings
from src.scene import Scene
from src.util import Vec3, lerp, sample_hemisphere


class Renderer:
    def __init__(self):
        self.scene: Scene = None  # type: ignore

        self.render_result: RenderResult = None  # type: ignore
        self.render_settings: RenderSettings = None  # type: ignore

    # SampledSpectrum LiRandomWalk(RayDifferential ray,
    #         SampledWavelengths &lambda, Sampler sampler,
    #         ScratchBuffer &scratchBuffer, int depth) const {
    #     <<Intersect ray with scene and return if no intersection>>
    #        pstd::optional<ShapeIntersection> si = Intersect(ray);
    #        if (!si) {
    #            <<Return emitted light from infinite light sources>>
    #               SampledSpectrum Le(0.f);
    #               for (Light light : infiniteLights)
    #                   Le += light.Le(ray, lambda);
    #               return Le;

    #        }
    #        SurfaceInteraction &isect = si->intr;

    #     <<Get emitted radiance at surface intersection>>
    #        Vector3f wo = -ray.d;
    #        SampledSpectrum Le = isect.Le(wo, lambda);

    #     <<Terminate random walk if maximum depth has been reached>>
    #        if (depth == maxDepth)
    #            return Le;

    #     <<Compute BSDF at random walk intersection point>>
    #        BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);

    #     <<Randomly sample direction leaving surface for random walk>>
    #        Point2f u = sampler.Get2D();
    #        Vector3f wp = SampleUniformSphere(u);

    #     <<Evaluate BSDF at surface for sampled direction>>
    #        SampledSpectrum fcos = bsdf.f(wo, wp) * AbsDot(wp, isect.shading.n);
    #        if (!fcos)
    #            return Le;

    #     <<Recursively trace ray to estimate incident radiance at surface>>
    #        ray = isect.SpawnRay(wp);
    #        return Le  + fcos * LiRandomWalk(ray, lambda, sampler, scratchBuffer,
    #                                         depth + 1) / (1 / (4 * Pi));

    # }

    def raytrace(self, ray: Ray) -> Vec3:
        L = Vec3(0)
        beta = Vec3(1)
        depth = 0
        while beta.x + beta.y + beta.z:
            intersection = self.scene.intersect(ray)

            # Sample environment light if no intersection occured
            if not intersection:
                L += beta.prod(self.scene.get_environment(ray))
                break
            shading_frame = Frame.from_normal(intersection.normal)
            wi = shading_frame.to_local(
                -ray.direction
            )  # incoming ray direction in local space.

            # Sample emitted light from intersected object
            if intersection.obj.material.emission_strength > 0:
                L += beta.prod(intersection.obj.Le())

            # Break loop if depth has exceeded max depth
            if depth >= self.render_settings.MAX_BOUNCES:
                break

            wo = intersection.obj.bxdf.sample_direction(
                wi
            )  # outgoiing ray direction in local space.
            pdf = intersection.obj.bxdf.pdf(wi, wo)
            beta = beta.prod(
                intersection.obj.bxdf.reflection_coefficient(wo, wi).prod(
                    intersection.obj.material.color
                )
                / pdf
                * abs(wi.y)
            )
            ray.origin = intersection.pos
            ray.direction = shading_frame.to_global(wo)
            depth += 1

        return L

    def render_area(self, area: List[float] = [0, 1, 0, 1]) -> None:
        for x in range(
            int(self.render_settings.WIDTH * area[0]),
            int(self.render_settings.WIDTH * area[1]),
        ):
            for y in range(
                int(self.render_settings.HEIGHT * area[2]),
                int(self.render_settings.HEIGHT * area[3]),
            ):
                u, v = (x + 0.5) / self.render_settings.WIDTH * 2 - 1, (
                    y + 0.5
                ) / self.render_settings.HEIGHT * 2 * self.render_settings.ASPECT - self.render_settings.ASPECT
                ray = self.scene.camera.get_ray(
                    u,
                    v,
                )
                self.render_result.add_to_px(x, y, self.raytrace(ray))

    def render(self) -> None:
        while self.render_result.rendered_passes < self.render_settings.RENDER_PASSES:
            self.render_result.rendered_passes += 1
            self.render_area(self.render_settings.AREA)
            # np.random.seed(np.random.randint(0, 10000))

        self.render_result.finished = True

    def start_render(self, render_settings: RenderSettings) -> RenderResult:
        self.init_render(render_settings)
        t = Thread(target=self.render, daemon=True)
        t.start()
        return self.render_result

    def render_threaded(self, vertical_splits: int, horizontal_splits: int) -> None:
        w = (
            self.render_settings.AREA[1] - self.render_settings.AREA[0]
        ) / vertical_splits
        h = (
            self.render_settings.AREA[3] - self.render_settings.AREA[2]
        ) / horizontal_splits
        areas = []
        for i in range(vertical_splits):
            for j in range(horizontal_splits):
                areas.append(
                    [
                        i * w + self.render_settings.AREA[0],
                        (i + 1) * w + self.render_settings.AREA[0],
                        j * h + self.render_settings.AREA[2],
                        (j + 1) * h + self.render_settings.AREA[2],
                    ]
                )
        while self.render_result.rendered_passes < self.render_settings.RENDER_PASSES:
            self.render_result.rendered_passes += 1
            np.random.seed(np.random.randint(0, 10000))
            threads = [
                Thread(
                    target=self.render_area,
                    args=(area,),
                    daemon=True,
                )
                for area in areas
            ]

            for t in threads:
                t.start()

            for t in threads:
                t.join()

        self.render_result.finished = True

    def start_render_threaded(
        self,
        render_settings: RenderSettings,
        vertical_splits: int,
        horizontal_splits: int,
    ) -> RenderResult:
        self.init_render(render_settings)
        t = Thread(
            target=(self.render_threaded),
            args=(vertical_splits, horizontal_splits),
            daemon=True,
        )
        t.start()
        return self.render_result

    def init_render(self, render_settings: RenderSettings) -> None:
        self.render_settings = render_settings
        self.render_result = RenderResult(
            render_settings.WIDTH,
            render_settings.HEIGHT,
            self.render_settings.RENDER_PASSES,
        )
        self.scene = render_settings.SCENE
