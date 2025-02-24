from threading import Thread

import numpy as np

from src.core.util import lerp


class PostProcessing:
    ###### POST PROCESSING ######
    # Post processing pipeline:
    # 1. Fog
    # 2. Bloom
    # 3. Exposure
    # 4. White Balancing
    # 5. Contrast
    # 6. Brightness
    # 7. Color Filtering
    # 8. Saturation
    # 9. Tone Mapping
    # 9. Gamma Correction
    def __init__(
        self,
        exposure: int | float = 1,
        brightness: int | float = 0,
        contrast: int | float = 1,
        saturation: int | float = 1,
        gamma: int | float = 1,
    ) -> None:
        self.exposure = exposure
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.gamma = gamma

    def process(self, img_arr: np.ndarray) -> np.ndarray:
        processed_img_arr = self.exposure_correction(img_arr)
        processed_img_arr = self.contrast_and_brightness_correction(processed_img_arr)
        processed_img_arr = self.saturation_correction(processed_img_arr)
        processed_img_arr = self.reinhardt_tone_mapping(processed_img_arr)
        processed_img_arr = self.gamma_correction(processed_img_arr)
        return processed_img_arr

    def grayscale(self, img_arr: np.ndarray) -> np.ndarray:
        return np.dot(img_arr, np.array([0.299, 0.587, 0.114]))

    def exposure_correction(self, img) -> np.ndarray:
        return img * self.exposure

    # def white_balancing(img, temp, tint):
    #     t1 = temp * 10 /6
    #     t2 = tint * 10 /6

    #     x

    def contrast_and_brightness_correction(self, img_arr: np.ndarray) -> np.ndarray:
        return np.clip((self.contrast * (img_arr - 0.5) + self.brightness + 0.5), 0, 1)

    def saturation_correction(self, img_arr: np.ndarray) -> np.ndarray:
        gray = self.grayscale(img_arr)[..., None] * np.array([[[1, 1, 1]]])  # type: ignore
        return np.clip((lerp(gray, img_arr, self.saturation)), 0, 1)

    def reinhardt_tone_mapping(
        self, img_arr: np.ndarray, a=0.18, saturation=1.0
    ) -> np.ndarray:
        img_arr = img_arr + 1e-8
        Lw = self.grayscale(img_arr)
        Lwa = np.exp(np.mean(np.log(Lw)))  # calculate the global adaptation luminance
        Lm = a / Lwa * Lw  # calculate the adapted luminance
        Ld = (
            Lm * (1 + Lm / (saturation**2)) / (1 + Lm)
        )  # apply the tonemapping function
        Ld_norm = Ld / np.max(Ld)  # normalize the output luminance to the range [0, 1]
        Ld_norm_3d = np.stack(
            (Ld_norm, Ld_norm, Ld_norm), axis=-1
        )  # create a 3-channel image from the luminance values
        output = np.clip(
            img_arr / Lw[..., None] * Ld_norm_3d, 0, 1  # type: ignore
        )  # apply the tonemapping to each pixel and clip the result to the range [0, 1]
        return output

    def gamma_correction(self, img_arr: np.ndarray) -> np.ndarray:
        return np.power(img_arr, self.gamma)
