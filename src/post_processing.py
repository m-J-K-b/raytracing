import numpy as np

from src.util import lerp


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
    def __init__(self, exposure=1, brightness=0, contrast=1, saturation=1, gamma=1):
        self.exposure = exposure
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.gamma = gamma

    def process(self, img_arr):
        processed_img_arr = self.exposure_correction(img_arr)
        processed_img_arr = self.contrast_and_brightness_correction(processed_img_arr)
        processed_img_arr = self.saturation_correction(processed_img_arr)
        processed_img_arr = self.reinhardt_tonemapping(processed_img_arr)
        processed_img_arr = self.gamma_correction(processed_img_arr)
        return processed_img_arr

    def greyscale(self, img):
        return np.dot(img, np.array([0.299, 0.587, 0.114]))

    def exposure_correction(self, img):
        return img * self.exposure

    # def white_balancing(img, temp, tint):
    #     t1 = temp * 10 /6
    #     t2 = tint * 10 /6

    #     x

    def contrast_and_brightness_correction(self, img):
        return np.clip((self.contrast * (img - 0.5) + self.brightness + 0.5), 0, 1)

    def saturation_correction(self, img):
        grey = self.greyscale(img)[:, :, None] * np.array([[[1, 1, 1]]])
        return np.clip((lerp(grey, img, self.saturation)), 0, 1)

    def reinhardt_tonemapping(self, img, a=0.18, saturation=1.0):
        img = img + 1e-8
        Lw = self.greyscale(img)
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
            img / Lw[..., None] * Ld_norm_3d, 0, 1
        )  # apply the tonemapping to each pixel and clip the result to the range [0, 1]
        return output

    def gamma_correction(self, img):
        return np.power(img, self.gamma)
