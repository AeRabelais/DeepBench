from src.deepbench.image.image import Image
from src.deepbench import astro_object

import numpy as np
from scipy import ndimage
import warnings


class TimeSeriesImage(Image):
    def __init__(self, objects, image_shape):

        # Requires the object to be a single dictionary
        assert len(image_shape) >= 2, "Image shape must be >=2"
        assert len(objects) == 1, "Must pass exactly one image"

        super().__init__(objects, image_shape)

    def combine_objects(self):

        """
        Utilize Image._generate_astro_objects to overlay all selected astro objects into one image
        If object parameters are not included in object list, defaults are used.
        Updates SkyImage.image.

        Current input parameter assumptions (totally up to change):
        [{
            "object_type":"<object_type>",
            "object_parameters":{<parameters for that object>}
        }]

        """

        generative_object = self.objects[0]
        if "object_type" in generative_object.keys():
            object_type = generative_object["object_type"]

            object_parameters = (
                {}
                if "object_parameters" not in generative_object.keys()
                else generative_object["object_parameters"]
            )

            additional_object = self._generate_astro_object(
                object_type, object_parameters
            )

            self.image = additional_object.create_object()

        else:
            # TODO Test for this check
            warnings.warn("Parameters 'object_type' needed to generate")

    def generate_noise(self, noise_type, **kwargs):
        """
        Add noise to an image
        Updates SkyImage.image

        :param noise_type: Type of noise add. Select from [“gaussian”,“poisson”]
        :param kwargs: arg required for the noise. ["gaussian"->sigma, "poisson"->lam]
        """
        noise_map = {
            "gaussian": self._generate_gaussian_noise,
            "poisson": self._generate_poisson_noise,
        }

        if noise_type not in noise_map.keys():
            raise NotImplementedError(f"{noise_type} noise type not available")

        assert (
            self.image is not None
        ), "Image not generated, please run SkyImage.combine_objects"

        noise = noise_map[noise_type](**kwargs)
        self.image += noise

    def _generate_gaussian_noise(self, sigma=1):
        return ndimage.gaussian_filter(self.image, sigma=sigma)

    def _generate_poisson_noise(self, image_noise=1):
        return np.random.poisson(lam=image_noise, size=self.image.shape)
