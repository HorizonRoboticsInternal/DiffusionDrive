from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Union
from abc import abstractmethod

from dataclasses import dataclass, field
import numpy.typing as npt
import cv2
import numpy as np
from numpy import random
import PIL
from PIL import Image

import torch
import torchvision

FeatureDataType = Union[npt.NDArray[np.float32], torch.Tensor]

@dataclass
class Cameras:
    intrinsics: FeatureDataType  # [1, N, 3, 3]
    extrinsics: FeatureDataType  # [1, N, 4, 4]
    imgs: Optional[List[FeatureDataType]] = None
    distortions: Optional[List[FeatureDataType]] = None
    img_filename: Optional[List[str]] = None


class AbstractCameraPipelines(object):
    @abstractmethod
    def __call__(self, cameras: Cameras) -> Cameras:
        pass

    @abstractmethod
    def __repr__(self):
        pass


def resize_and_crop_image(
        img: PIL.Image.Image,
        resize_dims: Tuple,
        crop: Tuple,
):
    """ Bilinear resizing followed by cropping."""
    img = img.resize(resize_dims, resample=PIL.Image.BILINEAR)
    img = img.crop(crop)
    return img


def update_intrinsics(intrinsics: np.ndarray,
                      top_crop: float = 0.0,
                      left_crop: float = 0.0,
                      scale_width: float = 1.0,
                      scale_height: float = 1.0):
    """
    Update camera intrinsics based on crop and scale parameters.

    :param intrinsics: np.ndarray with last two dimension with shape [3, 3]
    :param top_crop: float
    :param left_crop: float
    :param scale_width: float
    :param scale_height: float
    """
    updated_intrinsics = intrinsics.copy()
    # Adjust intrinsics scale due to resizing
    updated_intrinsics[..., 0, 0] *= scale_width
    updated_intrinsics[..., 0, 2] *= scale_width
    updated_intrinsics[..., 1, 1] *= scale_height
    updated_intrinsics[..., 1, 2] *= scale_height

    # Adjust principal point due to cropping
    updated_intrinsics[..., 0, 2] -= left_crop
    updated_intrinsics[..., 1, 2] -= top_crop

    return updated_intrinsics.astype(np.float32)


class RemoveDistortionMultiViewImages(AbstractCameraPipelines):
    def __init__(self, to_pil: bool = True):
        self.to_pil = to_pil
        self.blur = True

    def __call__(self, cameras: Cameras) -> Cameras:
        imgs = cameras.imgs
        intrinsics = cameras.intrinsics[0]  # [1, 6, 3, 3] -> [6, 3, 3]
        distortions = cameras.distortions
        undistorted_imgs = []
        for img, intrinsic, distortion in zip(imgs, intrinsics, distortions):
            undistorted_img = cv2.undistort(img, intrinsic, distortion)
            if self.blur:
                undistorted_img = cv2.GaussianBlur(undistorted_img, (31, 31), 0)
            if self.to_pil:
                undistorted_img = Image.fromarray(undistorted_img)
            undistorted_imgs.append(undistorted_img)

        cameras.imgs = undistorted_imgs
    
        return cameras

class CarlaSimulationInit(AbstractCameraPipelines):
    def __init__(self):
        pass

    def __call__(self, cameras: Cameras) -> Cameras:
        imgs = []
        for img in cameras.imgs:
            img = Image.fromarray(img)
            imgs.append(img)

        cameras.imgs = imgs
    
        return cameras


class PhotoMetricDistortionMultiViewImage(AbstractCameraPipelines):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    :param brightness_delta (int): delta of brightness.
    :param contrast_range (tuple): range of contrast.
    :param saturation_range (tuple): range of saturation.
    :param hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, cameras: Cameras) -> Cameras:
        """Call function to perform photometric distortion on images.
        :param cameras: Result cameras object from loading pipeline.
        :return cameras: Result cameras object with images distorted.
        """
        imgs = cameras.imgs
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                       self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                              self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        cameras.imgs = new_imgs
        return cameras

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


class LoadMultiViewImages(AbstractCameraPipelines):
    """Load multi channel images from a list of separate channel files.

    Expect results['img_filename'] to be a list of filenames.
    """

    def __init__(self):
        pass

    def __call__(self, cameras: Cameras) -> Cameras:
        """Call function to load multi-view image from files.

        :param cameras: Result cameras object containing multi-view image filenames.
        :return cameras: The result cameras object containing the multi-view image data.
        """
        cameras.imgs = []
        for image_filename in cameras.img_filename:
            img = Image.open(image_filename)
            cameras.imgs.append(img)
        return cameras

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


class CropMultiViewImages(AbstractCameraPipelines):
    """Resize and crop multi channel images"""

    def __init__(
        self,
        image_params: Dict,
    ):
        self.crop = self.get_cropping_parameters(
            image_params)

    @staticmethod
    def get_cropping_parameters(image_params):
        original_height, original_width = image_params[
            'original_height'], image_params['original_width']
        final_height, final_width = image_params['final_dim']

        crop_h = image_params['top_crop']
        crop_w = int(max(0, (original_width - final_width) / 2))
        # Left, top, right, bottom crops.
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        return crop

    def __call__(self, cameras: Cameras) -> Cameras:

        for i in range(len(cameras.imgs)):
            cameras.imgs[i] = cameras.imgs[i].crop(self.crop)

        return cameras

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

class ResizeCropMultiViewImages(AbstractCameraPipelines):
    """Resize and crop multi channel images"""

    def __init__(
        self,
        image_params: Dict,
        update_intrinsics: bool = True,
    ):
        self.augmentation_parameters = self.get_resizing_and_cropping_parameters(
            image_params)
        self.update_intrinsics = update_intrinsics

    @staticmethod
    def get_resizing_and_cropping_parameters(image_params):
        original_height, original_width = image_params[
            'original_height'], image_params['original_width']
        final_height, final_width = image_params['final_dim']

        resize_scale = image_params['resize_scale']
        resize_dims = (int(original_width * resize_scale),
                       int(original_height * resize_scale))
        resized_width, resized_height = resize_dims

        crop_h = image_params['top_crop']
        crop_w = int(max(0, (resized_width - final_width) / 2))
        # Left, top, right, bottom crops.
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        if resized_width != final_width:
            print('Zero padding left and right parts of the image.')
        if crop_h + final_height != resized_height:
            print('Zero padding bottom part of the image.')

        return {
            'scale_width': resize_scale,
            'scale_height': resize_scale,
            'resize_dims': resize_dims,
            'crop': crop,
        }

    def __call__(self, cameras: Cameras) -> Cameras:

        for i in range(len(cameras.imgs)):
            cameras.imgs[i] = resize_and_crop_image(
                cameras.imgs[i], self.augmentation_parameters['resize_dims'],
                self.augmentation_parameters['crop'])
        # Combine resize/cropping in the intrinsics
        top_crop = self.augmentation_parameters['crop'][1]
        left_crop = self.augmentation_parameters['crop'][0]

        if self.update_intrinsics:
            intrinsics = cameras.intrinsics
            updated_intrinsics = update_intrinsics(
                intrinsics, top_crop, left_crop,
                scale_width=self.augmentation_parameters['scale_width'],
                scale_height=self.augmentation_parameters['scale_height']
            )
            cameras.intrinsics = updated_intrinsics

        return cameras

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


class NormalizeMultiviewImage(AbstractCameraPipelines):
    """Normalize the image. PIL -> torch.Tensor
    :param mean (sequence): Mean values of 3 channels.
    :param std (sequence): Std values of 3 channels.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalise_image = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, cameras: Cameras) -> Cameras:
        """Call function to normalize images.
        :param cameras: Result cameras object from loading pipeline.
        :param cameras: Normalized images in cameras object.
        """

        for i in range(len(cameras.imgs)):
            cameras.imgs[i] = self.normalise_image(cameras.imgs[i])

        return cameras

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std})'
        return repr_str