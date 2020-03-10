"""This class is used for internally representing a BlockParty floor"""
from typing import Union

import random
import cv2
import numpy as np


class Floor:
    # Used to define requirements expected of given floor
    FLOOR_WIDTH = 48
    MAX_COLOURS = 16

    # Constants to recolour and scale floor for display purposes
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    CANVAS_COLOURS = {
        0: (255, 0, 0),  # Red
        1: (255, 152, 173),  # Pink
        2: (255, 165, 0),  # Orange
        3: (255, 255, 0),  # Yellow
        4: (191, 255, 0),  # Lime
        5: (0, 255, 0),  # Green
        6: (0, 128, 128),  # Cyan
        7: (48, 204, 255),  # Light Blue
        8: (0, 0, 255),  # Blue
        9: (128, 0, 128),  # Purple
        10: (255, 0, 255),  # Magenta
        11: (101, 67, 33),  # Brown
        12: (100, 100, 100),  # Light Gray
        13: (220, 220, 220),  # Gray
        14: BLACK,  # Black
        15: WHITE,  # White
    }
    CANVAS_SCALE = 30

    def __init__(self, image: Union[str, np.ndarray]):
        """Reads an image file representing a BlockParty floor and store an internal
        representation of it

        Args:
            image (Union[str, np.ndarray]): path to image of floor or a valid numpy
                                            array representation of an image

        Attributes:
            floor (np.ndarray): the internal 2D representation of the floor where each
                                unique colour is a different integer

        Raises:
            TypeError: if provided floor is not a path as string or numpy array
            FileNotFoundError: if provided path is invalid
            AssertionError: if floor is of unexpected size or has too many colours
        """
        if isinstance(image, str):
            image = cv2.imread(image)

            if image is None:
                raise FileNotFoundError(
                    f'could not find a valid image for given path, "{str}"'
                )
        elif not isinstance(image, np.ndarray):
            raise TypeError(
                f"floor must be either a str path or numpy array, given {type(image)}"
            )

        assert image.shape[0] == image.shape[1] == self.FLOOR_WIDTH, (
            f"provided floor must be square of side length {self.FLOOR_WIDTH},"
            f"given floor has dimensions {self.floor.shape[:2]}"
        )

        self.floor = self.simplify_image(image)

        assert self.floor.count <= 16, (
            f"number of colours in floor must be less than {self.MAX_COLOURS},"
            f"given floor has {self.floor.count}"
        )

        self.shuffle()  # Initial shuffle to initialize canvas for display purposes

    @property
    def count(self):
        """Returns the number of colours in the floor

        Returns:
            int: number of colours in the floor
        """
        return len(np.unique(self.floor))

    def display(self, scale: int = None, path: str = None):
        """Displays an image of the floor

        Args:
            scale (int, optional): factor to upscale original image by, defaults to 30
            path (str, optional): saves image to specified path if provided along with
                                  returning image as numpy array

        Returns:
            np.ndarray: the image to display represented as a numpy array
        """
        if scale is None:
            scale = self.CANVAS_SCALE

        canvas = cv2.resize(
            self._canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )

        if path:
            cv2.imwrite(path, cv2.cvtColor(cv2.COLOR_BGR2RGB))

        return canvas

    def shuffle(self):
        """Shuffles colours used for floor display
        """
        random.shuffle(self.CANVAS_COLOURS)

        # We map each integer to a colour, turning our floor back into a list of pixels
        canvas = np.array(np.vectorize(self.CANVAS_COLOURS.get)(self.floor.flatten())).T
        self._canvas = canvas.reshape(self.FLOOR_WIDTH, self.FLOOR_WIDTH, 3)

    @staticmethod
    def simplify_image(image: np.ndarray):
        """Removes pixel data from an image to produce an array where each pixel is
        represented by a unique integer

        Args:
            image (np.ndarray): the image to simplify

        Returns:
            np.ndarray: the simplified image with the same height and width dimensions
        """
        # We flatten the array into a 1D list of pixels to get all unique pixel colours
        dims = image.shape[:2]
        image = image.reshape(dims[0] * dims[1], 3)

        # Convert to tuples which are hashable and allow for a dict when simplifying
        colours = [tuple(colour) for colour in np.unique(image, axis=0)]
        colours = dict(zip(colours, range(len(colours))))

        image = np.array([colours[tuple(pixel)] for pixel in image])

        return image.reshape(dims)  # Reshape to original size
