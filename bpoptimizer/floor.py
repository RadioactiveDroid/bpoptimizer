"""This class is used for internally representing a BlockParty floor"""
from typing import List, Tuple, Union

import os
import random
import warnings

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

import cv2
from tqdm import tqdm


class Floor:
    # Constants to recolour and scale floor for display purposes
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)

    CANVAS_COLOURS = {
        0: RED,  # Red
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

    # Used to define requirements expected of given floor
    MAX_COLOURS = len(CANVAS_COLOURS)

    def __init__(self, image: Union[str, np.ndarray], interval: float = 0.25):
        """Reads an image file representing a BlockParty floor and store an internal
        representation of it

        Args:
            image (Union[str, np.ndarray]):
                path to image of floor or a valid numpy array representation of an image
            interval (float, optional):
                dictates how fine the spot calculation will be, defaults to 0.25 meaning
                that every position checked will be 0.25 blocks apart.
                Note: values below 0.25 can get very slow!

        Attributes:
            floor (np.ndarray):
                the internal 2D representation of the floor where each unique colour is
                a different integer

        Raises:
            TypeError: if provided floor is not a path as string or numpy array
            FileNotFoundError: if provided path is invalid
            AssertionError: if floor is of unexpected size or has too many colours
        """
        if isinstance(image, str):
            path = os.path.abspath(image)
            image = cv2.imread(path)

            if image is None:
                raise FileNotFoundError(
                    f'could not find a valid image for given path, "{path}"'
                )
        elif not isinstance(image, np.ndarray):
            raise TypeError(
                f"floor must be either a str path or numpy array, given {type(image)}"
            )

        assert image.shape[0] == image.shape[1], (
            f"provided floor must be square,"
            f"given floor has dimensions {image.shape[:2]}"
        )

        self.width = image.shape[0]
        self.floor = self.simplify_image(image)

        assert self.count <= 16, (
            f"number of colours in floor must be less than {self.MAX_COLOURS},"
            f"given floor has {self.count}"
        )

        assert (
            interval > 0 and interval <= 1
        ), f"Interval must be within the bounds (0, 1], given {interval}"

        # Construct list of all possible standing coordinates
        self._points = np.arange(0, self.width, interval)
        self._interval = interval

        self._recolour = True  # Whether canvas needs to be recoloured before display

        self._optimize()

    @property
    def count(self) -> int:
        """Number of colours in the floor"""
        return len(np.unique(self.floor))

    @property
    def _coords(self) -> np.ndarray:
        """The possible standing coordinates that will be checked

        I chose to compute this on request as opposed to storing it due to the amount
        of memory it takes up.
        """
        return np.transpose(
            [
                np.repeat(self._points, len(self._points)),
                np.tile(self._points, len(self._points)),
            ]
        )

    def display(
        self,
        spot: np.ndarray = None,
        reachable_distance: float = float("inf"),
        scale: int = 30,
        path: str = None,
    ) -> np.ndarray:
        """Displays an image of the floor

        Args:
            spot (np.ndarray, optional):
                if provided, draws the location, lines to all unique colours from the
                location and writes the distance to farther away colours as well
            rechable_distance (float, optional):
                if provided, highlights any spots that would be too far to reach from
                the spot being display
            scale (int, optional):
                factor to upscale original image by, defaults to 30
            path (str, optional):
                saves image to specified path if provided along with returning image as
                numpy array

        Returns:
            np.ndarray:
                the image to display represented as a numpy array
        """

        # Used to offset coordinates to where they should be drawn in display
        def offset(spot: Tuple[float, float]) -> Tuple[float, float]:
            return tuple(
                int((float(x) + (0.5 * self._interval)) * scale)  # type: ignore
                for x in spot
            )[::-1]

        if self._recolour:
            # We map each integer to a colour, turning our floor into a list of pixels
            self._canvas = np.array(
                np.vectorize(lambda i: list(self.CANVAS_COLOURS.values())[i])(
                    self.floor.flatten()
                )
            ).T.reshape(self.width, self.width, 3)

            self._recolour = False

        thickness = round(scale / 10)
        text_scale = scale / 45

        canvas = cv2.resize(
            self._canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )

        for line in range(0, self.width * scale, scale):
            cv2.line(
                canvas,
                (line, 0),
                (line, self.width * scale),
                color=self.BLACK,
                thickness=1,
            )
            cv2.line(
                canvas,
                (0, line),
                (self.width * scale, line),
                color=self.BLACK,
                thickness=1,
            )

        if spot is not None:
            if scale < 20:
                warnings.warn(
                    "A scale below 20 will likely result in an unreadable image."
                )

            spot_index = self._points_to_indicies(spot.reshape(1, 2))[0]
            spot_location = offset(spot)

            cv2.circle(
                canvas,
                spot_location,
                radius=round(scale / 10),
                color=self.WHITE,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

            targets = self._indicies_to_points(self._target_dict[spot_index])
            distances = self._distance_dict[spot_index]

            for target, distance in zip(targets, distances):
                target_location = offset(target)
                line_colour = self.WHITE if distance <= reachable_distance else self.RED

                cv2.line(
                    canvas,
                    spot_location,
                    target_location,
                    color=line_colour,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )

            # We make two seperate loops to ensure that text is always above lines
            for target, distance in zip(targets, distances):
                target_location = offset(target)

                if distance > 2:
                    cv2.putText(
                        canvas,
                        f"{distance:.1f}",
                        target_location,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=text_scale,
                        color=self.BLACK,
                        thickness=thickness,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        canvas,
                        f"{distance:.1f}",
                        target_location,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=text_scale,
                        color=self.WHITE,
                        thickness=round(thickness * 0.25),
                        lineType=cv2.LINE_AA,
                    )

        if path:
            cv2.imwrite(path, cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_BGR2RGB))

        return canvas

    def shuffle(self) -> None:
        """Shuffles colours used for floor display"""
        random.shuffle(self.CANVAS_COLOURS)  # type: ignore
        self._recolour = True

    @staticmethod
    def simplify_image(image: np.ndarray) -> np.ndarray:
        """Removes pixel data from an image to produce an array where each pixel is
        represented by a unique integer

        Args:
            image (np.ndarray):
                the image to simplify

        Returns:
            np.ndarray: the simplified image with the same height and width dimensions
        """
        # We flatten the array into a 1D list of pixels to get all unique pixel colours
        dims = image.shape[:2]
        image = image.reshape(dims[0] * dims[1], 3)

        # Convert to tuples which are hashable and allow for a dict when simplifying
        colours = [tuple(colour) for colour in np.unique(image, axis=0)]
        colours_dict = dict(zip(colours, range(len(colours))))

        image = np.array([colours_dict[tuple(pixel)] for pixel in image])

        return image.reshape(dims)  # Reshape to original size

    def _points_to_indicies(self, points: np.ndarray) -> np.ndarray:
        """Simplify list of coordinate to unsigned integer representation"""
        return (
            (points[:, 0] / self._interval) * len(self._points)
            + (points[:, 1] / self._interval)
        ).astype("uint32")

    def _indicies_to_points(self, indicies: np.ndarray) -> np.ndarray:
        """Retrieve list of coordinates from unsigned integers used to represent them"""
        return np.stack(
            (
                (indicies // len(self._points)) * self._interval,
                (indicies % len(self._points)) * self._interval,
            )
        ).T

    def _optimize(self) -> None:
        """Builds the target dictionary for every position on the given floor

        This dictionary contains, for each position, the locations of the nearest block
        of every colour other than the position itself along with the euclidean distance
        to that location.
        """
        coords = self._coords

        # Scale up floor using Kronecker product to match scale of coords
        scaled_floor = np.kron(
            self.floor, np.ones((int(1 / self._interval), int(1 / self._interval)))
        )

        self._target_dict = np.empty((len(coords), self.count), dtype="uint32")
        self._distance_dict = np.empty(self._target_dict.shape, dtype="float16")

        for colour in tqdm(range(self.count)):
            # Create k-d tree for all coordinate matching the current colour
            tree = KDTree(coords[(scaled_floor == colour).flatten()])

            distance, target = tree.query(coords, k=1, workers=-1)

            self._target_dict[:, colour] = self._points_to_indicies(tree.data[target])
            self._distance_dict[:, colour] = distance

    def get_spots(self, reachable_distance: float = float("inf")) -> np.ndarray:
        """Returns a list of the best spots based on how far you are able to travel

        For determining the best spots, we first try to maximize the number of unique
        colours that can be reached, from those candidates we take the ones which
        minimize the distance to the farthest colour that can be reached and finally
        we reduce those candidates by taking the spots with the lowest sum of squared
        distances to the colours that can be reached. If there are multiple candidates
        left, all are returned.

        Args:
            reachable_distance (float, optional):
                the maximum number of blocks that you can travel,
                no limit if not provided

        Returns:
            List[Tuple[int, int]]: a list of coords for the best spots
        """
        # First filter to spots that reach the most colours
        colours = np.where(self._distance_dict <= reachable_distance, 1, 0).sum(axis=1)
        colours_mask = (colours == colours.max()).nonzero()[0]

        # Next filter to spots that minimize distance to the further unique colour
        farthest = np.where(
            self._distance_dict[colours_mask] <= reachable_distance,
            self._distance_dict[colours_mask],
            0,
        ).max(axis=1)
        farthest_mask = colours_mask[farthest == farthest.min()]

        # Lastly filter to spots which minimize total squared distance to all unique
        # colours. We use squared distance since having all really close and one really
        # far colours should be worse than having all at an equally less close distance
        total = np.where(
            self._distance_dict[farthest_mask] <= reachable_distance,
            self._distance_dict[farthest_mask] ** 2,
            0,
        ).sum(axis=1)
        total_mask = farthest_mask[total == total.min()]

        spots = self._indicies_to_points(total_mask)
        distances = cdist(spots, [[self.width / 2, self.width / 2]]).reshape(
            spots.shape[0]
        )

        return spots[distances.argsort()]
