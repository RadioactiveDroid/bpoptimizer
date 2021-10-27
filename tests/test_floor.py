#!/usr/bin/env python

"""Tests for `floor` module."""

import os
import pytest
import numpy as np

from bpoptimizer import Floor

DIR = os.path.dirname(__file__)


def get_floor(file, interval=0.25):
    return Floor(os.path.join(DIR, f"floors/{file}.png"), interval=interval)


class TestFloor:
    @staticmethod
    def test_valid_floor():
        floor = get_floor("valid_floor")

        assert floor.count == 16
        assert np.sum(floor.floor) == sum(range(16)) * (12 * 12)

    @staticmethod
    def test_canvas():
        floor = get_floor("valid_floor")

        assert floor.display().shape == ((48 * 30), (48 * 30), 3)
        assert floor.display(scale=5).shape == ((48 * 5), (48 * 5), 3)


class TestFloorExceptions:
    @staticmethod
    def test_bad_path():
        with pytest.raises(FileNotFoundError):
            get_floor("no_file")

    @staticmethod
    def test_bad_type():
        with pytest.raises(TypeError):
            Floor(0)

    @staticmethod
    def test_not_square():
        with pytest.raises(AssertionError):
            get_floor("not_square")

    @staticmethod
    def test_too_many_colours():
        with pytest.raises(AssertionError):
            get_floor("too_many_colours")


class TestOptimize:
    @staticmethod
    def test_find_spots():
        floor = get_floor("valid_floor")

        assert floor.get_spots() == [
            (23.75, 23.75),
            (23.75, 24.0),
            (24.0, 23.75),
            (24.0, 24.0),
        ]

        assert floor.get_spots(reachable_distance=16) == [
            (20.75, 23.75),
            (20.75, 24.0),
            (23.75, 20.75),
            (23.75, 27.0),
            (24.0, 20.75),
            (24.0, 27.0),
            (27.0, 23.75),
            (27.0, 24.0),
        ]

    @staticmethod
    def test_different_interval():
        floor = get_floor("valid_floor", interval=0.5)

        spots = floor.get_spots(reachable_distance=10)

        assert len(spots) == 16
        assert all([(spot in spots) for spot in ((18.0, 29.5), (29.5, 17.5))])
