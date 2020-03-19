#!/usr/bin/env python

"""Tests for `floor` module."""

import os
import pytest
import numpy as np

from bpoptimizer import Floor

DIR = os.path.dirname(__file__)


def get_floor(file):
    return Floor(os.path.join(DIR, f"floors/{file}.png"))


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
        assert floor.display(5).shape == ((48 * 5), (48 * 5), 3)


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
    def test_wrong_size():
        with pytest.raises(AssertionError):
            get_floor("wrong_size")

    @staticmethod
    def test_too_many_colours():
        with pytest.raises(AssertionError):
            get_floor("too_many_colours")
