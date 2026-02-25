"""Tests for Assignment 8: Murder Mystery Agents.

Tests check output artifacts only — students run the notebook first,
then these tests verify the saved results.
"""

import json
import os
import pytest

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")


# ---------- Part 1: Murder at the Mountain Cabin ----------

class TestPart1:
    @pytest.fixture(autouse=True)
    def load_results(self):
        path = os.path.join(OUTPUT_DIR, "part1_results.json")
        if os.path.exists(path):
            with open(path) as f:
                self.results = json.load(f)
        else:
            self.results = None

    def test_part1_results_exist(self):
        assert self.results is not None, "output/part1_results.json not found — run the notebook first"

    def test_part1_has_required_fields(self):
        assert self.results is not None, "output/part1_results.json not found"
        for field in ["killer", "weapon", "motive", "evidence", "transcript"]:
            assert field in self.results, f"Missing field: {field}"

    def test_part1_killer_is_larry(self):
        assert self.results is not None, "output/part1_results.json not found"
        assert "larry" in self.results["killer"].lower(), (
            f"Expected killer to be Larry, got: {self.results['killer']}"
        )

    def test_part1_evidence_nonempty(self):
        assert self.results is not None, "output/part1_results.json not found"
        assert isinstance(self.results["evidence"], list), "evidence should be a list"
        assert len(self.results["evidence"]) > 0, "evidence list is empty"

    def test_part1_transcript_shows_investigation(self):
        assert self.results is not None, "output/part1_results.json not found"
        assert isinstance(self.results["transcript"], list), "transcript should be a list"
        assert len(self.results["transcript"]) >= 5, (
            f"Transcript has {len(self.results['transcript'])} entries — "
            "agent should use tools at least 5 times"
        )


# ---------- Part 2: Death at St. Mercy Hospital ----------

class TestPart2:
    @pytest.fixture(autouse=True)
    def load_results(self):
        path = os.path.join(OUTPUT_DIR, "part2_results.json")
        if os.path.exists(path):
            with open(path) as f:
                self.results = json.load(f)
        else:
            self.results = None

    def test_part2_results_exist(self):
        assert self.results is not None, "output/part2_results.json not found — run the notebook first"

    def test_part2_has_required_fields(self):
        assert self.results is not None, "output/part2_results.json not found"
        for field in ["killer", "weapon", "time_of_death", "reasoning", "transcript"]:
            assert field in self.results, f"Missing field: {field}"

    def test_part2_killer_is_blake(self):
        assert self.results is not None, "output/part2_results.json not found"
        assert "blake" in self.results["killer"].lower(), (
            f"Expected killer to be Dr. Blake, got: {self.results['killer']}"
        )

    def test_part2_weapon_is_syringe(self):
        assert self.results is not None, "output/part2_results.json not found"
        assert "syringe" in self.results["weapon"].lower(), (
            f"Expected weapon to be syringe, got: {self.results['weapon']}"
        )

    def test_part2_time_of_death(self):
        assert self.results is not None, "output/part2_results.json not found"
        assert "9:30" in self.results["time_of_death"], (
            f"Expected time of death to include 9:30, got: {self.results['time_of_death']}"
        )

    def test_part2_reasoning_substantive(self):
        assert self.results is not None, "output/part2_results.json not found"
        assert len(self.results["reasoning"]) >= 50, (
            f"Reasoning is only {len(self.results['reasoning'])} chars — "
            "provide detailed logical reasoning"
        )
