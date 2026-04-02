from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "zones": {
        "general_zone": {
            "type": "general",
            "roi": {"x": 0, "y": 0, "w": 640, "h": 480},
            "area_sqm": 100.0,
            "video_source": "test_general.mp4",
            "calibration_factor": 0.005,
        },
        "gate_zone": {
            "type": "gate",
            "roi": {"x": 0, "y": 0, "w": 640, "h": 480},
            "area_sqm": 50.0,
            "video_source": "test_gate.mp4",
            "tripwire": {"p1": [320, 0], "p2": [320, 480]},
            "in_direction": "right",
        },
    }
}

DUMMY_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def make_pipeline(config=None, threshold=40):
    """Construct a VisionPipeline with all heavy dependencies mocked.

    After construction the pipeline object's .model, .dense_estimators,
    .gate_counters, .flow_analyzer, and .publishers attributes all hold mock
    instances that can be inspected in tests.
    """
    cfg = config or SAMPLE_CONFIG

    with (
        patch("vision.pipeline.YOLO") as mock_yolo_cls,
        patch("vision.pipeline.aioredis") as mock_aioredis,
        patch("vision.pipeline.ZonePublisher") as mock_pub_cls,
        patch("vision.pipeline.GateTripwireCounter") as mock_gate_cls,
        patch("vision.pipeline.OpticalFlowAnalyzer") as mock_flow_cls,
        patch("vision.pipeline.DenseEstimator") as mock_dense_cls,
    ):
        # YOLO: single shared instance
        mock_yolo_cls.return_value = MagicMock()

        # Redis client (async, not used in unit tests)
        mock_aioredis.from_url.return_value = MagicMock()

        # ZonePublisher: one AsyncMock per zone, keyed so we can retrieve them
        pub_store: dict[str, AsyncMock] = {}

        def _pub_factory(redis, zone_id, **kw):
            inst = AsyncMock()
            pub_store[zone_id] = inst
            return inst

        mock_pub_cls.side_effect = _pub_factory

        # GateTripwireCounter: one MagicMock per gate zone
        gate_store: dict = {}
        gate_call_count = 0

        def _gate_factory(**kw):
            nonlocal gate_call_count
            inst = MagicMock()
            gate_store[gate_call_count] = inst
            gate_call_count += 1
            return inst

        mock_gate_cls.side_effect = _gate_factory

        # DenseEstimator: one MagicMock per general zone
        dense_store: dict = {}
        dense_call_count = 0

        def _dense_factory(**kw):
            nonlocal dense_call_count
            inst = MagicMock()
            dense_store[dense_call_count] = inst
            dense_call_count += 1
            return inst

        mock_dense_cls.side_effect = _dense_factory

        # OpticalFlowAnalyzer: single shared mock, analyze() -> empty dict
        mock_flow_instance = MagicMock()
        mock_flow_instance.analyze.return_value = {}
        mock_flow_cls.return_value = mock_flow_instance

        from vision.pipeline import VisionPipeline

        pipeline = VisionPipeline(
            yolo_model_path="fake.pt",
            zone_config=cfg,
            switch_threshold=threshold,
        )

    # After the `with` block the patches are removed, but the pipeline already
    # holds references to the mock *instances*, so they remain usable.
    return pipeline


def _set_yolo_box_count(pipeline, count: int) -> None:
    """Configure pipeline.model() to report *count* detected boxes."""
    mock_result = MagicMock()
    mock_result.boxes = [MagicMock() for _ in range(count)]
    pipeline.model.return_value = [mock_result]


# ---------------------------------------------------------------------------
# 1. Auto-switch: YOLO >= threshold -> DenseEstimator is called
# ---------------------------------------------------------------------------

class TestAutoSwitch:
    def test_dense_estimator_called_when_yolo_exceeds_threshold(self):
        """YOLO returns 50 boxes (>= threshold=40) → DenseEstimator.estimate invoked."""
        pipeline = make_pipeline(threshold=40)
        _set_yolo_box_count(pipeline, 50)

        dense = pipeline.dense_estimators["general_zone"]
        dense.estimate.return_value = 80

        asyncio.run(pipeline.process_frame("general_zone", DUMMY_FRAME, "general"))

        dense.estimate.assert_called_once()

    def test_dense_count_used_as_raw_count_when_above_threshold(self):
        """When dense mode is active the published raw_count comes from DenseEstimator."""
        pipeline = make_pipeline(threshold=40)
        _set_yolo_box_count(pipeline, 55)

        dense = pipeline.dense_estimators["general_zone"]
        dense.estimate.return_value = 120

        asyncio.run(pipeline.process_frame("general_zone", DUMMY_FRAME, "general"))

        publisher = pipeline.publishers["general_zone"]
        publish_kwargs = publisher.publish.call_args.kwargs
        assert publish_kwargs["raw_count"] == 120
        assert publish_kwargs["source"] == "cv_dense"


# ---------------------------------------------------------------------------
# 2. Sparse path: YOLO < threshold -> DenseEstimator is NOT called
# ---------------------------------------------------------------------------

class TestSparsePath:
    def test_dense_estimator_not_called_when_yolo_below_threshold(self):
        """YOLO returns 10 boxes (< threshold=40) → DenseEstimator.estimate not invoked."""
        pipeline = make_pipeline(threshold=40)
        _set_yolo_box_count(pipeline, 10)

        asyncio.run(pipeline.process_frame("general_zone", DUMMY_FRAME, "general"))

        dense = pipeline.dense_estimators["general_zone"]
        dense.estimate.assert_not_called()

    def test_yolo_count_used_as_raw_count_in_sparse_mode(self):
        """In sparse mode the published raw_count equals the YOLO detection count."""
        pipeline = make_pipeline(threshold=40)
        _set_yolo_box_count(pipeline, 15)

        asyncio.run(pipeline.process_frame("general_zone", DUMMY_FRAME, "general"))

        publisher = pipeline.publishers["general_zone"]
        publish_kwargs = publisher.publish.call_args.kwargs
        assert publish_kwargs["raw_count"] == 15
        assert publish_kwargs["source"] == "cv_yolo"

    def test_threshold_boundary_is_inclusive(self):
        """YOLO count exactly equal to switch_threshold triggers dense mode."""
        pipeline = make_pipeline(threshold=40)
        _set_yolo_box_count(pipeline, 40)

        dense = pipeline.dense_estimators["general_zone"]
        dense.estimate.return_value = 60

        asyncio.run(pipeline.process_frame("general_zone", DUMMY_FRAME, "general"))

        dense.estimate.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Warmup bypass: DenseEstimator returns -1 -> YOLO count used as fallback
# ---------------------------------------------------------------------------

class TestWarmupBypass:
    def test_yolo_count_used_when_dense_returns_minus_one(self):
        """DenseEstimator signals warmup (-1) → pipeline falls back to YOLO count."""
        pipeline = make_pipeline(threshold=40)
        _set_yolo_box_count(pipeline, 50)

        dense = pipeline.dense_estimators["general_zone"]
        dense.estimate.return_value = -1  # warmup sentinel

        asyncio.run(pipeline.process_frame("general_zone", DUMMY_FRAME, "general"))

        publisher = pipeline.publishers["general_zone"]
        publish_kwargs = publisher.publish.call_args.kwargs
        assert publish_kwargs["raw_count"] == 50
        assert publish_kwargs["source"] == "cv_yolo"

    def test_dense_estimator_still_called_during_warmup(self):
        """Even when the estimator is warming up it is still called (it updates the model)."""
        pipeline = make_pipeline(threshold=40)
        _set_yolo_box_count(pipeline, 50)

        dense = pipeline.dense_estimators["general_zone"]
        dense.estimate.return_value = -1

        asyncio.run(pipeline.process_frame("general_zone", DUMMY_FRAME, "general"))

        dense.estimate.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Gate routing: zone_type="gate" -> GateTripwireCounter is invoked
# ---------------------------------------------------------------------------

class TestGateRouting:
    def test_gate_counter_process_frame_called(self):
        """zone_type='gate' → GateTripwireCounter.process_frame is invoked."""
        pipeline = make_pipeline()
        _set_yolo_box_count(pipeline, 5)

        asyncio.run(pipeline.process_frame("gate_zone", DUMMY_FRAME, "gate"))

        gate_counter = pipeline.gate_counters["gate_zone"]
        gate_counter.process_frame.assert_called_once_with(DUMMY_FRAME)

    def test_dense_estimator_not_called_for_gate_zone(self):
        """Gate zones never invoke DenseEstimator."""
        pipeline = make_pipeline()
        _set_yolo_box_count(pipeline, 5)

        asyncio.run(pipeline.process_frame("gate_zone", DUMMY_FRAME, "gate"))

        # No dense estimator should be created for gate zones.
        assert "gate_zone" not in pipeline.dense_estimators

    def test_gate_zone_publishes_with_cv_source(self):
        """Gate zone publishes with source='cv'."""
        pipeline = make_pipeline()
        _set_yolo_box_count(pipeline, 5)

        asyncio.run(pipeline.process_frame("gate_zone", DUMMY_FRAME, "gate"))

        publisher = pipeline.publishers["gate_zone"]
        publish_kwargs = publisher.publish.call_args.kwargs
        assert publish_kwargs["source"] == "cv"


# ---------------------------------------------------------------------------
# 5. Video loop: cap.read() returning False triggers cap.set reset
# ---------------------------------------------------------------------------

class TestVideoLoop:
    def test_cap_set_called_to_loop_video_on_read_failure(self):
        """When cap.read() fails, the pipeline resets the video to frame 0."""
        pipeline = make_pipeline()

        with (
            patch("vision.pipeline.cv2.VideoCapture") as mock_cap_cls,
            patch.object(pipeline, "process_frame", new_callable=AsyncMock),
        ):
            mock_cap = MagicMock()
            mock_cap_cls.return_value = mock_cap

            # First read: end-of-file signal.
            # Second read (after reset): also fails → mark_stale then break.
            mock_cap.read.side_effect = [(False, None), (False, None)]

            pipeline.publishers["general_zone"].mark_stale = AsyncMock()

            asyncio.run(
                pipeline.run_feed("general_zone", "test_general.mp4", "general")
            )

            # cap.set must have been called to rewind the video.
            mock_cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 0)

    def test_mark_stale_called_when_feed_fails_after_reset(self):
        """If the feed fails even after rewinding, mark_stale is published."""
        pipeline = make_pipeline()

        with (
            patch("vision.pipeline.cv2.VideoCapture") as mock_cap_cls,
            patch.object(pipeline, "process_frame", new_callable=AsyncMock),
        ):
            mock_cap = MagicMock()
            mock_cap_cls.return_value = mock_cap
            mock_cap.read.side_effect = [(False, None), (False, None)]

            pipeline.publishers["general_zone"].mark_stale = AsyncMock()

            asyncio.run(
                pipeline.run_feed("general_zone", "test_general.mp4", "general")
            )

            pipeline.publishers["general_zone"].mark_stale.assert_called_once()

    def test_process_frame_called_after_successful_loop(self):
        """After rewinding, if the next read succeeds, processing continues normally."""
        pipeline = make_pipeline()

        with (
            patch("vision.pipeline.cv2.VideoCapture") as mock_cap_cls,
            patch.object(pipeline, "process_frame", new_callable=AsyncMock),
        ):
            mock_cap = MagicMock()
            mock_cap_cls.return_value = mock_cap

            # First read: fails (end of file).
            # Second read: succeeds (after rewind).
            # Third read: also fails to end the loop cleanly.
            mock_cap.read.side_effect = [
                (False, None),               # triggers rewind
                (True, DUMMY_FRAME),          # post-rewind success
                (False, None),               # next iteration: end again
                (False, None),               # after second rewind: permanent fail
            ]

            pipeline.publishers["general_zone"].mark_stale = AsyncMock()

            asyncio.run(
                pipeline.run_feed("general_zone", "test_general.mp4", "general")
            )

            # process_frame should have been called at least once (for the successful frame).
            pipeline.process_frame.assert_called()
