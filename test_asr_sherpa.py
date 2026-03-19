from __future__ import annotations

from types import SimpleNamespace
import unittest

from asr_sherpa import SherpaASRWorker


class _FakeStream:
    def __init__(self):
        self.samples: list[float] = []

    def accept_waveform(self, sample_rate: int, samples: list[float]):
        self.sample_rate = sample_rate
        self.samples = list(samples)


class _FakeRecognizer:
    def __init__(self, results: list[str]):
        self._results = list(results)
        self._streams: list[_FakeStream] = []

    def create_stream(self) -> _FakeStream:
        stream = _FakeStream()
        self._streams.append(stream)
        return stream

    def decode_stream(self, stream: _FakeStream):
        return None

    def get_result(self, stream: _FakeStream) -> str:
        index = self._streams.index(stream)
        return self._results[index]


class SherpaASRWorkerSenseVoiceTests(unittest.TestCase):
    def _make_worker(self, results: list[str]) -> SherpaASRWorker:
        worker = SherpaASRWorker.__new__(SherpaASRWorker)
        worker.sample_rate = 16000
        worker.vad_max_speech_duration = 20.0
        worker._recognizer = _FakeRecognizer(results)
        worker._stream_started_wall_ts = 1000.0
        worker._vad_processed_samples = 0
        worker._pending_events = []
        return worker

    def test_long_vad_segment_is_split_before_sensevoice_decode(self):
        worker = self._make_worker(["第一段", "第二段"])
        samples = [0.1] * int(21 * worker.sample_rate)
        vad_segment = SimpleNamespace(samples=samples, start=0)

        segments = worker._decode_vad_segment(vad_segment)

        self.assertEqual([seg.text for seg in segments], ["第一段。", "第二段。"])
        self.assertEqual(segments[0].ts_start, 0.0)
        self.assertAlmostEqual(segments[0].ts_end, 18.0, places=2)
        self.assertAlmostEqual(segments[1].ts_start, 18.0, places=2)
        self.assertAlmostEqual(segments[1].ts_end, 21.0, places=2)
        events = worker.drain_events()
        self.assertTrue(any(event.kind == "vad_segment_split" for event in events))


if __name__ == "__main__":
    unittest.main()
