from __future__ import annotations

import array
import importlib
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from astrbot.api import logger

try:  # pragma: no cover
    from .models import ASRSegment
except ImportError:  # pragma: no cover
    from models import ASRSegment


RKNN_THREAD_VALUES = {1, 0, -1, -2, -3, -4}
SENTENCE_END_CHARS = "。！？!?；;.\n"
DEFAULT_SENSEVOICE_SAFE_DECODE_SECONDS = 18.0
ASR_STRATEGY_SENSEVOICE_VAD_OFFLINE = "sensevoice_vad_offline"


@dataclass(frozen=True, slots=True)
class SherpaModelSpec:
    provider: str
    tokens: Path
    model_dir: Path
    model_format: str
    architecture: str
    model: Path | None = None


@dataclass(frozen=True, slots=True)
class SherpaRuntimeProbeResult:
    enabled: bool
    reason: str
    provider: str = ""
    model_format: str = ""
    threads: int = 1
    asr_strategy: str = ASR_STRATEGY_SENSEVOICE_VAD_OFFLINE
    spec: SherpaModelSpec | None = None
    vad_model_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ASRDebugEvent:
    kind: str
    message: str
    level: str = "info"
    wall_ts: float = 0.0
    ts_start: float = 0.0
    ts_end: float = 0.0
    text: str = ""


class SherpaRuntimeAdapter:
    def __init__(
        self,
        *,
        model_dir: str,
        sample_rate: int = 16000,
        threads: int = 1,
        vad_model_path: str = "",
        vad_threshold: float = 0.3,
        vad_min_silence_duration: float = 0.35,
        vad_min_speech_duration: float = 0.25,
        vad_max_speech_duration: float = 20.0,
        sense_voice_language: str = "auto",
        sense_voice_use_itn: bool = True,
        sherpa_module=None,
        wheel_support_checker=None,
    ):
        self.model_dir = model_dir
        self.sample_rate = max(8000, int(sample_rate or 16000))
        self.threads = int(threads or 1)
        self.asr_strategy = ASR_STRATEGY_SENSEVOICE_VAD_OFFLINE
        self.vad_model_path = str(vad_model_path or "").strip()
        self.vad_threshold = float(vad_threshold or 0.3)
        self.vad_min_silence_duration = max(0.0, float(vad_min_silence_duration or 0.35))
        self.vad_min_speech_duration = max(0.0, float(vad_min_speech_duration or 0.25))
        self.vad_max_speech_duration = max(0.1, float(vad_max_speech_duration or 20.0))
        self.sense_voice_language = str(sense_voice_language or "auto").strip() or "auto"
        self.sense_voice_use_itn = bool(sense_voice_use_itn)
        self._sherpa_module = sherpa_module
        self._wheel_support_checker = wheel_support_checker or sherpa_wheel_has_rknn_support

    def probe(self) -> SherpaRuntimeProbeResult:
        try:
            spec = detect_sherpa_model(self.model_dir)
            vad_model_path = Path(self.vad_model_path).expanduser().resolve()
            if not vad_model_path.exists() or not vad_model_path.is_file():
                raise FileNotFoundError(f"ASR VAD model missing: {vad_model_path}")
        except Exception as e:
            return SherpaRuntimeProbeResult(enabled=False, reason=str(e))

        try:
            sherpa_onnx = self._sherpa_module or importlib.import_module("sherpa_onnx")
        except Exception as e:
            return SherpaRuntimeProbeResult(
                enabled=False,
                reason=f"sherpa_onnx unavailable: {e}",
            )

        threads = normalize_sherpa_threads(spec.provider, self.threads)
        if spec.provider == "rknn" and not self._wheel_support_checker(sherpa_onnx):
            return SherpaRuntimeProbeResult(
                enabled=False,
                reason="current sherpa_onnx wheel has no RKNN support (ldd missing librknnrt.so)",
                provider=spec.provider,
                model_format=spec.model_format,
                threads=threads,
                spec=spec,
                vad_model_path=vad_model_path,
            )

        self._sherpa_module = sherpa_onnx
        return SherpaRuntimeProbeResult(
            enabled=True,
            reason="",
            provider=spec.provider,
            model_format=spec.model_format,
            threads=threads,
            spec=spec,
            vad_model_path=vad_model_path,
        )

    def create_recognizer(self, probe: SherpaRuntimeProbeResult):
        if not probe.enabled or probe.spec is None:
            raise RuntimeError(probe.reason or "runtime probe is disabled")
        sherpa_onnx = self._sherpa_module or importlib.import_module("sherpa_onnx")
        spec = probe.spec
        if not spec.model:
            raise RuntimeError("sense voice model layout is incomplete")

        kwargs = dict(
            model=str(spec.model),
            tokens=str(spec.tokens),
            provider=probe.provider,
            num_threads=probe.threads,
            use_itn=self.sense_voice_use_itn,
        )
        language = normalize_sense_voice_language(self.sense_voice_language)
        if language and language not in {"auto", "auto-detect"}:
            kwargs["language"] = language

        try:
            return sherpa_onnx.OfflineRecognizer.from_sense_voice(**kwargs)
        except TypeError:
            kwargs.pop("language", None)
            return sherpa_onnx.OfflineRecognizer.from_sense_voice(**kwargs)

    def create_vad(self, probe: SherpaRuntimeProbeResult):
        if probe.vad_model_path is None:
            raise RuntimeError("VAD model path is unavailable")
        sherpa_onnx = self._sherpa_module or importlib.import_module("sherpa_onnx")
        config = sherpa_onnx.VadModelConfig()
        silero = getattr(config, "silero_vad", None)
        if silero is not None:
            silero.model = str(probe.vad_model_path)
            silero.threshold = self.vad_threshold
            silero.min_silence_duration = self.vad_min_silence_duration
            silero.min_speech_duration = self.vad_min_speech_duration
            silero.max_speech_duration = self.vad_max_speech_duration
        if hasattr(config, "sample_rate"):
            config.sample_rate = self.sample_rate
        if hasattr(config, "num_threads"):
            config.num_threads = max(1, int(self.threads if self.threads > 0 else 1))
        if hasattr(config, "provider"):
            config.provider = "cpu"
        buffer_size_in_seconds = max(30.0, self.vad_max_speech_duration * 2.0)
        try:
            return sherpa_onnx.VoiceActivityDetector(
                config,
                buffer_size_in_seconds=buffer_size_in_seconds,
            )
        except TypeError:
            return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds)


class SherpaASRWorker:
    def __init__(
        self,
        *,
        model_dir: str,
        sample_rate: int = 16000,
        threads: int = 1,
        vad_model_path: str = "",
        vad_threshold: float = 0.3,
        vad_min_silence_duration: float = 0.35,
        vad_min_speech_duration: float = 0.25,
        vad_max_speech_duration: float = 20.0,
        sense_voice_language: str = "auto",
        sense_voice_use_itn: bool = True,
    ):
        self.sample_rate = max(8000, int(sample_rate or 16000))
        self.threads = int(threads or 1)
        self.asr_strategy = ASR_STRATEGY_SENSEVOICE_VAD_OFFLINE
        self.vad_threshold = float(vad_threshold or 0.3)
        self.vad_min_silence_duration = max(0.0, float(vad_min_silence_duration or 0.35))
        self.vad_min_speech_duration = max(0.0, float(vad_min_speech_duration or 0.25))
        self.vad_max_speech_duration = max(0.1, float(vad_max_speech_duration or 20.0))
        self.sense_voice_language = str(sense_voice_language or "auto").strip() or "auto"
        self.sense_voice_use_itn = bool(sense_voice_use_itn)
        self.enabled = False
        self.reason = ""
        self.provider = ""
        self.model_format = ""
        self._recognizer = None
        self._vad = None
        self._adapter: SherpaRuntimeAdapter | None = None
        self._probe: SherpaRuntimeProbeResult | None = None
        self._stream_started_wall_ts = 0.0
        self._pcm_remainder = b""
        self._vad_buffer: list[float] = []
        self._vad_window_size = _vad_window_size_for_sample_rate(self.sample_rate)
        self._vad_processed_samples = 0
        self._vad_speech_active = False
        self._pending_events: list[ASRDebugEvent] = []
        self._model_dir = Path(model_dir).expanduser()
        self._vad_model_path = str(vad_model_path or "").strip()
        self._try_init()

    def drain_events(self) -> list[ASRDebugEvent]:
        events = list(self._pending_events)
        self._pending_events = []
        return events

    def _emit_event(
        self,
        kind: str,
        message: str,
        *,
        level: str = "info",
        wall_ts: float = 0.0,
        ts_start: float = 0.0,
        ts_end: float = 0.0,
        text: str = "",
    ):
        event = ASRDebugEvent(
            kind=kind,
            message=message,
            level=level,
            wall_ts=wall_ts,
            ts_start=ts_start,
            ts_end=ts_end,
            text=text,
        )
        self._pending_events.append(event)
        if level == "warning":
            logger.warning(f"[bili_watcher] {message}")

    def _reset_sensevoice_tracking_state(self):
        self._pcm_remainder = b""
        self._stream_started_wall_ts = 0.0
        self._vad_buffer = []
        self._vad_processed_samples = 0
        self._vad_speech_active = False
        if self._adapter is not None and self._probe is not None and self.enabled:
            try:
                self._vad = self._adapter.create_vad(self._probe)
            except Exception as e:
                self.reason = f"vad reset failed: {e}"
                self.enabled = False
                self._vad = None

    def restart_stream(self, *, flush_partial: bool = True, reason: str = "") -> list[ASRSegment]:
        if not self.enabled or self._recognizer is None:
            return []
        emitted = self._collect_ready_vad_segments() if flush_partial else []
        self._reset_sensevoice_tracking_state()
        if reason:
            logger.warning(f"[bili_watcher] ASR stream restarted: {reason}")
        return emitted

    def _try_init(self):
        adapter = SherpaRuntimeAdapter(
            model_dir=str(self._model_dir),
            sample_rate=self.sample_rate,
            threads=self.threads,
            vad_model_path=self._vad_model_path,
            vad_threshold=self.vad_threshold,
            vad_min_silence_duration=self.vad_min_silence_duration,
            vad_min_speech_duration=self.vad_min_speech_duration,
            vad_max_speech_duration=self.vad_max_speech_duration,
            sense_voice_language=self.sense_voice_language,
            sense_voice_use_itn=self.sense_voice_use_itn,
        )
        probe = adapter.probe()
        if not probe.enabled:
            self.reason = probe.reason
            return
        try:
            self._recognizer = adapter.create_recognizer(probe)
            self._vad = adapter.create_vad(probe)
            self._adapter = adapter
            self._probe = probe
            self.provider = probe.provider
            self.model_format = probe.model_format
            self.threads = probe.threads
            self.enabled = True
        except Exception as e:
            self.reason = f"init failed: {e}"

    def feed_pcm(self, pcm_chunk: bytes) -> list[ASRSegment]:
        if not self.enabled or self._recognizer is None or self._vad is None:
            return []
        if not pcm_chunk:
            return []
        return self._feed_pcm_sensevoice(pcm_chunk)

    def _pcm_bytes_to_floats(self, pcm_chunk: bytes) -> tuple[list[float], int]:
        pcm_chunk = self._pcm_remainder + pcm_chunk
        if len(pcm_chunk) % 2 == 1:
            self._pcm_remainder = pcm_chunk[-1:]
            pcm_chunk = pcm_chunk[:-1]
        else:
            self._pcm_remainder = b""

        n_samples = len(pcm_chunk) // 2
        if n_samples <= 0:
            return [], 0
        ints = array.array("h")
        ints.frombytes(pcm_chunk[: n_samples * 2])
        return [x / 32768.0 for x in ints], n_samples

    def _feed_pcm_sensevoice(self, pcm_chunk: bytes) -> list[ASRSegment]:
        floats, n_samples = self._pcm_bytes_to_floats(pcm_chunk)
        if n_samples <= 0:
            return []
        chunk_duration = n_samples / self.sample_rate
        if self._stream_started_wall_ts <= 0:
            self._stream_started_wall_ts = max(0.0, time.time() - chunk_duration)
        self._vad_buffer.extend(floats)

        emitted: list[ASRSegment] = []
        while len(self._vad_buffer) >= self._vad_window_size:
            window = self._vad_buffer[: self._vad_window_size]
            del self._vad_buffer[: self._vad_window_size]
            before_samples = self._vad_processed_samples
            prev_speech = self._vad_speech_active
            self._vad.accept_waveform(window)
            self._vad_processed_samples += len(window)
            current_speech = prev_speech
            try:
                current_speech = bool(self._vad.is_speech_detected())
            except Exception:
                current_speech = prev_speech
            if current_speech and not prev_speech:
                ts_start = before_samples / self.sample_rate
                wall_ts = self._stream_started_wall_ts + ts_start
                self._emit_event(
                    "vad_start",
                    f"VAD speech started at audio={ts_start:.2f}s wall={wall_ts:.2f}",
                    wall_ts=wall_ts,
                    ts_start=ts_start,
                    ts_end=ts_start,
                )
            self._vad_speech_active = current_speech
            emitted.extend(self._collect_ready_vad_segments())
        return emitted

    def _collect_ready_vad_segments(self) -> list[ASRSegment]:
        if self._vad is None:
            return []
        emitted: list[ASRSegment] = []
        while True:
            try:
                empty = bool(self._vad.empty())
            except Exception:
                empty = True
            if empty:
                break
            segment = self._vad.front
            try:
                self._vad.pop()
            except Exception:
                break
            self._vad_speech_active = False
            emitted.extend(self._decode_vad_segment(segment))
        return emitted

    def _decode_vad_segment(self, vad_segment: object) -> list[ASRSegment]:
        samples = getattr(vad_segment, "samples", None)
        if samples is None:
            return []
        if hasattr(samples, "tolist"):
            sample_list = list(samples.tolist())
        else:
            sample_list = list(samples)
        if not sample_list:
            return []

        start_sample = int(getattr(vad_segment, "start", max(0, self._vad_processed_samples - len(sample_list))))
        ts_start = max(0.0, start_sample / self.sample_rate)
        ts_end = ts_start + (len(sample_list) / self.sample_rate)
        wall_ts_start = self._stream_started_wall_ts + ts_start if self._stream_started_wall_ts > 0 else 0.0
        wall_ts_end = self._stream_started_wall_ts + ts_end if self._stream_started_wall_ts > 0 else 0.0
        self._emit_event(
            "vad_segment",
            f"VAD segment complete audio={ts_start:.2f}-{ts_end:.2f}s duration={ts_end - ts_start:.2f}s",
            wall_ts=wall_ts_end,
            ts_start=ts_start,
            ts_end=ts_end,
        )
        max_chunk_samples = self._sensevoice_safe_decode_max_samples()
        chunk_count = max(1, math.ceil(len(sample_list) / max_chunk_samples))
        if chunk_count > 1:
            self._emit_event(
                "vad_segment_split",
                (
                    "VAD segment exceeds SenseVoice safe decode window; "
                    f"split into {chunk_count} chunk(s) of <= {max_chunk_samples / self.sample_rate:.2f}s"
                ),
                wall_ts=wall_ts_end,
                ts_start=ts_start,
                ts_end=ts_end,
            )

        emitted: list[ASRSegment] = []
        for chunk_idx in range(chunk_count):
            offset = chunk_idx * max_chunk_samples
            chunk_samples = sample_list[offset : offset + max_chunk_samples]
            if not chunk_samples:
                continue

            chunk_ts_start = ts_start + (offset / self.sample_rate)
            chunk_ts_end = chunk_ts_start + (len(chunk_samples) / self.sample_rate)
            chunk_wall_ts_start = (
                self._stream_started_wall_ts + chunk_ts_start if self._stream_started_wall_ts > 0 else 0.0
            )
            chunk_wall_ts_end = (
                self._stream_started_wall_ts + chunk_ts_end if self._stream_started_wall_ts > 0 else 0.0
            )
            chunk_label = f"chunk {chunk_idx + 1}/{chunk_count}" if chunk_count > 1 else "segment"

            try:
                text = self._decode_sensevoice_samples(chunk_samples)
            except Exception as e:
                self._emit_event(
                    "asr_failure",
                    f"SenseVoice recognition failed on {chunk_label}: {e}",
                    level="warning",
                    wall_ts=chunk_wall_ts_end,
                    ts_start=chunk_ts_start,
                    ts_end=chunk_ts_end,
                )
                continue

            normalized = _normalize_sentence_text(text)
            if not normalized:
                self._emit_event(
                    "asr_failure",
                    (
                        "SenseVoice recognition returned empty result"
                        if chunk_count == 1
                        else f"SenseVoice recognition returned empty result on {chunk_label}"
                    ),
                    level="warning",
                    wall_ts=chunk_wall_ts_end,
                    ts_start=chunk_ts_start,
                    ts_end=chunk_ts_end,
                )
                continue

            segment = ASRSegment(
                text=_ensure_sentence_terminal(normalized),
                ts_start=chunk_ts_start,
                ts_end=chunk_ts_end,
                conf=0.6,
                wall_ts_start=chunk_wall_ts_start,
                wall_ts_end=chunk_wall_ts_end,
            )
            emitted.append(segment)
            self._emit_event(
                "asr_success",
                (
                    f"SenseVoice recognition success: {normalized}"
                    if chunk_count == 1
                    else f"SenseVoice recognition success on {chunk_label}: {normalized}"
                ),
                wall_ts=chunk_wall_ts_end,
                ts_start=chunk_ts_start,
                ts_end=chunk_ts_end,
                text=normalized,
            )
        return emitted

    def _sensevoice_safe_decode_max_seconds(self) -> float:
        configured = max(1.0, float(self.vad_max_speech_duration or 0.0))
        return max(1.0, min(configured, DEFAULT_SENSEVOICE_SAFE_DECODE_SECONDS))

    def _sensevoice_safe_decode_max_samples(self) -> int:
        return max(1, int(self._sensevoice_safe_decode_max_seconds() * self.sample_rate))

    def _decode_sensevoice_samples(self, samples: list[float]) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, samples)
        self._recognizer.decode_stream(stream)
        return _extract_offline_stream_result(self._recognizer, stream)

    def _flush_active_vad_segment(self) -> list[ASRSegment]:
        if self._vad is None:
            return []
        if self._vad_buffer:
            pad_size = max(0, self._vad_window_size - len(self._vad_buffer))
            window = self._vad_buffer + ([0.0] * pad_size)
            self._vad_buffer = []
            self._vad.accept_waveform(window)
            self._vad_processed_samples += len(window)
            try:
                self._vad_speech_active = bool(self._vad.is_speech_detected())
            except Exception:
                pass
        if not self._vad_speech_active:
            return self._collect_ready_vad_segments()

        silence_seconds = max(self.vad_min_silence_duration, self._vad_window_size / self.sample_rate)
        silence_windows = max(1, math.ceil((silence_seconds * self.sample_rate) / self._vad_window_size)) + 1
        silence = [0.0] * self._vad_window_size
        for _ in range(silence_windows):
            self._vad.accept_waveform(silence)
            self._vad_processed_samples += len(silence)
        self._vad_speech_active = False
        return self._collect_ready_vad_segments()

    def flush(self) -> list[ASRSegment]:
        self._pcm_remainder = b""
        if not self.enabled:
            return []
        emitted = self._collect_ready_vad_segments()
        emitted.extend(self._flush_active_vad_segment())
        return emitted

    def status_text(self) -> str:
        if self.enabled:
            return (
                f"enabled(strategy={self.asr_strategy}, provider={self.provider}, "
                f"format={self.model_format}, threads={self.threads})"
            )
        return f"disabled(strategy={self.asr_strategy}, reason={self.reason})"


def _normalize_sentence_text(text: str) -> str:
    text = re.sub(r"\s+", "", str(text or "").strip())
    return text


def _ensure_sentence_terminal(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    if cleaned[-1] in SENTENCE_END_CHARS:
        return cleaned
    return f"{cleaned}。"


def _extract_recognizer_text(result: object) -> str:
    if result is None:
        return ""
    text = getattr(result, "text", None)
    if text is not None:
        return str(text or "").strip()
    return str(result or "").strip()


def _extract_offline_stream_result(recognizer: object, stream: object) -> str:
    if recognizer is not None:
        get_result = getattr(recognizer, "get_result", None)
        if callable(get_result):
            try:
                return _extract_recognizer_text(get_result(stream))
            except Exception:
                pass
    return _extract_recognizer_text(getattr(stream, "result", None))


def sherpa_wheel_has_rknn_support(sherpa_onnx_module) -> bool:
    base = Path(getattr(sherpa_onnx_module, "__file__", "")).resolve().parent
    matches = sorted((base / "lib").glob("_sherpa_onnx*.so"))
    if not matches:
        return False
    so_path = matches[0]
    try:
        proc = subprocess.run(
            ["ldd", str(so_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return False
    output = proc.stdout + "\n" + proc.stderr
    return "librknnrt.so" in output


def _resolve_onnx_component(model_dir: Path, stem: str) -> Path | None:
    exact = model_dir / f"{stem}.onnx"
    if exact.exists():
        return exact

    patterns = [
        f"{stem}*.int8.onnx",
        f"{stem}*.onnx",
    ]
    for pattern in patterns:
        items = sorted(model_dir.glob(pattern))
        if items:
            return items[0]
    return None


def detect_sherpa_model(model_dir: str | Path) -> SherpaModelSpec:
    path = Path(model_dir).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"ASR model dir missing: {path}")

    candidate_dirs = [path]
    candidate_dirs.extend(sorted(child for child in path.iterdir() if child.is_dir()))

    saw_tokens = False
    for candidate in candidate_dirs:
        tokens = candidate / "tokens.txt"
        if not tokens.exists():
            continue
        saw_tokens = True
        spec = _detect_sense_voice_model(candidate, tokens)
        if spec is not None:
            return spec

    if not saw_tokens:
        raise FileNotFoundError(f"ASR tokens missing: {path / 'tokens.txt'}")

    raise FileNotFoundError(
        "ASR model files missing for sensevoice_vad_offline: expect model.rknn + tokens.txt, "
        "or model*.onnx + tokens.txt"
    )


def _detect_sense_voice_model(model_dir: Path, tokens: Path) -> SherpaModelSpec | None:
    model_rknn = model_dir / "model.rknn"
    if model_rknn.exists():
        return SherpaModelSpec(
            provider="rknn",
            tokens=tokens,
            model_dir=model_dir,
            model_format="rknn",
            architecture="sense_voice",
            model=model_rknn,
        )

    model_onnx = _resolve_onnx_component(model_dir, "model")
    if model_onnx:
        return SherpaModelSpec(
            provider="cpu",
            tokens=tokens,
            model_dir=model_dir,
            model_format="onnx",
            architecture="sense_voice",
            model=model_onnx,
        )
    return None


def normalize_sherpa_threads(provider: str, threads: int) -> int:
    t = int(threads or 1)
    if provider == "rknn":
        return t if t in RKNN_THREAD_VALUES else 1
    return max(1, t)


def normalize_sense_voice_language(raw_language: object) -> str:
    value = str(raw_language or "auto").strip().lower()
    if not value:
        return "auto"
    aliases = {
        "zh-cn": "zh",
        "zh-hans": "zh",
        "zh-hant": "yue",
        "cantonese": "yue",
        "english": "en",
        "japanese": "ja",
        "korean": "ko",
    }
    return aliases.get(value, value)


def _vad_window_size_for_sample_rate(sample_rate: int) -> int:
    return 512 if int(sample_rate or 16000) >= 16000 else 256


def build_asr_worker_or_none(
    *,
    model_dir: str,
    sample_rate: int,
    threads: int,
    vad_model_path: str = "",
    vad_threshold: float = 0.3,
    vad_min_silence_duration: float = 0.35,
    vad_min_speech_duration: float = 0.25,
    vad_max_speech_duration: float = 20.0,
    sense_voice_language: str = "auto",
    sense_voice_use_itn: bool = True,
) -> SherpaASRWorker | None:
    worker = SherpaASRWorker(
        model_dir=model_dir,
        sample_rate=sample_rate,
        threads=threads,
        vad_model_path=vad_model_path,
        vad_threshold=vad_threshold,
        vad_min_silence_duration=vad_min_silence_duration,
        vad_min_speech_duration=vad_min_speech_duration,
        vad_max_speech_duration=vad_max_speech_duration,
        sense_voice_language=sense_voice_language,
        sense_voice_use_itn=sense_voice_use_itn,
    )
    if worker.enabled:
        logger.info(
            f"[bili_watcher] ASR enabled: strategy={worker.asr_strategy} "
            f"provider={worker.provider} format={worker.model_format}"
        )
        return worker
    logger.warning(f"[bili_watcher] ASR disabled: {worker.reason}")
    return None
