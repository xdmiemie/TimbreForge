import numpy as np
import soundfile as sf

from timbreforge.audio import RATE
from timbreforge.covers import mix_audio


def test_mix_keeps_silence_positions_and_stereo(tmp_path):
    stem = np.full((RATE * 4, 2), [0.025, 0.05], dtype=np.float32)
    voice = np.zeros(len(stem), dtype=np.float32)
    voice[RATE : 2 * RATE] = 0.125
    voice[3 * RATE :] = 0.1
    sf.write(tmp_path / "伴奏.wav", stem, RATE, subtype="FLOAT")
    sf.write(tmp_path / "人声.wav", voice, RATE, subtype="FLOAT")
    report = mix_audio(tmp_path / "伴奏.wav", tmp_path / "人声.wav", tmp_path / "试听.wav", len(stem), 0, 0)
    result, rate = sf.read(tmp_path / "试听.wav")
    assert rate == RATE
    assert report["samples"] == len(stem)
    np.testing.assert_allclose(result, stem + voice[:, None], rtol=0, atol=2e-7)


def test_transpose_keeps_timeline_and_shifts_backing(tmp_path):
    x = np.arange(2 * RATE) / RATE
    sine = (0.2 * np.sin(2 * np.pi * 440 * x)).astype(np.float32)
    sf.write(tmp_path / "backing.wav", np.column_stack((sine, sine * 0.5)), RATE, subtype="FLOAT")
    sf.write(tmp_path / "vocals.wav", np.zeros_like(sine), RATE, subtype="FLOAT")
    destination = tmp_path / "result.wav"
    mix_audio(tmp_path / "backing.wav", tmp_path / "vocals.wav", destination, len(x), 0, 0, 12)
    result, rate = sf.read(destination)
    assert rate == RATE and result.shape == (len(x), 2)
    sample = result[RATE // 2:3 * RATE // 2, 0]
    peak_frequency = np.fft.rfftfreq(len(sample), 1 / rate)[np.argmax(np.abs(np.fft.rfft(sample)))]
    assert abs(peak_frequency - 880) <= 2
