"""Tests for stereo (dual-channel) speaker separation."""

import numpy as np

from meeting_transcriber.server import (
  _make_on_final,
  _make_on_partial,
  _new_session,
)


class TestStereoSplitting:
  """Test that 3-channel PCM splitting produces correct mine/theirs channels."""

  def test_split_3ch_float32_to_mono_pcm(self):
    """Simulate the audio_callback splitting logic for 3-channel input.

    Aggregate Device layout: Ch1-2 = BlackHole (對方), Ch3 = mic (我方).
    """
    frames = 160
    bh_ch1 = np.full(frames, -0.3, dtype=np.float32)  # BlackHole Ch1 (對方)
    bh_ch2 = np.full(frames, -0.3, dtype=np.float32)  # BlackHole Ch2 (unused)
    mic_ch3 = np.full(frames, 0.5, dtype=np.float32)   # Mic (我方)
    indata = np.column_stack([bh_ch1, bh_ch2, mic_ch3])

    assert indata.shape == (frames, 3)

    # Split channels (same logic as audio_callback in stereo mode)
    theirs = indata[:, 0]  # BlackHole Ch1
    mine = indata[:, 2]    # Mic
    pcm_theirs = (theirs * 32767).astype(np.int16).tobytes()
    pcm_mine = (mine * 32767).astype(np.int16).tobytes()

    # Each mono channel should be frames * 2 bytes (int16)
    assert len(pcm_mine) == frames * 2
    assert len(pcm_theirs) == frames * 2

    # Verify values: mine ~16383, theirs ~-9830
    mine_values = np.frombuffer(pcm_mine, dtype=np.int16)
    theirs_values = np.frombuffer(pcm_theirs, dtype=np.int16)
    assert np.allclose(mine_values, 16383, atol=1)
    assert np.allclose(theirs_values, -9830, atol=1)

  def test_split_3ch_silence_produces_zero_pcm(self):
    """Silence on all channels should produce all-zero PCM."""
    frames = 100
    indata = np.zeros((frames, 3), dtype=np.float32)

    theirs = indata[:, 0]
    mine = indata[:, 2]
    pcm_theirs = (theirs * 32767).astype(np.int16).tobytes()
    pcm_mine = (mine * 32767).astype(np.int16).tobytes()

    mine_values = np.frombuffer(pcm_mine, dtype=np.int16)
    theirs_values = np.frombuffer(pcm_theirs, dtype=np.int16)
    assert np.all(mine_values == 0)
    assert np.all(theirs_values == 0)


class TestSpeakerTaggedMessages:
  """Test that transcript messages include speaker field when tagged."""

  def test_on_final_with_speaker_includes_speaker_in_message(self):
    session = _new_session()
    session["active"] = True
    session["start_time"] = 1000000.0

    on_final = _make_on_final(session, speaker="我方")
    on_final("你好，我是小明")

    # Check transcript_chunks has speaker prefix
    assert len(session["transcript_chunks"]) == 1
    assert session["transcript_chunks"][0].startswith("[我方]")

    # Check WS queue has speaker field in transcript message
    transcript_msgs = [m for m in session["_ws_queue"] if m["type"] == "transcript"]
    assert len(transcript_msgs) == 1
    assert transcript_msgs[0]["speaker"] == "我方"
    assert transcript_msgs[0]["text"] == "你好，我是小明"

  def test_on_final_without_speaker_has_no_speaker_field(self):
    session = _new_session()
    session["active"] = True
    session["start_time"] = 1000000.0

    on_final = _make_on_final(session)
    on_final("一般逐字稿")

    # Check transcript_chunks has no prefix
    assert session["transcript_chunks"][0] == "一般逐字稿"

    # Check WS queue has no speaker field
    transcript_msgs = [m for m in session["_ws_queue"] if m["type"] == "transcript"]
    assert "speaker" not in transcript_msgs[0]

  def test_on_partial_with_speaker_includes_speaker_in_message(self):
    session = _new_session()

    on_partial = _make_on_partial(session, speaker="對方")
    on_partial("正在說話...")

    assert len(session["_ws_queue"]) == 1
    msg = session["_ws_queue"][0]
    assert msg["type"] == "transcript_partial"
    assert msg["speaker"] == "對方"
    assert msg["text"] == "正在說話..."

  def test_on_partial_without_speaker_has_no_speaker_field(self):
    session = _new_session()

    on_partial = _make_on_partial(session)
    on_partial("正在說話...")

    msg = session["_ws_queue"][0]
    assert "speaker" not in msg


class TestCreateAppStereo:
  """Test that create_app accepts stereo parameter."""

  def test_create_app_with_stereo_false(self):
    from meeting_transcriber.server import create_app

    app = create_app(stereo=False)
    assert app.state.stereo is False

  def test_create_app_with_stereo_true(self):
    from meeting_transcriber.server import create_app

    app = create_app(stereo=True)
    assert app.state.stereo is True

  def test_create_app_default_stereo_is_false(self):
    from meeting_transcriber.server import create_app

    app = create_app()
    assert app.state.stereo is False
