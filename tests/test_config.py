"""Tests for configuration loading, validation, and initialization."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from meeting_transcriber.config import (
  ConfigError,
  get_config_dir,
  init_config,
  load_config,
)


class TestGetConfigDir:
  """Tests for get_config_dir()."""

  def test_returns_path_object(self):
    config_dir = get_config_dir()
    assert isinstance(config_dir, Path)

  def test_default_uses_platformdirs(self):
    config_dir = get_config_dir()
    # platformdirs on macOS: ~/Library/Application Support/meeting-transcriber
    # on Linux: ~/.config/meeting-transcriber
    assert config_dir.name == "meeting-transcriber"

  def test_ends_with_meeting_transcriber(self):
    config_dir = get_config_dir()
    assert str(config_dir).endswith("meeting-transcriber")


class TestLoadConfig:
  """Tests for load_config()."""

  def test_loads_from_dotenv_file(self, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
      "OPENAI_API_KEY=sk-test-from-file\n"
      "ANTHROPIC_API_KEY=sk-ant-test-from-file\n"
      "AUDIO_DEVICE=MacBook Pro Microphone\n"
      "TRANSCRIPTION_LANGUAGE=en\n"
      "TRANSCRIPTION_ENGINE=whisper\n"
      "SUMMARY_MODEL=claude-sonnet-4-20250514\n"
    )
    config = load_config(config_dir=tmp_path)
    assert config["openai_api_key"] == "sk-test-from-file"
    assert config["anthropic_api_key"] == "sk-ant-test-from-file"
    assert config["audio_device"] == "MacBook Pro Microphone"
    assert config["transcription_language"] == "en"
    assert config["transcription_engine"] == "whisper"
    assert config["summary_model"] == "claude-sonnet-4-20250514"

  def test_env_var_overrides_dotenv(self, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-file\nANTHROPIC_API_KEY=sk-ant-from-file\n")
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}, clear=False):
      config = load_config(config_dir=tmp_path)
    assert config["openai_api_key"] == "sk-from-env"

  def test_defaults_applied(self, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
      "OPENAI_API_KEY=sk-test\n"
      "ANTHROPIC_API_KEY=sk-ant-test\n"
    )
    config = load_config(config_dir=tmp_path)
    assert config["transcription_language"] == "zh"
    assert config["transcription_engine"] == "openai"
    assert config["summary_model"] == "claude-sonnet-4-20250514"

  def test_missing_openai_key_raises_error(self, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-ant-test\n")
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
      load_config(config_dir=tmp_path)

  def test_missing_anthropic_key_raises_error(self, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-test\n")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
      load_config(config_dir=tmp_path)

  def test_missing_keys_error_includes_instructions(self, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("")
    with pytest.raises(ConfigError, match="mt init"):
      load_config(config_dir=tmp_path)

  def test_no_env_file_still_reads_env_vars(self, tmp_path):
    env_vars = {
      "OPENAI_API_KEY": "sk-from-env",
      "ANTHROPIC_API_KEY": "sk-ant-from-env",
    }
    with patch.dict(os.environ, env_vars, clear=False):
      config = load_config(config_dir=tmp_path)
    assert config["openai_api_key"] == "sk-from-env"
    assert config["anthropic_api_key"] == "sk-ant-from-env"

  def test_audio_device_defaults_to_none(self, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
      "OPENAI_API_KEY=sk-test\n"
      "ANTHROPIC_API_KEY=sk-ant-test\n"
    )
    config = load_config(config_dir=tmp_path)
    assert config["audio_device"] is None


class TestInitConfig:
  """Tests for init_config()."""

  def test_creates_config_dir(self, tmp_path):
    config_dir = tmp_path / "meeting-transcriber"
    init_config(config_dir=config_dir)
    assert config_dir.is_dir()

  def test_creates_env_template(self, tmp_path):
    config_dir = tmp_path / "meeting-transcriber"
    init_config(config_dir=config_dir)
    env_file = config_dir / ".env"
    assert env_file.exists()

  def test_env_template_contains_keys(self, tmp_path):
    config_dir = tmp_path / "meeting-transcriber"
    init_config(config_dir=config_dir)
    content = (config_dir / ".env").read_text()
    assert "OPENAI_API_KEY" in content
    assert "ANTHROPIC_API_KEY" in content
    assert "AUDIO_DEVICE" in content
    assert "TRANSCRIPTION_LANGUAGE" in content
    assert "TRANSCRIPTION_ENGINE" in content
    assert "SUMMARY_MODEL" in content

  def test_does_not_overwrite_existing_env(self, tmp_path):
    config_dir = tmp_path / "meeting-transcriber"
    config_dir.mkdir(parents=True)
    env_file = config_dir / ".env"
    env_file.write_text("OPENAI_API_KEY=my-precious-key\n")
    init_config(config_dir=config_dir)
    content = env_file.read_text()
    assert "my-precious-key" in content

  def test_returns_config_dir_path(self, tmp_path):
    config_dir = tmp_path / "meeting-transcriber"
    result = init_config(config_dir=config_dir)
    assert result == config_dir
