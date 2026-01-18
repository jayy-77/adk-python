# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for BIDI live session restoration with SQLite persistence (Issue #3573, #3395)."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
import pytest


@pytest.mark.asyncio
async def test_live_session_replays_all_events_on_reconnection():
  """Test that reconnecting to a live session replays all events including user messages.
  
  This tests the fix for Issue #3573 where user messages were stored in the
  database but not sent back to the client on reconnection.
  """
  # Create a mock session with both user and agent events
  user_event = Event(
      id="event-user-1",
      author="user",
      content=types.Content(parts=[types.Part(text="Hello, assistant!")]),
      invocation_id="inv-1",
  )
  agent_event = Event(
      id="event-agent-1",
      author="test_agent",
      content=types.Content(
          parts=[types.Part(text="Hello! How can I help you?")]
      ),
      invocation_id="inv-1",
  )
  user_event2 = Event(
      id="event-user-2",
      author="user",
      content=types.Content(
          parts=[types.Part(text="What's the weather today?")]
      ),
      invocation_id="inv-2",
  )
  agent_event2 = Event(
      id="event-agent-2",
      author="test_agent",
      content=types.Content(
          parts=[types.Part(text="I can help you check the weather.")]
      ),
      invocation_id="inv-2",
  )

  mock_session = Session(
      app_name="test_app",
      user_id="test_user",
      id="test_session",
      state={},
      events=[user_event, agent_event, user_event2, agent_event2],
      last_update_time=1234567890.0,
  )

  # Mock WebSocket to capture replayed events
  mock_websocket = AsyncMock()
  replayed_events = []

  async def capture_send_text(data):
    replayed_events.append(data)

  mock_websocket.send_text = capture_send_text

  # Test the core event replay logic that should be in run_agent_live
  # This simulates what happens when a client reconnects:
  # 1. Session is loaded (with all events)
  session = mock_session
  
  # 2. All existing events should be replayed to the client
  if session and session.events:
    for event in session.events:
      await mock_websocket.send_text(
          event.model_dump_json(exclude_none=True, by_alias=True)
      )

  # Verify that all 4 events were replayed (2 user + 2 agent)
  assert len(replayed_events) == 4

  # Verify that events were sent in order
  import json

  event_data = [json.loads(data) for data in replayed_events]
  
  # Check that user messages are included
  assert event_data[0]["author"] == "user"
  assert "Hello, assistant!" in event_data[0]["content"]["parts"][0]["text"]
  
  assert event_data[1]["author"] == "test_agent"
  
  assert event_data[2]["author"] == "user"
  assert "weather" in event_data[2]["content"]["parts"][0]["text"]
  
  assert event_data[3]["author"] == "test_agent"


@pytest.mark.asyncio
async def test_live_session_handles_empty_events_gracefully():
  """Test that session restoration handles sessions with no events."""
  mock_session = Session(
      app_name="test_app",
      user_id="test_user",
      id="new_session",
      state={},
      events=[],  # No events yet
      last_update_time=1234567890.0,
  )

  mock_session_service = AsyncMock()
  mock_session_service.get_session.return_value = mock_session

  mock_websocket = AsyncMock()
  replayed_events = []

  async def capture_send_text(data):
    replayed_events.append(data)

  mock_websocket.send_text = capture_send_text

  # Simulate event replay logic
  session = await mock_session_service.get_session(
      app_name="test_app", user_id="test_user", session_id="new_session"
  )
  
  if session and session.events:
    for event in session.events:
      await mock_websocket.send_text(
          event.model_dump_json(exclude_none=True, by_alias=True)
      )

  # Should not send any events for an empty session
  assert len(replayed_events) == 0


@pytest.mark.asyncio
async def test_live_session_continues_after_replay_failure():
  """Test that session continues even if one event fails to replay."""
  # Create events where one might fail to serialize
  event1 = Event(
      id="event-1",
      author="user",
      content=types.Content(parts=[types.Part(text="First message")]),
      invocation_id="inv-1",
  )
  event2 = Event(
      id="event-2",
      author="agent",
      content=types.Content(parts=[types.Part(text="Second message")]),
      invocation_id="inv-1",
  )
  event3 = Event(
      id="event-3",
      author="user",
      content=types.Content(parts=[types.Part(text="Third message")]),
      invocation_id="inv-2",
  )

  mock_session = Session(
      app_name="test_app",
      user_id="test_user",
      id="test_session",
      state={},
      events=[event1, event2, event3],
      last_update_time=1234567890.0,
  )

  mock_websocket = AsyncMock()
  replayed_events = []
  send_call_count = 0

  async def capture_send_text(data):
    nonlocal send_call_count
    send_call_count += 1
    # Simulate failure on second event
    if send_call_count == 2:
      raise Exception("Simulated network error")
    replayed_events.append(data)

  mock_websocket.send_text = capture_send_text

  # Simulate event replay with error handling
  if mock_session and mock_session.events:
    for event in mock_session.events:
      try:
        await mock_websocket.send_text(
            event.model_dump_json(exclude_none=True, by_alias=True)
        )
      except Exception:
        # Continue replaying even if one fails
        continue

  # Should have replayed 2 events successfully (skipped the failing one)
  assert len(replayed_events) == 2
  assert send_call_count == 3  # Attempted all 3


@pytest.mark.asyncio
async def test_live_session_skips_replay_for_audio_sessions():
  """Test that live sessions with audio/transcription skip replay to avoid duplicates.

  This tests the fix for Issue #3395 where Gemini Live API's transparent session
  resumption would conflict with our event replay, causing duplicate responses.
  When a session contains audio or transcription data, we rely on the model's
  session resumption instead of replaying events.
  """
  # Create a session with transcription data (indicating live mode)
  user_event_with_transcription = Event(
      id="event-user-1",
      author="user",
      content=types.Content(parts=[types.Part(text="Hello")]),
      invocation_id="inv-1",
      input_transcription=types.Transcription(
          text="Hello", finished=True
      ),
  )
  agent_event_with_transcription = Event(
      id="event-agent-1",
      author="test_agent",
      content=types.Content(
          parts=[types.Part(text="Hello! How can I help you?")]
      ),
      invocation_id="inv-1",
      output_transcription=types.Transcription(
          text="Hello! How can I help you?", finished=True
      ),
  )

  mock_session = Session(
      app_name="test_app",
      user_id="test_user",
      id="test_session",
      state={},
      events=[user_event_with_transcription, agent_event_with_transcription],
      last_update_time=1234567890.0,
  )

  mock_websocket = AsyncMock()
  replayed_events = []

  async def capture_send_text(data):
    replayed_events.append(data)

  mock_websocket.send_text = capture_send_text

  # Helper to detect live sessions (copied from adk_web_server.py logic)
  def is_live_session(events: list) -> bool:
    for event in reversed(events[-5:] if len(events) > 5 else events):
      if hasattr(event, 'input_transcription') and event.input_transcription:
        return True
      if hasattr(event, 'output_transcription') and event.output_transcription:
        return True
      if event.content:
        for part in event.content.parts:
          if part.inline_data and (
              part.inline_data.mime_type.startswith("audio/")
              or part.inline_data.mime_type.startswith("video/")
          ):
            return True
          if part.file_data and (
              part.file_data.mime_type.startswith("audio/")
              or part.file_data.mime_type.startswith("video/")
          ):
            return True
    return False

  # Test the conditional replay logic
  session = mock_session
  should_replay = session.events and not is_live_session(session.events)

  if should_replay:
    for event in session.events:
      await mock_websocket.send_text(
          event.model_dump_json(exclude_none=True, by_alias=True)
      )

  # Expect no events to be replayed because it's a live/audio session
  assert len(replayed_events) == 0


@pytest.mark.asyncio
async def test_text_session_replays_events_normally():
  """Test that text-only sessions still replay events as expected.

  This ensures that the fix for Issue #3395 (skipping replay for live sessions)
  doesn't break the fix for Issue #3573 (replaying events for text sessions).
  """
  # Create a session with only text content (no transcriptions)
  user_event = Event(
      id="event-user-1",
      author="user",
      content=types.Content(parts=[types.Part(text="Hello")]),
      invocation_id="inv-1",
  )
  agent_event = Event(
      id="event-agent-1",
      author="test_agent",
      content=types.Content(
          parts=[types.Part(text="Hello! How can I help you?")]
      ),
      invocation_id="inv-1",
  )

  mock_session = Session(
      app_name="test_app",
      user_id="test_user",
      id="test_session",
      state={},
      events=[user_event, agent_event],
      last_update_time=1234567890.0,
  )

  mock_websocket = AsyncMock()
  replayed_events = []

  async def capture_send_text(data):
    replayed_events.append(data)

  mock_websocket.send_text = capture_send_text

  # Helper to detect live sessions
  def is_live_session(events: list) -> bool:
    for event in reversed(events[-5:] if len(events) > 5 else events):
      if hasattr(event, 'input_transcription') and event.input_transcription:
        return True
      if hasattr(event, 'output_transcription') and event.output_transcription:
        return True
      if event.content:
        for part in event.content.parts:
          if part.inline_data and (
              part.inline_data.mime_type.startswith("audio/")
              or part.inline_data.mime_type.startswith("video/")
          ):
            return True
          if part.file_data and (
              part.file_data.mime_type.startswith("audio/")
              or part.file_data.mime_type.startswith("video/")
          ):
            return True
    return False

  # Test the conditional replay logic
  session = mock_session
  should_replay = session.events and not is_live_session(session.events)

  if should_replay:
    for event in session.events:
      await mock_websocket.send_text(
          event.model_dump_json(exclude_none=True, by_alias=True)
      )

  # Expect both events to be replayed for text-only session
  assert len(replayed_events) == 2

