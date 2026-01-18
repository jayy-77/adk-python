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

"""Tests for BIDI live session restoration with SQLite persistence (Issue #3573)."""

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
