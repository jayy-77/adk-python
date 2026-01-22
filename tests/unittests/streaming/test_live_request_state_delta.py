from __future__ import annotations

from google.adk.agents.live_request_queue import LiveRequest
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest
from pydantic import ValidationError

from .. import testing_utils


def test_live_request_state_delta_updates_session():
  response = LlmResponse(turn_complete=True)
  mock_model = testing_utils.MockModel.create([response])

  root_agent = Agent(
      name="root_agent",
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(root_agent=root_agent)
  live_request_queue = LiveRequestQueue()
  live_request_queue.send(
      LiveRequest(state_delta={"user_preference": "dark_mode"})
  )
  live_request_queue.close()

  _ = runner.run_live(live_request_queue)

  assert runner.session.state["user_preference"] == "dark_mode"


def test_live_request_state_delta_validation():
  LiveRequest(
      content=types.Content(
          role="user", parts=[types.Part.from_text(text="Hello")]
      ),
      state_delta={"theme": "dark"},
  )

  with pytest.raises(ValidationError):
    LiveRequest(
        state_delta={"theme": "dark"},
        blob=types.Blob(data=b"\x00\xFF", mime_type="audio/pcm"),
    )

  with pytest.raises(ValidationError):
    LiveRequest(
        close=True,
        content=types.Content(
            role="user", parts=[types.Part.from_text(text="Hello")]
        ),
    )

  with pytest.raises(ValidationError):
    LiveRequest()
