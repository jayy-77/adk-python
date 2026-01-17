# Copyright 2025 Google LLC
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

"""Unit tests for tool iteration limit to prevent infinite loops (Issue #4179)."""

import pytest
from google.adk.agents.invocation_context import ToolIterationsLimitExceededError
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.genai import types

from ... import testing_utils


@pytest.mark.asyncio
async def test_default_max_tool_iterations_value():
  """Test that the default max_tool_iterations is 50."""
  run_config = RunConfig()
  assert run_config.max_tool_iterations == 50


@pytest.mark.asyncio
async def test_increment_tool_iteration_count():
  """Test that tool iteration counter increments and enforces limit."""
  agent = Agent(name='test_agent')
  run_config = RunConfig(max_tool_iterations=3)
  
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test', run_config=run_config
  )
  
  # Should not raise for first 3 increments
  invocation_context.increment_tool_iteration_count()  # 1
  invocation_context.increment_tool_iteration_count()  # 2
  invocation_context.increment_tool_iteration_count()  # 3
  
  # 4th increment should raise ToolIterationsLimitExceededError
  with pytest.raises(ToolIterationsLimitExceededError) as exc_info:
    invocation_context.increment_tool_iteration_count()  # 4 - exceeds limit
  
  assert 'Max number of tool iterations limit of' in str(exc_info.value)
  assert '3' in str(exc_info.value)


@pytest.mark.asyncio
async def test_reset_tool_iteration_count():
  """Test that tool iteration counter resets properly."""
  agent = Agent(name='test_agent')
  run_config = RunConfig(max_tool_iterations=2)
  
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test', run_config=run_config
  )
  
  # First cycle: increment twice
  invocation_context.increment_tool_iteration_count()  # 1
  invocation_context.increment_tool_iteration_count()  # 2
  
  # Reset the counter
  invocation_context.reset_tool_iteration_count()
  
  # Should not raise after reset - can increment again
  invocation_context.increment_tool_iteration_count()  # 1 (reset)
  invocation_context.increment_tool_iteration_count()  # 2 (reset)
  
  # 3rd increment should raise
  with pytest.raises(ToolIterationsLimitExceededError):
    invocation_context.increment_tool_iteration_count()  # 3 - exceeds limit


@pytest.mark.asyncio
async def test_max_tool_iterations_disabled():
  """Test that setting max_tool_iterations to 0 disables enforcement."""
  agent = Agent(name='test_agent')
  run_config = RunConfig(max_tool_iterations=0)
  
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test', run_config=run_config
  )
  
  # Should not raise even after many increments when limit is disabled
  for _ in range(100):
    invocation_context.increment_tool_iteration_count()
  
  # No exception raised - test passes


@pytest.mark.asyncio
async def test_max_tool_iterations_validator():
  """Test that RunConfig validator warns about disabled limit."""
  import logging
  import warnings
  
  # Setting to 0 should trigger a warning
  with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    run_config = RunConfig(max_tool_iterations=0)
    assert run_config.max_tool_iterations == 0
  
  # Setting to positive value should not raise
  run_config = RunConfig(max_tool_iterations=50)
  assert run_config.max_tool_iterations == 50
