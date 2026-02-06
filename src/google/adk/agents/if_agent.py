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

"""Conditional branching agent implementation (IfAgent/ConditionalAgent)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any
from typing import AsyncGenerator
from typing import ClassVar
from typing import Dict
from typing import Type

from typing_extensions import override

from ..agents.base_agent import BaseAgent
from ..agents.base_agent import BaseAgentState
from ..agents.base_agent_config import BaseAgentConfig
from ..agents.config_agent_utils import resolve_code_reference
from ..agents.if_agent_config import IfAgentConfig
from ..agents.invocation_context import InvocationContext
from ..events.event import Event
from ..utils.context_utils import Aclosing
from ..utils.feature_decorator import experimental

logger = logging.getLogger('google_adk.' + __name__)


@experimental
class IfAgentState(BaseAgentState):
  """State for IfAgent."""

  condition_result: bool = False
  """The result of the condition evaluation."""

  chosen_agent: str = ''
  """The name of the chosen sub-agent."""


@experimental
class IfAgent(BaseAgent):
  """Conditional branching agent for runtime workflow decisions.

  IfAgent evaluates a condition function at runtime and delegates execution
  to one of two sub-agents (if_true or if_false) based on the result.

  The condition function receives the InvocationContext and must return a
  boolean. This enables dynamic routing based on:
  - Session state and history
  - User message content
  - Custom metadata
  - External state or API calls

  Example Python usage:
    def is_premium_user(ctx: InvocationContext) -> bool:
      return ctx.session.custom_metadata.get('tier') == 'premium'

    if_agent = IfAgent(
      name='user_router',
      description='Routes to specialized agents based on user tier',
      condition=is_premium_user,
      sub_agents=[premium_agent, standard_agent]
    )

  Example YAML usage:
    name: user_router
    agent_class: IfAgent
    description: Routes to specialized agents based on user tier
    condition:
      module: my_module.conditions
      function: is_premium_user
    sub_agents:
      - agent_ref: premium_agent
      - agent_ref: standard_agent
  """

  config_type: ClassVar[Type[BaseAgentConfig]] = IfAgentConfig
  """The config type for this agent."""

  def __init__(
      self,
      *,
      condition: Callable[[InvocationContext], bool] | None = None,
      **kwargs: Any,
  ):
    """Initializes an IfAgent.

    Args:
      condition: A callable that takes InvocationContext and returns bool.
        If True, executes the first sub-agent (if_true).
        If False, executes the second sub-agent (if_false).
        If None, defaults to always True.
      **kwargs: Additional arguments passed to BaseAgent.

    Raises:
      ValueError: If sub_agents is not provided or doesn't have exactly 2
        agents.
    """
    super().__init__(**kwargs)
    self._condition = condition or (lambda ctx: True)

    if not self.sub_agents or len(self.sub_agents) != 2:
      raise ValueError(
          f'IfAgent {self.name} requires exactly 2 sub-agents '
          f'(if_true and if_false), but got {len(self.sub_agents) if self.sub_agents else 0}.'
      )

  @override
  @classmethod
  @experimental
  def _parse_config(
      cls: Type[IfAgent],
      config: IfAgentConfig,
      config_abs_path: str,
      kwargs: Dict[str, Any],
  ) -> Dict[str, Any]:
    """Parses IfAgent-specific configuration from YAML.

    Args:
      config: The IfAgentConfig parsed from YAML.
      config_abs_path: Absolute path to the config file.
      kwargs: Keyword arguments being built for agent instantiation.

    Returns:
      Updated kwargs dictionary with resolved condition.
    """
    if config.condition:
      kwargs['condition'] = resolve_code_reference(config.condition)
    return kwargs

  @property
  def if_true_agent(self) -> BaseAgent:
    """Returns the agent to execute when condition is True."""
    return self.sub_agents[0]

  @property
  def if_false_agent(self) -> BaseAgent:
    """Returns the agent to execute when condition is False."""
    return self.sub_agents[1]

  def _evaluate_condition(self, ctx: InvocationContext) -> bool:
    """Evaluates the condition function safely.

    Args:
      ctx: The invocation context.

    Returns:
      The result of the condition evaluation, or False if an error occurs.
    """
    try:
      result = self._condition(ctx)
      logger.debug(
          'IfAgent %s: condition evaluated to %s', self.name, result
      )
      return bool(result)
    except Exception as e:  # pylint: disable=broad-except
      logger.warning(
          'IfAgent %s: condition evaluation failed with error: %s. '
          'Defaulting to False.',
          self.name,
          e,
      )
      return False

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Runs the IfAgent by evaluating condition and delegating to chosen agent.

    Args:
      ctx: The invocation context.

    Yields:
      Events from the chosen sub-agent.
    """
    # Load or initialize state
    agent_state = self._load_agent_state(ctx, IfAgentState)

    if agent_state:
      # Resuming from a saved state
      condition_result = agent_state.condition_result
      chosen_agent_name = agent_state.chosen_agent
      logger.debug(
          'IfAgent %s: resuming with condition_result=%s, chosen_agent=%s',
          self.name,
          condition_result,
          chosen_agent_name,
      )
    else:
      # First invocation: evaluate condition
      condition_result = self._evaluate_condition(ctx)
      chosen_agent = (
          self.if_true_agent if condition_result else self.if_false_agent
      )
      chosen_agent_name = chosen_agent.name

      # Save state before executing sub-agent
      if ctx.is_resumable:
        agent_state = IfAgentState(
            condition_result=condition_result,
            chosen_agent=chosen_agent_name,
        )
        ctx.set_agent_state(self.name, agent_state=agent_state)
        yield self._create_agent_state_event(ctx)

    # Execute the chosen sub-agent
    chosen_agent = (
        self.if_true_agent if condition_result else self.if_false_agent
    )

    async with Aclosing(chosen_agent.run_async(ctx)) as agen:
      async for event in agen:
        yield event

    # Mark completion
    if ctx.is_resumable:
      ctx.set_agent_state(self.name, end_of_agent=True)
      yield self._create_agent_state_event(ctx)

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Runs the IfAgent in live mode with bidirectional streaming.

    Args:
      ctx: The invocation context.

    Yields:
      Events from the chosen sub-agent.
    """
    # Evaluate condition
    condition_result = self._evaluate_condition(ctx)
    chosen_agent = (
        self.if_true_agent if condition_result else self.if_false_agent
    )

    logger.debug(
        'IfAgent %s (live): condition=%s, executing agent=%s',
        self.name,
        condition_result,
        chosen_agent.name,
    )

    # Execute the chosen sub-agent in live mode
    async with Aclosing(chosen_agent.run_live(ctx)) as agen:
      async for event in agen:
        yield event
