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

"""Config definition for IfAgent."""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict
from pydantic import Field

from ..agents.base_agent_config import BaseAgentConfig
from ..agents.common_configs import CodeConfig
from ..utils.feature_decorator import experimental


@experimental
class IfAgentConfig(BaseAgentConfig):
  """The config for the YAML schema of an IfAgent.

  IfAgent enables conditional branching in agent workflows based on runtime
  conditions. It evaluates a condition function against the invocation context
  and delegates to one of two sub-agents based on the result.

  Example YAML:
    name: conditional_router
    agent_class: IfAgent
    description: Routes to specialized agents based on user intent
    condition:
      module: my_module.conditions
      function: is_technical_query
    sub_agents:
      - agent_ref: technical_agent
      - agent_ref: general_agent

  Example with inline condition arguments:
    name: priority_router
    agent_class: IfAgent
    description: Routes based on priority threshold
    condition:
      module: my_module.conditions
      function: check_priority_threshold
      args:
        threshold: 5
    sub_agents:
      - agent_ref: high_priority_agent
      - agent_ref: normal_priority_agent
  """

  model_config = ConfigDict(
      extra='forbid',
  )

  agent_class: str = Field(
      default='IfAgent',
      description='The value is used to uniquely identify the IfAgent class.',
  )

  condition: Optional[CodeConfig] = Field(
      default=None,
      description=(
          'Optional. IfAgent.condition. A CodeConfig that references a callable'
          ' (function, lambda, or class with __call__) that takes an'
          ' InvocationContext and returns a boolean. If True, executes'
          ' if_true agent; if False, executes if_false agent. If not provided,'
          ' defaults to always True.'
      ),
  )
