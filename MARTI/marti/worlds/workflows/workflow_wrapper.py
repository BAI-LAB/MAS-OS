import time
import ray
import asyncio
import copy
from typing import Dict, List, Any, Optional
from marti.helpers.logging import init_logger
from marti.verifiers.auto_verify import auto_verify

logger = init_logger(__name__)


@ray.remote
class MultiAgentWrapper:
    """Wrapper for managing multiple agents and their interactions."""
    
    def __init__(self, agents: List[Dict[str, Any]], workflow_args, *args, **kwargs):
        """
        Initialize multi-agent wrapper.
        
        Args:
            agents: List of agent configurations, each containing:
                - name: Agent identifier
                - llm: LLMRayActorAsync instance
                - tokenizer: Tokenizer for the agent
                - sampling_params: Default sampling parameters
                - workflow_func_path: Path to agent step function (optional)
        """
        self.agents = agents
        self.workflow_args = workflow_args
        self.workflow_func_path = kwargs.pop("workflow_func_path")
        self.result_queue = asyncio.Queue()
        
        # Load workflow function once during initialization
        if self.workflow_func_path.endswith(".py"):
            import importlib.util
            spec = importlib.util.spec_from_file_location("workflow", self.workflow_func_path)
            workflow_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(workflow_module)
            self.workflow_func = workflow_module.workflow
        else:
            raise ValueError("Workflow path must be a Python file")

    async def add_requests(
        self,
        tool_manager,
        prompts: List[str],
        labels: List[str],
        task: str,
        metadata: Optional[List[Dict]] = None,
        max_length: int = None,
        is_eval: bool = False
    ) -> List[Dict[str, Any]]:
        """Process requests using multi-agent workflow."""
        
        # Create semaphore to control concurrent workflow execution
        semaphore = asyncio.Semaphore(tool_manager.get_num_workers())
        
        async def execute_workflow(prompt: str, label: str, meta: Optional[Dict] = None, prompt_id: int = 0):
            """Execute a single workflow instance."""
            async with semaphore:
                workflow_start = time.time()
                
                try:
                    # Execute workflow directly without creating remote instance
                    result = await self.workflow_func(
                        prompt=prompt,
                        label=label,
                        agents=self.agents,
                        tool_manager=tool_manager,
                        task=task,
                        metadata=meta,
                        workflow_args=self.workflow_args,
                        max_length=max_length,
                        prompt_id=prompt_id,
                        is_eval=is_eval,
                    )
                    
                    # Add timing information
                    result["workflow_time"] = time.time() - workflow_start
                    result["prompt_id"] = prompt_id

                    # Store result
                    await self.result_queue.put(result)

                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    logger.error(f"Workflow execution error: {e}\n{tb}")
                    error_result = {
                        "prompt": prompt,
                        "label": label,
                        "trajectory": [],
                        "final_reward": 0,
                        "error": str(e),
                        "traceback": tb,
                        "prompt_id": prompt_id,
                        "workflow_time": time.time() - workflow_start
                    }
                    await self.result_queue.put(error_result)

        # Create tasks for all workflows
        if metadata is None:
            metadata = [{} for _ in range(len(prompts))]

        tasks = []
        for idx, (prompt, label, meta) in enumerate(zip(prompts, labels, metadata)):
            tasks.append(execute_workflow(prompt, label, copy.deepcopy(meta), prompt_id=idx))

        # Execute all workflows concurrently
        await asyncio.gather(*tasks)

    async def get_responses(self, expected_len: int) -> List[Dict[str, Any]]:
        results = []
        for _ in range(expected_len):
            results.append(await self.result_queue.get())
        return results