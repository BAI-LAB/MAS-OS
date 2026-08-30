import os
import ray
from typing import Any, List

from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from marti.helpers.logging import init_logger
from marti.models.vllm.wrapper import get_bundle_indices
from marti.helpers.common import ray_noset_visible_devices

logger = init_logger(__name__)


@ray.remote
def get_all_env_variables():
    import os
    return os.environ

class BaseLLMRayActor:
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        kwargs.pop("agent_func_path", None)
        noset_visible_devices = ray_noset_visible_devices()
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        full_determinism = kwargs.pop("full_determinism", False)
        if full_determinism:
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        self.kwargs = kwargs


@ray.remote
class LLMRayActor(BaseLLMRayActor):
    def __init__(self, *args, bundle_indices: list = None,  **kwargs):
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        # Disable vLLM V1 engine
        os.environ["VLLM_USE_V1"] = "0"
        
        import vllm
        self.llm = vllm.LLM(*args, **self.kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()

    def generate(self, *args, **kwargs):
        sampling_params = kwargs.get("sampling_params", None)
        if sampling_params is not None:
            truncate_prompt_tokens = sampling_params.truncate_prompt_tokens
        else:
            truncate_prompt_tokens = None
        
        if truncate_prompt_tokens is not None :
            if args:
                prompts = args[0]
                prompt_token_ids = self.tokenizer(
                        prompts,
                        add_special_tokens=False,
                        max_length=truncate_prompt_tokens,
                        truncation=True)["input_ids"]
            else:
                prompt_token_ids = [token_ids[:truncate_prompt_tokens] for token_ids in kwargs.get("prompt_token_ids")]
            
            clean_kwargs = {k: v for k, v in kwargs.items() 
                if k not in ("prompts", "prompt_token_ids")}
            return self.llm.generate(prompt_token_ids=prompt_token_ids, **clean_kwargs)
        else:
            return self.llm.generate(*args, **kwargs)

    # def add_requests(self, sampling_params, prompt_token_ids):
    #     """
    #     Process requests from rank0 and generate responses.
    #     Since only rank0 will send requests, we don't need to track actor ranks.
    #     """
    #     # from vllm.inputs import TokensPrompt

    #     requests = [TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids]
    #     responses = self.llm.generate(prompts=requests, sampling_params=sampling_params)
    #     self.response_queues.put(responses)

    # def get_responses(self):
    #     """
    #     Return the responses for the actor with the given rank
    #     """
    #     return self.response_queues.get()


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
    llm_actor_cls=LLMRayActor,
    agent_func_path=None
):
    import vllm

    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"

    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)

    if use_hybrid_engine and tensor_parallel_size == 1:
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        num_gpus = 0.2

    if not use_hybrid_engine:
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    vllm_engines = []

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = get_bundle_indices(shared_pg, i, tensor_parallel_size)

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg, 
            placement_group_capture_child_tasks=True, 
            placement_group_bundle_index=bundle_indices[0] if bundle_indices else i
        )

        vllm_engines.append(
            llm_actor_cls.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                pretrain,
                enforce_eager=enforce_eager,
                worker_extension_cls="marti.models.vllm.wrapper.WorkerWrap",
                distributed_executor_backend=distributed_executor_backend,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                dtype="bfloat16",
                seed=seed + i,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices,
                enable_prefix_caching=enable_prefix_caching,
                max_model_len=max_model_len,
                num_gpus=0.2 if use_hybrid_engine else 1,
                enable_sleep_mode=vllm_enable_sleep,
                agent_func_path=agent_func_path
            )
        )

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """
    Batch call a method on multiple vLLM engines.
    Args:
        engines: List of vLLM engine instances
        method_name: Name of the method to call
        rank_0_only: Only execute on rank 0 if True
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
    Returns:
        List of results from ray.get() if on rank 0, None otherwise
    """
    import torch

    if torch.distributed.is_initialized():
        if rank_0_only and torch.distributed.get_rank() != 0:
            return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)

if __name__ == "__main__":
    llm = LLMRayActor.remote("meta-llama/Llama-2-7b-chat-hf", tensor_parallel_size=4)
    output = ray.get(llm.generate.remote("San Franciso is a"))
    print(f"output: {output}")
