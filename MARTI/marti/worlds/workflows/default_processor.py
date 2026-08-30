from typing import List, Dict

def processor(
    trajectories: List[Dict],
    num_agents: int,
    global_args
) -> List[Dict[str, List]]:
    """
    Process collected trajectories into per-agent training samples.

    Args:
        trajectories: list of trajectory dicts (each has a "prompt" key plus per-turn items)
        num_agents: total number of agents
        sort_by_prompt: whether to group same-prompt trajectories together via a stable sort

    Returns:
        A list of length num_agents where each element is a dict with keys:
            "prompts", "outputs", "labels", "agent_indices" (for dual LoRA switching)
        Each is a list aligned by order.
    """
    # workflow knobs (optional)
    workflow_args = getattr(global_args, "workflow_args", None)
    if workflow_args is None and isinstance(global_args, dict):
        workflow_args = global_args.get("workflow_args", {})

    # group size for RLOO/GRPO-style estimators
    n_samples_per_prompt = 1
    try:
        if isinstance(global_args, dict):
            n_samples_per_prompt = int(global_args.get("n_samples_per_prompt", 1))
        else:
            n_samples_per_prompt = int(getattr(global_args, "n_samples_per_prompt", 1))
    except Exception:
        n_samples_per_prompt = 1
    n_samples_per_prompt = max(1, int(n_samples_per_prompt))
    # Default: don't drop early steps unless configured
    train_start_step = 1
    try:
        if isinstance(workflow_args, dict):
            train_start_step = int(workflow_args.get("train_start_step", 1))
        else:
            train_start_step = int(getattr(workflow_args, "train_start_step", 1))
    except Exception:
        train_start_step = 1

    # For MAS-style workflows, interleave samples by step across rollouts so that
    # n_samples_per_prompt grouping (RLOO/REINFORCE++) compares A vs B at the same step.
    interleave_by_step = True
    try:
        if isinstance(workflow_args, dict):
            interleave_by_step = bool(workflow_args.get("interleave_by_step", True))
        else:
            interleave_by_step = bool(getattr(workflow_args, "interleave_by_step", True))
    except Exception:
        interleave_by_step = True

    # group same prompts together by sorting (stable sort keeps rollout order within same prompt)
    trajectories = sorted(trajectories, key=lambda traj: traj.get("prompt", ""))

    # Initialize empty sample buckets for each agent
    samples = [
        {"prompts": [], "outputs": [], "labels": [], "agent_indices": []}
        for _ in range(num_agents)
    ]

    if not interleave_by_step:
        # Simple pass (legacy behavior): collect in raw order, optionally dropping early steps.
        for traj in trajectories:
            for turn in traj.get("trajectory", []):
                step_idx = (turn.get("metadata", {}) or {}).get("step_idx", 1)
                if isinstance(step_idx, int) and step_idx < train_start_step:
                    continue

                idx = turn["agent_index"]
                samples[idx]["prompts"].append(turn["agent_input"])
                samples[idx]["outputs"].append(turn["agent_output"])
                samples[idx]["labels"].append(turn["agent_reward"])
                samples[idx]["agent_indices"].append(idx)
        return samples

    # Interleaved mode:
    # - group trajectories by prompt (each group should have n_samples_per_prompt rollouts)
    # - within each prompt group, interleave by (agent_index, step_idx) across rollouts
    grouped: Dict[str, List[Dict]] = {}
    for traj in trajectories:
        grouped.setdefault(traj.get("prompt", ""), []).append(traj)

    for _prompt, traj_group in grouped.items():
        # build per-rollout lookup: (agent_index, step_idx) -> turn
        # Only use the first n rollouts (normally traj_group size == n_samples_per_prompt)
        # If fewer than n, we can't form a complete group for RLOO-style training.
        group_n = min(n_samples_per_prompt, len(traj_group))
        if n_samples_per_prompt > 1 and group_n < n_samples_per_prompt:
            continue

        rollout_maps = []
        for traj in traj_group[:group_n]:
            m = {}
            for turn in traj.get("trajectory", []):
                meta = turn.get("metadata", {}) or {}
                step_idx = meta.get("step_idx", 1)
                if isinstance(step_idx, int) and step_idx < train_start_step:
                    continue
                key = (turn.get("agent_index"), step_idx)
                m[key] = turn
            rollout_maps.append(m)

        # Interleave in increasing step order so rollout grouping (n_samples_per_prompt) is preserved.
        # IMPORTANT for RLOO: for each (agent_idx, step_idx), we must have exactly `group_n` samples,
        # otherwise total batch won't be divisible by group size and reward shaping will crash.
        for agent_idx in range(num_agents):
            # steps common to ALL rollouts for this agent
            step_sets = []
            for m in rollout_maps:
                step_sets.append({s for (a, s) in m.keys() if a == agent_idx})
            if not step_sets:
                continue
            common_steps = set.intersection(*step_sets)
            for step_idx in sorted(common_steps):
                key = (agent_idx, step_idx)
                for m in rollout_maps:
                    turn = m[key]
                    samples[agent_idx]["prompts"].append(turn["agent_input"])
                    samples[agent_idx]["outputs"].append(turn["agent_output"])
                    samples[agent_idx]["labels"].append(turn["agent_reward"])
                    samples[agent_idx]["agent_indices"].append(agent_idx)

    # Debug summary: how many samples were produced for each agent.
    # This helps detect the common failure mode where interleave_by_step + train_start_step
    # causes zero samples, leading to skipped_update and no ckpt.
    try:
        total_counts = [len(s["prompts"]) for s in samples]
        print(
            f"[Processor] interleave_by_step={interleave_by_step} train_start_step={train_start_step} "
            f"n_samples_per_prompt={n_samples_per_prompt} produced={total_counts}"
        )
    except Exception:
        pass

    return samples