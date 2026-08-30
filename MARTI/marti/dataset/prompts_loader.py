import random
import json
from tqdm import tqdm
from torch.utils.data import Dataset

def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None, add_prompt_suffix=None, add_system_prompt=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if add_prompt_suffix is not None:
            chat += add_prompt_suffix
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        if add_system_prompt:
            chat = [{"role": "system", "content": add_system_prompt}] + chat
        prompt = apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if add_prompt_suffix is not None:
            prompt += add_prompt_suffix
        if input_template:
            prompt = input_template.format(prompt)
    return prompt


class PromptDatasetWithLabel(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
        add_prompt_suffix=None
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(
            self.strategy.args, "apply_chat_template", False)

        add_think_token = getattr(self.strategy.args, "add_think_token", 0)
        add_system_prompt = getattr(
            self.strategy.args, "add_system_prompt", None)
        if apply_chat_template and self.tokenizer is not None:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        indice = 0
        print(dataset)
        for data in tqdm(dataset, desc="Preprocessing data"):
            prompt = preprocess_data(data, input_template, input_key,
                                     apply_chat_template, add_prompt_suffix, add_system_prompt)
            if apply_chat_template and add_think_token != 0:
                prompt = prompt + "<think>"
            label = data[label_key]

            if isinstance(label, list):
                label = label[0]
            if isinstance(label, float) or isinstance(label, int):
                label = str(label)

            metadata_key = self.strategy.args.metadata_key
            raw_meta = data.get(metadata_key, {})
            # `metadata` should be a dict. Keep it structured so downstream workflows can do `metadata.get(...)`.
            # Some datasets might store it as JSON string; decode if needed.
            if raw_meta is None:
                raw_meta = {}
            if isinstance(raw_meta, str):
                try:
                    raw_meta = json.loads(raw_meta)
                except Exception:
                    raw_meta = {}
            if not isinstance(raw_meta, dict):
                raw_meta = {}

            # Always include a stable id for debugging / tracing.
            meta = {"id": indice, **raw_meta}
            sample = {"prompt": prompt, "label": label, "indice": indice, "metadata": meta}

            self.prompts.append(sample)
            indice += 1

        for sample in random.sample(self.prompts, 2):
            print(sample)
            print("="*20)

    def get_all_prompts(self):
        return self.prompts

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
