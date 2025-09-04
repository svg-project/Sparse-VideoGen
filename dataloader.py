import json
import os


def load_prompt_or_image(prompt_source, prompt_idx, prompt, image_path):
    """
    Load the prompt or image path based on the prompt source.
    """
    if prompt_source == "prompt":
        assert prompt_idx == 0, "You have already provided a prompt"
        return prompt, image_path
    elif prompt_source == "I2V_VBench":
        # assert prompt is a json file
        assert prompt.endswith(".json"), "Prompt must be a json file"
        with open(prompt, "r") as f:
            prompts = json.load(f)

        prompt_idx = str(prompt_idx)
        original_prompt = prompts[prompt_idx]["original"]
        improved_prompt = prompts[prompt_idx]["improved"]
        image_path = os.path.join(image_path, f"{original_prompt}.jpg")
        assert os.path.exists(image_path), "Image path does not exist"

        return improved_prompt, image_path
    elif prompt_source == "I2V_Wan_Web":
        assert prompt == image_path, "Prompt and image path must be the same"

        prompt_idx = str(prompt_idx).zfill(3)
        prompt_path = os.path.join(prompt, f"{prompt_idx}/prompt.txt")
        image_path = os.path.join(image_path, f"{prompt_idx}/image.jpg")

        with open(prompt_path, "r") as f:
            prompt = f.read()
        return prompt, image_path

    elif prompt_source in ["T2V_Wan_VBench", "T2V_Hyv_VBench", "T2V_Hyv_Web"]:
        assert prompt.endswith(".txt"), "Prompt must be a txt file"
        with open(prompt, "r") as f:
            prompts = f.readlines()

        prompt = prompts[prompt_idx]
        return prompt, None
    elif prompt_source in ["T2V_Xingyang_Motion", "T2V_Xingyang_VBench"]:
        assert prompt.endswith(".txt"), "Prompt must be a txt file"
        with open(prompt, "r") as f:
            prompts = f.readlines()

        prompt = prompts[prompt_idx]
        return prompt, None
    else:
        raise ValueError(f"Invalid prompt source: {prompt_source}")
