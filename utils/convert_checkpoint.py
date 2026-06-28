import argparse
from pathlib import Path

import torch


def extract_model_state_dict(checkpoint):
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("state_dict", checkpoint)
    model_state_dict = {}
    for key, value in state_dict.items():
        while key.startswith(("model.", "module.")):
            key = key.split(".", 1)[1]
        model_state_dict[key] = value
    return model_state_dict


def load_model_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return extract_model_state_dict(checkpoint)


def ckpt_to_pth(ckpt_path, pth_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    torch.save(
        {
            "epoch": ckpt.get("epoch", 0),
            "model_state_dict": extract_model_state_dict(ckpt),
            "config": ckpt.get("hyper_parameters", {}),
        },
        pth_path,
    )


def pth_to_ckpt(pth_path, ckpt_path):
    pth = torch.load(pth_path, map_location="cpu", weights_only=False)
    model_state = pth.get("model_state_dict", pth.get("state_dict", pth))
    torch.save(
        {
            "epoch": pth.get("epoch", 0),
            "global_step": 0,
            "state_dict": {f"model.{key}": value for key, value in model_state.items()},
            "hyper_parameters": pth.get("config", {}),
        },
        ckpt_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Convert Lightning and legacy checkpoints")
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(
        ".pth" if input_path.suffix == ".ckpt" else ".ckpt"
    )
    if input_path.suffix == ".ckpt":
        ckpt_to_pth(input_path, output_path)
    elif input_path.suffix == ".pth":
        pth_to_ckpt(input_path, output_path)
    else:
        raise ValueError("input must be .ckpt or .pth")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
