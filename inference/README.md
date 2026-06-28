# MegaNorm Inference

Use `inference/infer_unified.py` for single-file or batch inference. Configure PatchNet and EdgeNet checkpoints in `configs/inference/base_config.yaml`.

```bash
python inference/infer_unified.py --config configs/inference/base_config.yaml --input path/to/input.ply --gpu 0
```

The script writes oriented normals to `outputs/inference/<config-name>/<input-name>.ply` and stores per-input metrics when ground-truth normals are present.
