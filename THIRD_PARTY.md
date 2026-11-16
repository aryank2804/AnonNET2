# Third-party code, models and data

This repository does **not** redistribute any of the following; fetch them from the original sources and respect their licences.

| Component | Used for | Source |
|---|---|---|
| AnonNET (MIT, © 2025 Anil Egin) | baseline; `baselines/batch_anonymize.py` and `baselines/scene_clips.py` are our modified copies of files from it | arXiv 2601.11635 |
| FLUX.1 [dev] | identity face generation (non-commercial licence) | https://huggingface.co/black-forest-labs/FLUX.1-dev |
| DreamID-V + Wan2.1 | video face swap | arXiv 2601.01425, https://github.com/Wan-Video/Wan2.1 |
| FaceXFormer | age / gender / race attributes | https://github.com/Kartik-3004/facexformer |
| InsightFace (ArcFace `buffalo_l`) | clustering, verification, attackers | https://github.com/deepinsight/insightface |
| SixDRepNet | head pose | `pip install sixdrepnet` |
| Q-Align (`q-future/one-align`), DINO ViT-B/16, VBench, I3D FVD (`flateon/FVD-I3D-torchscript`) | quality / coherence metrics | Hugging Face / original repos |
| VoxCeleb2, CelebV-HQ, MEAD, CREMA-D | source videos | original dataset pages (licences apply; videos are not included, only clip IDs) |
