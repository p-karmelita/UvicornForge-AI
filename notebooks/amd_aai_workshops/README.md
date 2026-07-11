# AMD AAI Workshops Notebooks

This directory contains prepared Jupyter notebooks for all AMD AAI Workshop sessions relevant to the UnicornForge AI hackathon project.

## Goal
Integrate powerful AMD technologies (MI300X, Radeon, ROCm tools) into UnicornForge AI to:
- Improve ML model training/inference for startup success scoring
- Replace or augment Grok LLM with local high-perf AMD inference for brief generation
- Add multi-modal and video generation features
- Demonstrate deep use of sponsor tech for maximum hackathon impact (originality, business value, presentation, video)

## Notebooks

1. **01_multi_turn_rl...** - Qwen3 GRPO RL training (adapt for brief refiner/critic)
2. **02_inference...** - Hyperloom AI agents for inference tuning
3. **03_efficient_llm...** - vLLM + LMCache for scalable local LLM serving
4. **04_accelerating...** - AMD ATOM for faster inference
5. **05_building_gpu...** - FlyDSL custom kernels for project ML components
6. **06_real2sim...** - Synthetic embodied data generation to augment dataset
7. **07_generate_video...** - ComfyUI for auto-generating pitch/demo videos from briefs
8. **08_build_your_openclaw...** - Multi-modal OpenClaw for image+text idea input

## Usage
- Run on systems with ROCm (see project `backend/Dockerfile.rocm`)
- Each notebook includes project-specific adaptation notes
- Experiment and integrate learnings back into `backend/ml/` and `backend/services/`

## Next Steps for Project
- Use these to power local AMD LLM backend
- Add AMD platform features to success prediction
- Enable video export and multi-modal input in the app
- Record 5-min demo video showcasing the AMD stack in action

See main project README and docs for overall architecture.
