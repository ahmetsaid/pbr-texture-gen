# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PBR Texture Generator - A Replicate Cog model that generates PBR (Physically Based Rendering) texture maps from text prompts using Flux.1 Schnell. Outputs diffuse, normal, roughness, and ambient occlusion maps.

## Architecture

- **predict.py**: Main Cog predictor class
  - `Predictor.setup()`: Loads Flux.1 Schnell model with CPU offload
  - `Predictor.predict()`: Generates textures, yields 4 PNG files as iterator
  - Helper functions: `generate_normal_map()`, `generate_roughness_map()`, `generate_ao_map()`, `make_seamless()`
  - `TEXTURE_KEYWORDS`: List for validating texture-related prompts

- **cog.yaml**: Cog configuration (GPU, CUDA 12.1, Python 3.11, dependencies)

## Commands

### Local Development (requires Docker)
```bash
# Test prediction locally
cog predict -i prompt="seamless wood texture"

# Build Docker image
cog build

# Push to Replicate
cog login
cog push r8.im/ahmetsaid/pbr-texture-gen
```

### CI/CD
Push to `master` branch triggers GitHub Actions workflow that builds and pushes to Replicate. Requires `REPLICATE_API_TOKEN` secret.

## Key Dependencies

- `diffusers==0.30.0` (for FluxPipeline)
- `torch==2.2`
- `transformers==4.43.3`
- `scipy` (for Sobel/gaussian filters in PBR map generation)

## Model Details

- Uses `black-forest-labs/FLUX.1-schnell` via HuggingFace
- 4 inference steps, guidance_scale=0.0 (Schnell-specific)
- CPU offload enabled for memory efficiency
- Runs on Nvidia L40S GPU on Replicate
