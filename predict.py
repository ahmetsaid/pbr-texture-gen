import os
import re
import tempfile
from typing import Iterator
from cog import BasePredictor, Input, Path
import torch
from PIL import Image
import numpy as np
from diffusers import FluxPipeline
from scipy.ndimage import sobel, gaussian_filter


# Texture-related keywords for validation
TEXTURE_KEYWORDS = [
    "texture", "material", "surface", "pattern", "seamless", "tileable",
    "pbr", "wood", "metal", "stone", "brick", "concrete", "fabric",
    "leather", "marble", "tile", "floor", "wall", "ground", "rock",
    "grass", "sand", "dirt", "rust", "paint", "plastic", "glass",
    "ceramic", "asphalt", "gravel", "carpet", "cloth", "paper",
    "bark", "moss", "ice", "snow", "water", "lava", "crystal",
    "scale", "skin", "fur", "grain", "woven", "knit", "plaster",
    "stucco", "terracotta", "granite", "slate", "cobblestone"
]


def is_texture_prompt(prompt: str) -> tuple[bool, str]:
    """Check if the prompt is texture-related."""
    prompt_lower = prompt.lower()

    # Check for texture keywords
    found_keywords = [kw for kw in TEXTURE_KEYWORDS if kw in prompt_lower]

    if found_keywords:
        return True, ""

    # Warning message for non-texture prompts
    warning = (
        "⚠️ WARNING: Your prompt doesn't appear to be texture-related. "
        "For best PBR results, include terms like: texture, material, surface, "
        "seamless, tileable, or specific material names (wood, metal, stone, etc.). "
        "Proceeding anyway, but results may not be suitable for PBR workflows."
    )
    return False, warning


def make_seamless(image: Image.Image, strength: float = 0.5) -> Image.Image:
    """Apply seamless tiling blend to image edges."""
    if strength <= 0:
        return image

    img_array = np.array(image, dtype=np.float32)
    h, w = img_array.shape[:2]

    # Blend size based on strength
    blend_size = int(min(h, w) * 0.25 * strength)
    if blend_size < 2:
        return image

    result = img_array.copy()

    # Create smooth blend weights
    weights = np.linspace(0, 1, blend_size)

    # Horizontal seam blending (left-right)
    for i, weight in enumerate(weights):
        # Blend left edge with right edge
        left_col = i
        right_col = w - blend_size + i

        if len(img_array.shape) == 3:
            result[:, left_col] = (1 - weight) * img_array[:, right_col] + weight * img_array[:, left_col]
            result[:, right_col] = weight * img_array[:, left_col] + (1 - weight) * img_array[:, right_col]
        else:
            result[:, left_col] = (1 - weight) * img_array[:, right_col] + weight * img_array[:, left_col]
            result[:, right_col] = weight * img_array[:, left_col] + (1 - weight) * img_array[:, right_col]

    # Vertical seam blending (top-bottom)
    for i, weight in enumerate(weights):
        top_row = i
        bottom_row = h - blend_size + i

        if len(img_array.shape) == 3:
            result[top_row, :] = (1 - weight) * result[bottom_row, :] + weight * result[top_row, :]
            result[bottom_row, :] = weight * result[top_row, :] + (1 - weight) * result[bottom_row, :]
        else:
            result[top_row, :] = (1 - weight) * result[bottom_row, :] + weight * result[top_row, :]
            result[bottom_row, :] = weight * result[top_row, :] + (1 - weight) * result[bottom_row, :]

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def generate_normal_map(diffuse: Image.Image, strength: float = 1.0) -> Image.Image:
    """Generate normal map from diffuse texture using Sobel operator."""
    # Convert to grayscale for height estimation
    gray = np.array(diffuse.convert("L"), dtype=np.float32) / 255.0

    # Apply slight blur to reduce noise
    gray = gaussian_filter(gray, sigma=0.5)

    # Calculate gradients using Sobel
    dx = sobel(gray, axis=1) * strength
    dy = sobel(gray, axis=0) * strength

    # Normalize
    dz = np.ones_like(gray)

    # Stack and normalize to create normal map
    normals = np.stack([dx, -dy, dz], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norm + 1e-8)

    # Convert from [-1, 1] to [0, 255]
    normal_map = ((normals + 1) * 0.5 * 255).astype(np.uint8)

    return Image.fromarray(normal_map, mode="RGB")


def generate_roughness_map(diffuse: Image.Image) -> Image.Image:
    """Generate roughness map from diffuse texture."""
    # Convert to grayscale
    gray = np.array(diffuse.convert("L"), dtype=np.float32)

    # Estimate roughness from local variance and intensity
    # Darker areas and high-variance areas tend to be rougher

    # Local variance for texture detail
    blurred = gaussian_filter(gray, sigma=3)
    local_var = gaussian_filter((gray - blurred) ** 2, sigma=5)
    local_var = local_var / (local_var.max() + 1e-8)

    # Invert intensity (darker = rougher for many materials)
    intensity = 1.0 - (gray / 255.0)

    # Combine factors
    roughness = 0.5 * local_var + 0.3 * intensity + 0.2
    roughness = np.clip(roughness * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(roughness, mode="L")


def generate_ao_map(diffuse: Image.Image) -> Image.Image:
    """Generate ambient occlusion map from diffuse texture."""
    # Convert to grayscale
    gray = np.array(diffuse.convert("L"), dtype=np.float32) / 255.0

    # AO is derived from local darkness/depth cues
    # Apply multi-scale blur for soft shadows
    ao_fine = gaussian_filter(gray, sigma=2)
    ao_medium = gaussian_filter(gray, sigma=8)
    ao_coarse = gaussian_filter(gray, sigma=16)

    # Combine scales - darker areas = more occlusion
    ao = 0.4 * ao_fine + 0.35 * ao_medium + 0.25 * ao_coarse

    # Normalize and invert (darker input = darker AO = more occlusion)
    ao = (ao - ao.min()) / (ao.max() - ao.min() + 1e-8)

    # Boost contrast and ensure mostly white with dark crevices
    ao = np.power(ao, 0.7)  # Gamma correction
    ao = 0.3 + 0.7 * ao  # Limit darkness range

    ao_map = (ao * 255).astype(np.uint8)

    return Image.fromarray(ao_map, mode="L")


class Predictor(BasePredictor):
    def setup(self):
        """Load the Flux.1 Schnell model."""
        print("Loading Flux.1 Schnell model...")

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16
        )
        self.pipe.to("cuda")

        # Enable memory optimizations
        self.pipe.enable_model_cpu_offload()

        print("Model loaded successfully!")

    def predict(
        self,
        prompt: str = Input(
            description="Text description of the texture to generate. Include terms like 'seamless', 'tileable', 'texture', or material names for best results.",
            default="seamless dark wood texture, highly detailed, 8k"
        ),
        resolution: int = Input(
            description="Output resolution for all texture maps",
            choices=[512, 1024, 2048],
            default=1024
        ),
        tiling_strength: float = Input(
            description="Strength of seamless tiling blend (0 = no tiling, 1 = maximum tiling)",
            ge=0.0,
            le=1.0,
            default=0.5
        ),
        seed: int = Input(
            description="Random seed for reproducibility. Use -1 for random.",
            default=-1
        ),
    ) -> Iterator[Path]:
        """Generate PBR texture maps from a text prompt."""

        # Validate prompt
        is_texture, warning = is_texture_prompt(prompt)
        if not is_texture:
            print(warning)

        # Handle seed
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(seed)

        print(f"Generating with seed: {seed}")
        print(f"Resolution: {resolution}x{resolution}")
        print(f"Tiling strength: {tiling_strength}")

        # Enhance prompt for texture generation
        enhanced_prompt = f"{prompt}, seamless tileable texture, top-down view, flat lighting, no perspective, uniform surface"

        # Generate base diffuse texture
        print("Generating diffuse texture...")
        image = self.pipe(
            prompt=enhanced_prompt,
            width=resolution,
            height=resolution,
            num_inference_steps=4,  # Schnell is optimized for 4 steps
            generator=generator,
            guidance_scale=0.0,  # Schnell doesn't use guidance
        ).images[0]

        # Apply seamless tiling
        if tiling_strength > 0:
            print(f"Applying seamless tiling (strength: {tiling_strength})...")
            image = make_seamless(image, tiling_strength)

        # Create temp directory for outputs
        output_dir = tempfile.mkdtemp()

        # Save diffuse map
        diffuse_path = os.path.join(output_dir, "diffuse.png")
        image.save(diffuse_path)
        print("✓ Diffuse map generated")
        yield Path(diffuse_path)

        # Generate and save normal map
        print("Generating normal map...")
        normal = generate_normal_map(image)
        if tiling_strength > 0:
            normal = make_seamless(normal, tiling_strength)
        normal_path = os.path.join(output_dir, "normal.png")
        normal.save(normal_path)
        print("✓ Normal map generated")
        yield Path(normal_path)

        # Generate and save roughness map
        print("Generating roughness map...")
        roughness = generate_roughness_map(image)
        if tiling_strength > 0:
            roughness = make_seamless(roughness, tiling_strength)
        roughness_path = os.path.join(output_dir, "roughness.png")
        roughness.save(roughness_path)
        print("✓ Roughness map generated")
        yield Path(roughness_path)

        # Generate and save AO map
        print("Generating ambient occlusion map...")
        ao = generate_ao_map(image)
        if tiling_strength > 0:
            ao = make_seamless(ao, tiling_strength)
        ao_path = os.path.join(output_dir, "ao.png")
        ao.save(ao_path)
        print("✓ AO map generated")
        yield Path(ao_path)

        print(f"\n✅ All PBR maps generated successfully!")
        print(f"Seed used: {seed}")
        if not is_texture:
            print(f"\n{warning}")
