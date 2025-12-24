# PBR Texture Generator

Generate complete PBR (Physically Based Rendering) texture maps from text descriptions using Flux.1 Schnell.

## Outputs

| Map | Description |
|-----|-------------|
| **Diffuse** | Base color/albedo texture |
| **Normal** | Surface detail and depth |
| **Roughness** | Surface smoothness (black=smooth, white=rough) |
| **AO** | Ambient occlusion for soft shadows |

## Usage

```python
import replicate

output = replicate.run(
    "ahmetsaid/pbr-texture-gen",
    input={
        "prompt": "seamless red brick wall texture, weathered, 8k",
        "resolution": 1024,
        "tiling_strength": 0.5,
        "seed": 42
    }
)

# Returns: [diffuse.png, normal.png, roughness.png, ao.png]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | - | Texture description. Include "seamless", "tileable" for best results |
| `resolution` | int | 1024 | Output size: 512, 1024, or 2048 |
| `tiling_strength` | float | 0.5 | Seamless blend strength (0-1) |
| `seed` | int | -1 | Random seed (-1 for random) |

## Example Prompts

- `seamless dark wood planks texture, oak, detailed grain, 8k`
- `tileable concrete wall texture, cracked, urban, weathered`
- `seamless grass texture, lush green, top-down view`
- `metal diamond plate texture, industrial, scratched`
- `seamless marble texture, white carrara, gold veins`
- `brick wall texture, old red bricks, mortar, weathered`

## Tips

1. **Always include "seamless" or "tileable"** for textures that need to repeat
2. **Add material details** like "weathered", "worn", "polished", "rough"
3. **Specify the view** - "top-down view" works best for floor textures
4. **Use quality terms** like "8k", "highly detailed", "photorealistic"

## License

Apache 2.0
