# Third-party assets

None of these files are stored in this repository. `goalkeeper/environments/assets.py` downloads them from the
pinned URLs below and converts them to USD.

| Asset | Source | Licence | Changes made |
|---|---|---|---|
| Futsal goal with net ("Porteria Futbol Sala 2") by atukeproductions | [Sketchfab](https://sketchfab.com/3d-models/none-2539585ea28a4b6e9d0c14d13c593c11), downloaded via the [Objaverse](https://huggingface.co/datasets/allenai/objaverse) mirror on Hugging Face | CC BY 4.0 | Converted GLB to USD, rotated to z-up, scaled 1.023× so the posts are exactly 3.0 m apart (inside), red stripes repainted white |
| Football | [Poly Haven "Football"](https://polyhaven.com/a/football) by Amal Kumar | CC0 | Used the inflated mesh only, recentred; wrapped in a rigid body with a sphere collider |
| Turf texture | [ambientCG Grass005](https://ambientcg.com/view?id=Grass005) | CC0 | Tinted per mowing stripe; whitened for line paint |
| Sky | [Poly Haven "Suburban Football Field"](https://polyhaven.com/a/suburban_football_field) HDRI (8K) | CC0 | None |
| Goalkeeper robot | Clearpath Jackal, NVIDIA Isaac Sim 5.1 asset library (`Isaac/Robots/Clearpath/Jackal/jackal.usd`) | NVIDIA Isaac Sim asset terms | Added a blocking panel as an extra rigid link on a fixed joint |

The goal model's CC BY 4.0 licence requires this credit to stay with any redistribution of renders or derived
files that include it.
