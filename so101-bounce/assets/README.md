# Assets

Real 3D models and textures for the ball-detector renders (`scripts/make_detector_data.py`). Physics does not use them.

- `fetch_simready.py` downloads into `raw/` (not committed).
- `convert_simready.py` writes `simready/`: OBJ meshes (≤ 30k triangles), albedo PNGs, `catalog.json`, `surfaces/`.

| Name in `simready/` | Source asset | Author | License |
|---|---|---|---|
| paddle_red | misc_toy_pongPaddle_bat_a01 | NVIDIA | CC-BY-4.0 |
| paddle_black | misc_toy_pongPaddle_bat_a02 | NVIDIA | CC-BY-4.0 |
| paddle_scan_orange | table_tennis_racket_3BBQBK5EWT | ALLSIDES | CC-BY-4.0 |
| mug_black | kitchenware_mug_a01 | NVIDIA | CC-BY-4.0 |
| mug_red | kitchenware_mug_f01 | NVIDIA | CC-BY-4.0 |
| mug_scan | mug_A07RDPW843 | ALLSIDES | CC-BY-4.0 |
| mouse_scan | computer_mouse_62RQ5CXV0E | ALLSIDES | CC-BY-4.0 |
| notebook | misc_notebook_a01 | NVIDIA | CC-BY-4.0 |
| tape_dispenser | misc_tape_dispenser_a01 | NVIDIA | CC-BY-4.0 |
| tape_roll | misc_tape_refill_a01 | NVIDIA | CC-BY-4.0 |
| marker_orange | stationery_marker_alcohol_chromeOrange_a01 | NVIDIA | CC-BY-4.0 |
| keyboard | computer_keyboard_a01 | NVIDIA | CC-BY-4.0 |
| cup_white | dishware_plastic_cup_a01 | NVIDIA | CC-BY-4.0 |
| lamp_white | lamp_table_bedside_a01 | NVIDIA | CC-BY-4.0 |
| surfaces/Wood049, Wood051, Wood066, WoodFloor041, Plastic006, Fabric030 | ambientCG 1K-JPG colour maps | ambientCG | CC0-1.0 |

Sources:
- NVIDIA SimReady Assets: https://huggingface.co/datasets/nvidia/simready-assets (revision 653d91e1)
- ALLSIDES SimReady Scanned Objects: https://huggingface.co/datasets/ALLSIDES/simready-scanned-objects
- Index of SimReady datasets: https://github.com/NVIDIA-Omniverse/simready-central
- ambientCG: https://ambientcg.com

Meshes were decimated and textures downscaled; otherwise unmodified. CC-BY-4.0: https://creativecommons.org/licenses/by/4.0/
