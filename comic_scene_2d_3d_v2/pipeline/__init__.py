# Comic Panel → 3D Low-Poly Blender Scene Pipeline
# =================================================
# This package converts 2D comic panel images into traversable 3D Blender scenes.
#
# Pipeline stages:
#   1. Depth Estimation (MiDaS / DPT)
#   2. Semantic Segmentation (SAM-2 / rembg)
#   3. Layer Decomposition (foreground/background separation)
#   4. Mesh Generation (depth-map → displaced plane, per-layer)
#   5. Color/Texture Projection (original colors baked onto meshes)
#   6. Blender Scene Assembly (environment, lighting, camera)
