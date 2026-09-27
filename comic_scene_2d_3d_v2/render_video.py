#!/usr/bin/env python3
"""
Pop-Up 3D Diorama Video Renderer
=================================
Renders a smooth camera orbit video showcasing the 3D pop-up book depth,
layered geometry, and environment of the comic panel scenes in Blender.

Pipeline:
  1. Opens the .blend scene
  2. Sets up a smooth orbiting camera trajectory with Track-To constraint
  3. Renders frame sequence via Blender EEVEE
  4. Encodes frames into high-quality H.264 MP4 video via imageio-ffmpeg
  5. Cleans up intermediate frames

Usage:
  python render_video.py output/panel1/panel1_scene.blend
  python render_video.py panel1.jpeg
  python render_video.py --all
"""

import os
import sys
import glob
import math
import shutil
import tempfile
import argparse
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_blender():
    """Find Blender executable."""
    candidates = [
        "/opt/homebrew/bin/blender",
        "/usr/local/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "blender",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    which_b = shutil.which("blender")
    if which_b:
        return which_b
    raise FileNotFoundError("Blender executable not found in PATH or standard locations.")


def get_ffmpeg_exe():
    """Get static ffmpeg binary bundled in imageio_ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        which_ff = shutil.which("ffmpeg")
        if which_ff:
            return which_ff
        return None


# Blender script that runs inside Blender to animate camera and render frames
BLENDER_ANIM_SCRIPT = """
import bpy
import math
import sys
import os

argv = sys.argv
if "--" in argv:
    args = argv[argv.index("--") + 1:]
else:
    args = []

frame_dir = args[0]
total_frames = int(args[1]) if len(args) > 1 else 72
res_pct = int(args[2]) if len(args) > 2 else 50
sweep_x = float(args[3]) if len(args) > 3 else 2.8
sweep_y = float(args[4]) if len(args) > 4 else 0.8

scene = bpy.context.scene
cam = scene.camera

if not cam:
    cam_data = bpy.data.cameras.new("RenderCam")
    cam = bpy.data.objects.new("RenderCam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

# Set render settings
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_percentage = res_pct
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.image_settings.color_depth = '8'

# Base camera distance
base_z = cam.location.z if cam.location.z > 0 else 10.0

# Create or reuse a target at the scene center
target_name = "CamOrbitTarget"
target = bpy.data.objects.get(target_name)
if not target:
    target = bpy.data.objects.new(target_name, None)
    target.location = (0, 0, 0.4)
    scene.collection.objects.link(target)
else:
    target.location = (0, 0, 0.4)

# Ensure camera has Track-To constraint
has_tt = False
for c in cam.constraints:
    if c.type == 'TRACK_TO':
        c.target = target
        c.track_axis = 'TRACK_NEGATIVE_Z'
        c.up_axis = 'UP_Y'
        has_tt = True
        break

if not has_tt:
    tt = cam.constraints.new('TRACK_TO')
    tt.target = target
    tt.track_axis = 'TRACK_NEGATIVE_Z'
    tt.up_axis = 'UP_Y'

# Set frame range
scene.frame_start = 1
scene.frame_end = total_frames

# Clear any previous animation on camera
cam.animation_data_clear()

# Animate camera trajectory:
# Frame 1: straight front view (looks like 2D flat panel)
# Frames 2 to 24: smooth sweep to right 3/4 angle (revealing pop-up layers!)
# Frames 25 to 54: smooth sweep across to left 3/4 angle (opposite parallax)
# Frames 55 to total_frames: return smoothly to center (perfect seamless loop)
for f in range(1, total_frames + 1):
    scene.frame_set(f)
    
    # Progress from 0 to 2*pi
    t = 2.0 * math.pi * (f - 1) / total_frames
    
    # Smooth Lissajous / orbital motion
    # X swings left/right with sine
    x = sweep_x * math.sin(t)
    
    # Y has subtle vertical tilt (figure-8 motion to see depth from above/below)
    y = sweep_y * math.sin(2.0 * t) - 0.2
    
    # Z pulls in slightly when viewing at angle to emphasize the 3D cutouts
    # and pulls back at center
    z_push = 0.8 * (abs(math.sin(t)))
    z = base_z - z_push
    
    cam.location = (x, y, z)
    cam.keyframe_insert(data_path="location", frame=f)

# Configure output path template
scene.render.filepath = os.path.join(frame_dir, "frame_")

print(f"[Blender] Starting render of {total_frames} frames to {frame_dir}...")
bpy.ops.render.render(animation=True)
print("[Blender] Animation render complete.")
"""


def render_scene_video(blend_path: str, output_mp4: str = None, total_frames: int = 72,
                       fps: int = 24, resolution_pct: int = 50, sweep_x: float = 2.8,
                       sweep_y: float = 0.8, keep_frames: bool = False):
    """
    Renders an animated showcase video of a .blend file.
    """
    blend_path = os.path.abspath(blend_path)
    if not os.path.exists(blend_path):
        raise FileNotFoundError(f"Blend file not found: {blend_path}")

    base_name = os.path.splitext(os.path.basename(blend_path))[0]
    scene_dir = os.path.dirname(blend_path)

    if output_mp4 is None:
        output_mp4 = os.path.join(scene_dir, f"{base_name}_showcase.mp4")
    output_mp4 = os.path.abspath(output_mp4)

    print("\n" + "=" * 70)
    print(f"🎬 RENDERING 3D POP-UP SHOWCASE VIDEO")
    print(f"   Scene:      {blend_path}")
    print(f"   Output MP4: {output_mp4}")
    print(f"   Frames:     {total_frames} @ {fps} fps ({total_frames / fps:.1f}s)")
    print(f"   Resolution: {resolution_pct}%")
    print("=" * 70)

    blender_bin = find_blender()
    ffmpeg_bin = get_ffmpeg_exe()

    # Create temporary directory for rendered frames
    temp_dir = tempfile.mkdtemp(prefix="blender_frames_")
    script_file = os.path.join(temp_dir, "anim_script.py")

    with open(script_file, "w") as f:
        f.write(BLENDER_ANIM_SCRIPT)

    cmd = [
        blender_bin,
        "--background",
        blend_path,
        "--python", script_file,
        "--",
        temp_dir,
        str(total_frames),
        str(resolution_pct),
        str(sweep_x),
        str(sweep_y),
    ]

    print("🎥 Invoking Blender renderer...")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            print(f"⚠ Blender returned error code {proc.returncode}")
            print(proc.stderr[-1000:])
    except Exception as e:
        print(f"❌ Error during Blender execution: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    # Collect rendered frames
    frame_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))
    if not frame_files:
        print("❌ No frames were rendered by Blender.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    print(f"✅ Rendered {len(frame_files)} frames. Encoding to MP4...")

    ffmpeg_bin = get_ffmpeg_exe()
    if not ffmpeg_bin:
        print("❌ Could not find ffmpeg binary for video encoding.")
        if not keep_frames:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    pattern = os.path.join(temp_dir, "frame_%04d.png")
    # Use ffmpeg with yuv420p for maximum compatibility (iOS, Mac QuickTime, browsers, Android)
    ff_cmd = [
        ffmpeg_bin,
        "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        output_mp4
    ]

    res = subprocess.run(ff_cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(output_mp4):
        print(f"🎉 Success! Showcase video saved to:\n   {output_mp4}")
        print(f"   Size: {os.path.getsize(output_mp4) / (1024*1024):.2f} MB")
    else:
        print(f"❌ ffmpeg encoding failed: {res.stderr}")

    # Cleanup temporary frame files
    if not keep_frames:
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print(f"Frames preserved in: {temp_dir}")

    return output_mp4


def main():
    parser = argparse.ArgumentParser(
        description="Render a 3D pop-up diorama showcase video from Blender scene"
    )
    parser.add_argument("target", nargs="?", help="Path to .blend file or panel image")
    parser.add_argument("--all", action="store_true", help="Render video for all current panels")
    parser.add_argument("--frames", type=int, default=72, help="Total frames (default: 72 = 3s @ 24fps)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24)")
    parser.add_argument("--res-pct", type=int, default=50, help="Resolution percentage (default: 50)")
    parser.add_argument("--sweep-x", type=float, default=2.8, help="Horizontal orbit sweep distance (default: 2.8)")
    parser.add_argument("--sweep-y", type=float, default=0.8, help="Vertical orbit sweep distance (default: 0.8)")
    parser.add_argument("--keep-frames", action="store_true", help="Keep rendered PNG frames")

    args = parser.parse_args()

    scenes = []

    if args.all:
        pattern = os.path.join(SCRIPT_DIR, "output", "*", "*_scene.blend")
        scenes = sorted(glob.glob(pattern))
        if not scenes:
            print("❌ No *_scene.blend files found in output directories.")
            sys.exit(1)
        print(f"Found {len(scenes)} scenes to render:")
        for s in scenes:
            print(f"  • {os.path.basename(s)}")

    elif args.target:
        target = os.path.abspath(args.target)
        if target.endswith(".blend"):
            scenes = [target]
        else:
            # Assume it's an image file or panel name
            base = os.path.splitext(os.path.basename(target))[0]
            candidate = os.path.join(SCRIPT_DIR, "output", base, f"{base}_scene.blend")
            if os.path.exists(candidate):
                scenes = [candidate]
            else:
                print(f"❌ Could not find blend scene for target '{args.target}' (checked {candidate})")
                sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    rendered_videos = []
    for sc in scenes:
        mp4_path = render_scene_video(
            sc,
            total_frames=args.frames,
            fps=args.fps,
            resolution_pct=args.res_pct,
            sweep_x=args.sweep_x,
            sweep_y=args.sweep_y,
            keep_frames=args.keep_frames,
        )
        if mp4_path and os.path.exists(mp4_path):
            rendered_videos.append(mp4_path)

    print("\n" + "=" * 70)
    print("🎬 VIDEO RENDERING COMPLETE")
    print("=" * 70)
    for v in rendered_videos:
        print(f"  ▶ {v} ({os.path.getsize(v) / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
