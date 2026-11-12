"""
Phase 2: General Spatiotemporal Video Event Summarizer.
Divides video timelines into event-driven + periodic chunks, slices active
annotated frames in RAM into overlapping 2x2 spatial windows (skipping empty
quadrants), prompts a VLM per active tile with explicit [ID:X] / [OBJ:Y]
legends, rolls tile notes into cohesive chunk summaries, and indexes them into
LanceDB (`video_events`) for whole-video VSS Q&A.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image


QUADRANT_DEFS = [
    ("Top-Left", 0.0, 0.0, 0.575, 0.575),
    ("Top-Right", 0.425, 0.0, 1.0, 0.575),
    ("Bottom-Left", 0.0, 0.425, 0.575, 1.0),
    ("Bottom-Right", 0.425, 0.425, 1.0, 1.0),
]


def _draw_tagged_annotations(
    frame: np.ndarray,
    person_items: List[dict],
    object_items: List[dict],
) -> np.ndarray:
    """
    Draws clear [ID:X] person boxes and [OBJ:Y] object boxes onto a RAM copy
    of the frame so VLM spatial tiles visually ground every track/object tag.
    """
    canvas = frame.copy()

    for p in person_items:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        pid = p["person_id"]
        pose = p.get("pose", "unknown")
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 210, 80), 2)
        tag = f"[ID:{pid}] {pose}"
        cv2.putText(
            canvas,
            tag,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 210, 80),
            2,
            cv2.LINE_AA,
        )

    for obj in object_items:
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        oid = obj["object_id"]
        lbl = obj.get("label", "obj")
        owner = obj.get("owner_track_id", "Unknown")
        status = obj.get("status", "attended")
        color = (0, 0, 255) if status == "abandoned" else (0, 165, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        tag = f"[OBJ:{oid}:{lbl} ID:{owner}]"
        if status == "abandoned":
            tag += " ABANDONED"
        cv2.putText(
            canvas,
            tag,
            (x1, min(canvas.shape[0] - 6, y2 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )

    return canvas


def _resize_max_width(frame: np.ndarray, max_width: int = 960) -> Tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / float(w)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _box_intersects_region(bbox: List[float], rx1: float, ry1: float, rx2: float, ry2: float) -> bool:
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)


class VideoEventSummarizer:
    """
    Buffers annotated temporal windows during pipeline ingestion, then slices
    non-empty 2x2 overlapping tiles in RAM and summarizes events via a VLM backend.
    """

    def __init__(
        self,
        summary_interval: float = 8.0,
        sample_step_sec: float = 2.0,
        scenes_dir: str = "data/scenes",
        enable_tiling: bool = True,
    ):
        self.summary_interval = float(summary_interval)
        self.sample_step_sec = float(sample_step_sec)
        self.scenes_dir = Path(scenes_dir)
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        self.enable_tiling = enable_tiling

        self.completed_chunks: List[dict] = []
        self._chunk_idx = 0
        self._current_chunk: Optional[dict] = None
        self._last_sample_sec = -1e9
        self._prev_person_ids: set[str] = set()

    def _start_new_chunk(self, timestamp: float) -> None:
        self._chunk_idx += 1
        self._current_chunk = {
            "chunk_id": f"chunk_{self._chunk_idx:04d}",
            "start_sec": float(timestamp),
            "end_sec": float(timestamp),
            "trigger_reason": "periodic",
            "samples": [],          # list of {'timestamp', 'jpg_bytes', 'shape', 'persons', 'objects'}
            "events": [],           # list of discrete text events
            "track_ids": set(),
            "object_ids": set(),
        }

    def observe_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        person_items: List[dict],
        object_items: List[dict],
        discrete_events: Optional[List[str]] = None,
        force_flush_reason: Optional[str] = None,
    ) -> None:
        """
        Called per processed video frame during pipeline ingestion.
        Keeps RAM tiny by JPEG-encoding sampled resized annotated frames only once
        every `sample_step_sec` or on discrete events.
        """
        if self._current_chunk is None:
            self._start_new_chunk(timestamp)

        chunk = self._current_chunk
        chunk["end_sec"] = float(timestamp)

        curr_pids = {str(p["person_id"]) for p in person_items}
        for pid in curr_pids:
            chunk["track_ids"].add(pid)
        for obj in object_items:
            chunk["object_ids"].add(str(obj["object_id"]))

        entered = sorted(curr_pids - self._prev_person_ids)
        exited = sorted(self._prev_person_ids - curr_pids)
        if entered:
            chunk["events"].append(f"At {timestamp:.1f}s entered: " + ", ".join(f"[ID:{x}]" for x in entered))
        if exited:
            chunk["events"].append(f"At {timestamp:.1f}s exited/occluded: " + ", ".join(f"[ID:{x}]" for x in exited))
        self._prev_person_ids = curr_pids

        if discrete_events:
            chunk["events"].extend(discrete_events)

        should_sample = (
            (timestamp - self._last_sample_sec) >= self.sample_step_sec
            or bool(discrete_events)
            or bool(force_flush_reason)
            or len(chunk["samples"]) == 0
        )

        if should_sample:
            annotated = _draw_tagged_annotations(frame, person_items, object_items)
            resized, scale = _resize_max_width(annotated, max_width=960)
            ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                scaled_persons = [
                    {**p, "bbox": [v * scale for v in p["bbox"]]}
                    for p in person_items
                ]
                scaled_objects = [
                    {**o, "bbox": [v * scale for v in o["bbox"]]}
                    for o in object_items
                ]
                chunk["samples"].append({
                    "timestamp": float(timestamp),
                    "jpg_bytes": buf.tobytes(),
                    "shape": resized.shape[:2],  # (h, w)
                    "persons": scaled_persons,
                    "objects": scaled_objects,
                })
                self._last_sample_sec = float(timestamp)

        elapsed = timestamp - chunk["start_sec"]
        if force_flush_reason and elapsed >= 2.0:
            chunk["trigger_reason"] = force_flush_reason
            self._seal_current_chunk()
        elif elapsed >= self.summary_interval:
            chunk["trigger_reason"] = "periodic"
            self._seal_current_chunk()

    def _seal_current_chunk(self) -> None:
        if self._current_chunk and self._current_chunk["samples"]:
            self.completed_chunks.append(self._current_chunk)
        self._current_chunk = None

    def resolve_local_ids(self, local_to_global: Dict[str, str]) -> None:
        """Remaps pre-confirmation `local:X` person tags to final Global Track IDs."""
        self._seal_current_chunk()
        for chunk in self.completed_chunks:
            chunk["track_ids"] = {
                local_to_global.get(tid, tid) for tid in chunk["track_ids"]
            }
            new_events = []
            for ev in chunk["events"]:
                for ltag, gid in local_to_global.items():
                    ev = ev.replace(f"[ID:{ltag}]", f"[ID:{gid}]")
                new_events.append(ev)
            chunk["events"] = new_events
            for sample in chunk["samples"]:
                for p in sample["persons"]:
                    pid = str(p["person_id"])
                    if pid in local_to_global:
                        p["person_id"] = local_to_global[pid]

    def _decode_sample_rgb(self, sample: dict) -> np.ndarray:
        arr = np.frombuffer(sample["jpg_bytes"], dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _summarize_chunk(self, chunk: dict, vlm_backend) -> Tuple[str, List[dict], str]:
        samples = chunk["samples"]
        # Pick up to 2 representative frames from the chunk (start/mid and end)
        rep_samples = [samples[0]] if len(samples) == 1 else [samples[0], samples[-1]]
        mid_sample = samples[len(samples) // 2]

        mid_rgb = self._decode_sample_rgb(mid_sample)
        keyframe_path = self.scenes_dir / f"{chunk['chunk_id']}_keyframe.jpg"
        cv2.imwrite(str(keyframe_path), cv2.cvtColor(mid_rgb, cv2.COLOR_RGB2BGR))

        s_sec = chunk["start_sec"]
        e_sec = chunk["end_sec"]
        ev_text = "; ".join(chunk["events"][:10]) if chunk["events"] else "No discrete tracker entry/exit alerts."

        h, w = mid_sample["shape"]
        tile_notes: List[dict] = []

        if self.enable_tiling:
            decoded_reps = [(s, self._decode_sample_rgb(s)) for s in rep_samples]
            for q_name, fx1, fy1, fx2, fy2 in QUADRANT_DEFS:
                rx1, ry1, rx2, ry2 = int(fx1 * w), int(fy1 * h), int(fx2 * w), int(fy2 * h)

                # Collect entities present in this quadrant across representative frames
                q_persons = {}
                q_objects = {}
                for s_meta, _ in decoded_reps:
                    for p in s_meta["persons"]:
                        if _box_intersects_region(p["bbox"], rx1, ry1, rx2, ry2):
                            q_persons[str(p["person_id"])] = p.get("pose", "unknown")
                    for o in s_meta["objects"]:
                        if _box_intersects_region(o["bbox"], rx1, ry1, rx2, ry2):
                            q_objects[str(o["object_id"])] = (
                                f"{o.get('label', 'obj')} (owner=[ID:{o.get('owner_track_id', 'Unknown')}], "
                                f"status={o.get('status', 'attended')})"
                            )

                # Skip completely empty quadrants to save compute & prevent hallucinations
                if not q_persons and not q_objects:
                    continue

                tile_images: List[Image.Image] = []
                visual_items: List[dict] = []
                for idx, (s_meta, rgb) in enumerate(decoded_reps, 1):
                    crop_rgb = rgb[ry1:ry2, rx1:rx2]
                    if crop_rgb.size == 0:
                        continue
                    tile_images.append(Image.fromarray(crop_rgb))
                    visual_items.append({
                        "type": "tile",
                        "label": f"{q_name} Quadrant Frame #{idx} @ {s_meta['timestamp']:.1f}s",
                    })

                if not tile_images:
                    continue

                p_legend = ", ".join(f"[ID:{pid}] ({pose})" for pid, pose in sorted(q_persons.items())) or "None"
                o_legend = ", ".join(f"[OBJ:{oid}] {desc}" for oid, desc in sorted(q_objects.items())) or "None"
                tile_prompt = (
                    f"Analyze ONLY the {q_name} spatial quadrant of this scene ({s_sec:.1f}s to {e_sec:.1f}s).\n"
                    f"Detected Tags Inside Quadrant:\n"
                    f"- People: {p_legend}\n"
                    f"- Objects: {o_legend}\n\n"
                    f"Describe concretely in 1-2 short factual sentences what actions, speaking/presenting, "
                    f"interactions, posture changes, or object handling are happening here. "
                    f"Refer to individuals by their exact [ID:X] tags and items by [OBJ:Y] tags."
                )
                try:
                    note = vlm_backend.generate_answer(
                        question="Summarize quadrant activity",
                        prompt_text=tile_prompt,
                        visual_items=visual_items,
                        images=tile_images,
                        max_new_tokens=160,
                    )
                except Exception as exc:
                    note = f"[Tile generation error: {exc}]"
                tile_notes.append({"quadrant": q_name, "note": note.strip()})

        # Fallback: whole keyframe if all spatial quadrants had 0 tracked boxes
        if not tile_notes:
            pil_full = Image.fromarray(mid_rgb)
            full_prompt = (
                f"Describe concisely in 1-2 sentences what is happening in this frame "
                f"between {s_sec:.1f}s and {e_sec:.1f}s. Tracker log: {ev_text}"
            )
            try:
                note = vlm_backend.generate_answer(
                    question="Summarize scene chunk",
                    prompt_text=full_prompt,
                    visual_items=[{"type": "scene", "label": f"Chunk {chunk['chunk_id']} ({s_sec:.1f}s-{e_sec:.1f}s)"}],
                    images=[pil_full],
                    max_new_tokens=160,
                )
            except Exception as exc:
                note = f"Scene window {s_sec:.1f}s-{e_sec:.1f}s. {ev_text} (VLM note unavailable: {exc})"
            tile_notes.append({"quadrant": "FullFrame", "note": note.strip()})

        # Roll up tile observations + tracker log into one crisp chunk summary
        if len(tile_notes) == 1 and not chunk["events"]:
            summary_text = tile_notes[0]["note"]
        else:
            tiles_joined = "\n".join(f"- {t['quadrant']}: {t['note']}" for t in tile_notes)
            rollup_prompt = (
                f"Synthesize the spatial window observations and tracker events for video interval "
                f"{s_sec:.1f}s - {e_sec:.1f}s into ONE cohesive 2-3 sentence summary.\n"
                f"Preserve exact [ID:X] person tags and [OBJ:Y] object tags.\n\n"
                f"Spatial Quadrant Notes:\n{tiles_joined}\n\n"
                f"Tracker & Object Events:\n{ev_text}\n\n"
                f"Cohesive Summary:"
            )
            try:
                summary_text = vlm_backend.generate_text_only(rollup_prompt, max_new_tokens=180)
            except Exception:
                summary_text = f"Interval {s_sec:.1f}s-{e_sec:.1f}s: " + " | ".join(
                    f"{t['quadrant']}: {t['note']}" for t in tile_notes
                )
                if chunk["events"]:
                    summary_text += f" | Events: {ev_text}"

        return summary_text.strip(), tile_notes, str(keyframe_path)

    def summarize_all(self, vlm_backend) -> List[dict]:
        """Runs VLM spatial-window + rollup summarization across all chunks."""
        self._seal_current_chunk()
        if not self.completed_chunks:
            return []

        print(f"\n[VideoSummarizer] Summarizing {len(self.completed_chunks)} spatiotemporal chunk(s)...")
        summarized = []
        for chunk in self.completed_chunks:
            summary_text, tile_notes, keyframe_path = self._summarize_chunk(chunk, vlm_backend)
            if not summary_text:
                continue
            summarized.append({
                "chunk_id": chunk["chunk_id"],
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],
                "trigger_reason": chunk["trigger_reason"],
                "summary_text": summary_text,
                "tile_notes": tile_notes,
                "referenced_track_ids": sorted(chunk["track_ids"]),
                "referenced_object_ids": sorted(chunk["object_ids"]),
                "keyframe_path": keyframe_path,
            })
            print(
                f"  -> Summarized {chunk['chunk_id']} ({chunk['start_sec']:.1f}s-{chunk['end_sec']:.1f}s): "
                f"{summary_text[:120]}..."
            )
        return summarized

    def finalize(self, vlm_backend, embedder, vector_db, video_source: str) -> int:
        """
        Seals the final buffered chunk, summarizes all chunks using `vlm_backend`,
        embeds summaries with `embedder`, and stores them in LanceDB `video_events`.
        Returns the number of indexed event chunks.
        """
        summarized = self.summarize_all(vlm_backend)
        if not summarized:
            print("[VideoSummarizer] No temporal chunks captured; skipping video_events indexing.")
            return 0

        indexed = 0
        for item in summarized:
            vec = embedder.extract_multimodal_text(item["summary_text"])
            vector_db.insert_video_event({
                **item,
                "siglip_vector": vec,
                "video_source": video_source,
            })
            indexed += 1
        return indexed
