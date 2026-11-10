# Crowd Counting R&D Handover

## Goal

Reduce false positives in crowd counting on an auditorium image. Chairs are being detected as heads in the density map, which inflates the count. Test a small set of pretrained models and see which one gives a count closest to the true number of people in the image.

No training or fine-tuning. Use pretrained weights only.

## Scope

Four approaches, one script file each, in a single folder named `rnd_crowd/`:

1. `lwcc.py`: CSRNet and DM-Count via the LWCC library (`pip install lwcc`)
2. `p2pnet.py`: P2PNet, point-based (clone `https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet`)
3. `apgcc.py`: APGCC, point-based (clone the APGCC repo and use its pretrained weights)
4. `fidtm.py`: FIDTM, density-style with focal inverse distance transform maps (clone the FIDTM repo, `https://github.com/dk-liang/FIDTM`)
5. `cltr.py`: CLTR, transformer-based point localization (clone the CLTR repo, `https://github.com/dk-liang/CLTR`)
6. `cctrans.py`: CCTrans, transformer-based density estimation with global context (clone the CCTrans repo, `https://github.com/dk-liang/CCTrans`)

Test on at most 3 images. The lwcc.py file covers both CSRNet and DM-Count, so run it once per model.

The repo URLs above are from memory. Confirm each one, and the pretrained weight download links, from the repo's own README before cloning. Use the repo's test or inference script as the reference for model construction, checkpoint loading, and output format.

## Requirements

- Keep the code simple. This is an R&D task, not a production system.
- One file per approach. No shared framework or config system.
- Each script takes an image path as its main argument and prints one line of JSON with at least `method` and `count`.
- Each script saves an overlay image showing the detections or density map, so the result can be checked visually.
- Post-processing options should be command-line flags (for example, a density threshold for LWCC, a score threshold for the point-based models).
- Install dependencies with pip. Use a virtual environment if it helps.

## Steps

1. Set up the environment: `pip install lwcc torch torchvision pillow matplotlib`. Confirm LWCC imports and check its model list (`LWCC.load_model` options and weight names) against the installed version, since names can differ between releases.
2. Clone the P2PNet, APGCC, FIDTM, CLTR and CCTrans repos. Use the pretrained weights each one publishes. CLTR outputs head points, like P2PNet and APGCC. FIDTM and CCTrans output density maps, like LWCC, so reuse the same density-threshold post-processing idea there. Read each repo's README and test script to get the correct model constructor, checkpoint format, and output keys. Do not guess these.
3. Write the three scripts.
4. Run each one on the 3 test images. Record the JSON output from each run.
5. Compare the counts across methods. Look at the overlays for each image, and note where chairs are still detected as heads.

## Test Images

Place up to 3 auditorium images in `rnd_crowd/images/`. If the ground-truth count is known for an image, record it in `rnd_crowd/ground_truth.json` like this:

```json
{"auditorium1.jpg": 120, "auditorium2.jpg": 85}
```

## Results to Report

For each image and each method, report:

- Estimated count
- Ground-truth count, if available
- Absolute error
- A short note on visible false positives (chair backs, empty seats, or other objects detected as heads)

Put the results in `rnd_crowd/results.md` as a single table, followed by a few sentences on which method looked most reliable on this domain and why.

## Notes

- If a model's weights cannot be downloaded or a repo does not run, skip that approach and note the reason in `results.md`. Do not block the other approaches on it.
- Post-processing (thresholds) can be tuned on the 3 images, but say clearly in the results that this is tuning on the test set, so the numbers are not presented as a fair benchmark.
- Stop after the six approaches above (LWCC counts as two: CSRNet and DM-Count). Do not add other models unless asked.