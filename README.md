<h1 align="center">GRID•SHOT</h1>

<p align="center"><strong>Calibrated phone photos → print-ready Gridfinity tool bins.</strong></p>

<p align="center">
  Local-first, GPU-accelerated capture for single tools, resumable batches, and a
  reusable tool library—without sending workshop photos to a hosted service.
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#batch-processing-and-tool-library">Batch + library</a> ·
  <a href="#product-status">Product status</a> ·
  <a href="#operations">Operations</a>
</p>

<p align="center">
  <img src="./assets/readme/gridshot-capture-flow.gif" width="720"
       alt="GridShot workflow showing two calibrated photos, RTX 5090 processing, outline review, and a generated Gridfinity tool bin">
</p>

<p align="center"><sub>Real GridShot capture: two photos → reviewed outline → generated 3D bin.</sub></p>

> [!IMPORTANT]
> GridShot is an accuracy-focused working prototype. Its software regression suite
> is extensive, but the full retained physical accuracy matrix is not complete yet.
> See [Product status](#product-status) for the current validation boundary.

## From photo evidence to printable geometry

<table>
  <tr>
    <td width="50%" align="center">
      <a href="./assets/readme/07-selection-editor.png">
        <img src="./assets/readme/07-selection-editor.png" width="100%" alt="GridShot photo-selection editor with calibration and readiness evidence">
      </a>
    </td>
    <td width="50%" align="center">
      <a href="./assets/readme/08-generated-bin.png">
        <img src="./assets/readme/08-generated-bin.png" width="100%" alt="GridShot result showing the corrected silhouette and generated 3D tool bin">
      </a>
    </td>
  </tr>
  <tr>
    <td valign="top"><strong>Review the evidence.</strong><br>Inspect calibration, segmentation, cleanup, and the accepted physical outline before generation.</td>
    <td valign="top"><strong>Generate from one canonical outline.</strong><br>Preview the exact bin, adjust its geometry, and export 3MF, STL, SVG, or GLB.</td>
  </tr>
</table>

## Why GridShot is different

| Focus | What GridShot does |
| --- | --- |
| Calibrated measurement | Uses a verified ChArUco mat plus camera/lens profiles instead of treating a visually plausible trace as physical truth. |
| Height-aware correction | Separates silhouette-driving height, maximum tool height, and desired recess depth. A calibrated second photo can solve thickness automatically. |
| Local ownership | Runs the web app, segmentation models, projects, and tool library on your own machine. |
| Conservative automation | Calibration, segmentation, pairing, thickness, and generation can pass, request review, or block. Uncertain batch matches remain unmatched. |
| Reproducible geometry | Retains source photos, calibration, accepted outlines, settings, and printer compensation so a bin can be regenerated rather than retraced. |

## How it works

```text
phone photo(s) → calibrated silhouette → reviewed physical outline
                → canonical bin geometry → STL / 3MF / SVG / GLB
```

1. Print and verify the calibration mat.
2. Photograph one tool, or take a calibrated two-view pair to solve thickness.
3. Refine the segmentation in the shared photo editor.
4. Review or edit the reconstructed physical cutout.
5. Generate a bin immediately or save the tool for later composition.

The same accepted outline can produce three bin styles:

| Style | Result |
| --- | --- |
| **Pocket** | A conventional solid bin with a recessed tool cavity. |
| **Stackable corral** | A lighter bin with a thin tool shelf, full-height separator, standard feet, and stacking lip. |
| **Live grid** | The complete corral plus usable Gridfinity sockets wherever a full 42 mm socket safely fits. |

## Quick start

### Requirements

- Tailscale CLI (optional) for the phone-friendly HTTPS endpoint used by the launch scripts

Choose one accelerated runtime:

**Apple Metal (initial support):**

- Apple-silicon Mac running macOS 14 or later
- Arm64 Python 3.12, [`uv`](https://docs.astral.sh/uv/), Node.js, and npm
- Xcode command-line tools (`xcode-select --install`)

**NVIDIA CUDA:**

- Docker with Compose and NVIDIA Container Toolkit
- NVIDIA Ampere-generation or newer GPU with at least **8 GB VRAM**

This fork validates the native Apple Metal/MPS path. The upstream CUDA path and
instructions are retained for compatibility but have not been revalidated here.

For the retained CUDA deployment, the 8 GB minimum covers the core SAM 2.1
interactive capture workflow. **12 GB or more is recommended** when using SAM 3
concept segmentation or RoMa dense matching. The current container runs inference
in BF16; CPU-only and pre-Ampere GPUs are not supported configurations.

The native Mac launcher runs MPS in FP32 with PyTorch CPU fallback disabled. It
keeps the interactive SAM 2.1 model resident, admits at most one optional model at
a time, evicts an optional model after five idle minutes on the next inference
request, and limits cached image embeddings to eight entries and an estimated
256 MiB. SAM 3 concept segmentation is enabled by default. RoMa is disabled by
default on Metal and remains an explicit opt-in because that path has not yet been
validated on MPS.

GridShot uses GPU 0 by default. On a multi-GPU host, choose a different NVIDIA CDI
device for the current launch:

```bash
GRIDSHOT_GPU_DEVICE=nvidia.com/gpu=1 scripts/up --no-tailscale
```

### 1. Create and verify a calibration mat

Ready-to-print calibration mats are included for each supported paper size:

| Paper | Mat ID | Download |
| --- | --- | --- |
| A4 | `a4-7x9-5c8f11` | [PDF](calibration-mats/gridshot-mat-a4-7x9-5c8f11.pdf) |
| A3 | `a3-10x14-030735` | [PDF](calibration-mats/gridshot-mat-a3-10x14-030735.pdf) |
| US Letter | `letter-7x9-7dd4fa` | [PDF](calibration-mats/gridshot-mat-letter-7x9-7dd4fa.pdf) |

Register the selected board in your local GridShot configuration before verifying
it. This also generates the same board in the ignored `mats/` runtime directory:

```bash
scripts/gridshot mat new --paper a4
scripts/gridshot mat verify <mat-id> --measured-x <mm> --measured-y <mm>
```

Replace `a4` with `a3` or `letter` when using another bundled size.

Print the selected mat at **100% / Actual Size**. Measure the marked X and Y spans
with calipers and record both values. An unverified mat cannot be used for capture.

### 2. Start GridShot

On an NVIDIA workstation:

```bash
scripts/up
```

On an Apple-silicon Mac, run the web and inference processes natively so
PyTorch can reach Metal:

```bash
scripts/up-macos
```

Stop the native services with `scripts/down-macos`. Run native CLI commands with
`scripts/gridshot-macos`; both use the same `config/`, `projects/`, and loopback
inference service as the web application.

SAM 3 is a gated Hugging Face model. Before the first concept-segmentation
request, obtain access from the [SAM 3 model page](https://huggingface.co/facebook/sam3)
and run `hf auth login`, or export an authorized `HF_TOKEN`. The native launcher
keeps model files in the project cache through `HF_HUB_CACHE` without redirecting
Hugging Face's normal credential store.

The reduced random-weight SAM 3 architecture smoke test passes on MPS with CPU
fallback disabled. Full pinned-checkpoint endpoint validation remains a release
gate until the test account has model access; run it with
`GRIDSHOT_RUN_MPS_SAM3_TESTS=1` after access is granted.

For a workstation-only deployment without Tailscale:

```bash
scripts/up --no-tailscale
# or on macOS
scripts/up-macos --no-tailscale
```

The launch scripts build and start the web and segmentation services, then expose
the web app through Tailscale:

- `http://localhost:8800` on the workstation
- `https://<host>.<tailnet>.ts.net/` from a tailnet-connected phone

`--no-tailscale` does not invoke the Tailscale CLI or change any existing Tailscale
Serve configuration.

### 3. Capture a tool

Take two photos from different camera positions for automatic thickness recovery,
or use one photo and enter the height at the tool's widest silhouette. The result
screen provides the corrected 1:1 outline, an orbitable 3D preview, and manufacturing
downloads.

## Batch processing and tool library

GridShot accepts a ZIP containing one or two photos per tool. Batch jobs are bounded,
cancellable, resumable, and checkpointed. Before anything enters the library, the
review screen shows proposed pairs, outlines, thickness results, warnings, and
readiness.

- Commit every reviewed tool, or explicitly commit only the ready subset.
- Keep unresolved tools as a draft for correction or recapture.
- Leave ambiguous image pairs unmatched instead of guessing.
- Retry safely without creating duplicate library entries.

Accepted tools become reusable local library records. Each record keeps the original
photo silhouette, corrected physical footprint, vertical measurements, calibration,
printer profile, settings, warnings, and non-destructive outline revisions.

From the library you can regenerate one tool, combine several tools into one bin, or
compose bins across a drawer. Cards, previews, and exports all use the same canonical
derivation path.

<table>
  <tr>
    <td width="50%" align="center">
      <a href="./assets/readme/10-batch-review-overlays.png">
        <img src="./assets/readme/10-batch-review-overlays.png" width="100%" alt="GridShot batch review with mask overlays, paired tools, and unmatched photos">
      </a>
    </td>
    <td width="50%" align="center">
      <a href="./assets/readme/14-drawer-preview.png">
        <img src="./assets/readme/14-drawer-preview.png" width="100%" alt="GridShot tool library with three selected tools and an exact 3D drawer preview">
      </a>
    </td>
  </tr>
  <tr>
    <td valign="top"><strong>Batch without guessing.</strong><br>Review mask overlays, proposed pairs, unmatched photos, thickness, and readiness before committing tools.</td>
    <td valign="top"><strong>Reuse accepted tools.</strong><br>Regenerate a bin, combine several tools, or compose separate bins across an exact drawer grid.</td>
  </tr>
</table>

## Printer compensation

GridShot can fit cavity compensation for a specific printer, material, nozzle, and
process. Print three independent copies of the long-baseline coupon, then repeat each
measurement option once per copy:

```bash
scripts/gridshot bench coupon --copies 3
scripts/gridshot bench record \
  --printer X1C --material PLA --nozzle-mm 0.4 --process standard \
  --a-x 124.3 --a-x 124.4 --a-x 124.3 \
  --a-y 24.7 --a-y 24.8 --a-y 24.7 \
  --b-x 24.7 --b-x 24.8 --b-x 24.7 \
  --b-y 7.7 --b-y 7.8 --b-y 7.7
scripts/gridshot bench printer-profiles
```

Profiles are immutable revisions. Measurements with borderline uncertainty are kept
for diagnosis but cannot become active; severe disagreement is rejected.

## Operations

GridShot stores capture sessions and generated artifacts in `projects/`. Calibration,
printer profiles, the tool library, and downloaded model caches live in `config/`.
Both directories are created automatically by the launch scripts; back them up before
moving or upgrading a deployment.

Copy `.env.example` to `.env` for persistent Docker/CUDA settings. The native
macOS launcher accepts the same settings as exported environment variables, for
example `HF_TOKEN=... scripts/up-macos`. Use `scripts/prune --dry-run` to preview
cleanup of old capture projects.

| Service | Port | Accelerator | Role |
| --- | ---: | --- | --- |
| `web` | `8800` | None | FastAPI, the React SPA, and the public API boundary |
| `segserver` | `8801` internal/loopback | CUDA or Metal/MPS | SAM 2.1 interactive segmentation, SAM 3 concept segmentation, and dense matching |

Runtime probes are exposed through the web service:

| Probe | Meaning |
| --- | --- |
| `/api/health/live` | The web process is alive; performs no dependency work. |
| `/api/health/ready` | Storage and interactive segmentation are ready for capture traffic. |
| `/api/health/capabilities` | Detailed model state, verified mats, and inference-queue usage. |

GPU inference is serialized with two waiting slots by default. Set
`GRIDSHOT_INFERENCE_QUEUE_SIZE` to change the queue capacity. Saturated requests fail
quickly with HTTP `429` and `Retry-After` instead of building an unbounded backlog.

Native macOS unified-memory controls are available as exported environment
variables:

| Setting | Native default | Meaning |
| --- | --- | --- |
| `GRIDSHOT_ENABLE_SAM3` | `1` | Enable the on-demand SAM 3 concept lane. |
| `GRIDSHOT_ENABLE_ROMA` | `0` | Install and enable the unvalidated RoMa Metal lane. |
| `GRIDSHOT_OPTIONAL_MODEL_RESIDENCY` | `single` | Keep only one optional model resident; `multi` opts out. |
| `GRIDSHOT_OPTIONAL_MODEL_IDLE_SECONDS` | `300` | Evict an idle optional model on the next inference request; `0` disables idle eviction. |
| `GRIDSHOT_EMBED_CACHE_MAX_ITEMS` | `8` | Maximum cached interactive image embeddings. |
| `GRIDSHOT_EMBED_CACHE_MAX_MIB` | `256` | Estimated combined tensor and decoded-image cache budget. |

`/api/health/capabilities` reports each optional lane as `disabled`,
`unavailable`, `not_loaded`, `ready`, `evicted`, or `error`, along with residency,
embedding-cache, and MPS allocator telemetry. To experiment with RoMa on Metal,
run `GRIDSHOT_ENABLE_ROMA=1 scripts/up-macos`; the launcher then installs the
separate `matcher` dependency extra. This does not imply that RoMa has passed the
Metal validation gate.

## Product status

Implemented today:

- Verified mat and immutable camera-profile calibration
- Interactive segmentation and non-destructive correction editors
- One-photo and calibrated two-photo capture
- Pocket, stackable-corral, and live-grid geometry
- Resumable batch review and fail-closed library commits
- Persistent tool library, multi-tool composition, and drawer export
- Versioned printer compensation and reproducible artifact provenance

Still gated on real evidence:

- The retained physical G1 accuracy matrix
- Production matcher thresholds selected and validated on a representative GridShot capture corpus
- Published first-print-fit and recapture-rate claims

## License

GridShot is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md)
for personal and other noncommercial use. Commercial use, commercial services, and
commercial products require a separate written license from the copyright holder.
