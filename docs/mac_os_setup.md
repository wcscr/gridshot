# macOS installation (Apple Silicon)

GridShot runs natively on Apple Silicon so PyTorch can use the Metal Performance
Shaders (MPS) backend. The Linux/NVIDIA deployment uses containers, but the macOS
runtime, Python environment, web build, and model caches live on the host.

This guide covers a fresh macOS installation. After setup, the same public
commands used on Linux select the native Mac implementation automatically:

```bash
scripts/up
scripts/gridshot --help
```

## Why GridShot runs natively on macOS

GridShot uses a custom PyTorch service for its inference models that is not
currently supported by Docker or Docker Model Runner in a way that makes it
straightforward to use MPS acceleration from within a container. GridShot on
Apple Silicon currently runs the local stack natively on macOS.

## Requirements

- An Apple Silicon Mac (`arm64`) running macOS 14 or later
- Internet access for dependencies and model weights
- Permission to install the Xcode command-line tools
- A supported Node.js LTS release with npm
- `uv` for Python and project dependency management
- A Hugging Face account with SAM 3 access if concept segmentation will be used
- Tailscale only if the app must be available to a phone over the tailnet

Docker Desktop is not required for the Mac runtime.

## 1. Verify the Mac and shell architecture

Run:

```bash
uname -s
uname -m
sw_vers -productVersion
```

The first two commands must report `Darwin` and `arm64`. GridShot rejects Intel
Macs, and an Apple Silicon Mac running an `x86_64` shell through Rosetta will not
use the supported native environment. Open a native arm64 Terminal session before
continuing.

PyTorch documents the `mps` device as its interface to Metal acceleration on
macOS: [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html).

## 2. Install the Xcode command-line tools

Install Apple's compiler, SDK, and Git command-line tools:

```bash
xcode-select --install
```

Complete the macOS installer dialog, then verify the installation:

```bash
xcode-select -p
git --version
```

The developer directory normally resolves to
`/Library/Developer/CommandLineTools`. See Apple's
[command-line tools installation guide](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools)
for other installation and update methods.

## 3. Install uv and Python

Install `uv` with its official standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new Terminal window if the installer updates the shell path, then verify:

```bash
uv --version
```

GridShot requires Python 3.12 or later. `uv` downloads a compatible arm64 Python
interpreter automatically when the project environment is created, so a separate
system Python installation is not required. To install Python 3.12 explicitly:

```bash
uv python install 3.12
```

See the official `uv` documentation for
[installation options](https://docs.astral.sh/uv/getting-started/installation/)
and [managed Python versions](https://docs.astral.sh/uv/guides/install-python/).

## 4. Install Node.js and npm

Install a currently supported Node.js LTS release for macOS on Apple Silicon from
the [official Node.js download page](https://nodejs.org/en/download). Node.js 22
or newer satisfies the checked-in web dependency metadata. The Node installer
includes npm.

Verify both the version and architecture:

```bash
node --version
npm --version
node -p 'process.arch'
```

The architecture must report `arm64`. Use an LTS release that the
[Node.js release schedule](https://nodejs.org/en/about/previous-releases) marks as
supported; do not use an end-of-life release.

## 5. Prepare the repository

Clone GridShot if necessary, then change into the repository root:

```bash
git clone https://github.com/aac-78/gridshot.git
cd gridshot
```

Confirm the public launch script detects the Mac implementation:

```bash
scripts/up --help
```

The usage text should describe the native macOS Metal service. There is no manual
virtual-environment activation step: `scripts/up` creates or updates `.venv` from
the checked-in lockfile.

For persistent model or runtime settings, create the local environment file:

```bash
cp .env.example .env
```

The native launcher reads `.env` automatically. Variables exported by the
invoking shell take precedence. The file is ignored by Git.

## 6. Configure SAM 3 access

Interactive SAM 2.1 segmentation does not require authentication. SAM 3 concept
segmentation loads on demand and uses a gated Hugging Face checkpoint. Without
SAM 3 access, automatic tool detection records a warning and falls back to a
less reliable CPU path when no empty-mat reference exists; interactive editing
is unaffected.

1. Request and accept access on the
   [SAM 3 model page](https://huggingface.co/facebook/sam3).
2. Authenticate from the same macOS user account that will run GridShot:

```bash
uvx hf auth login
uvx hf auth whoami
```

The official [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
documents browser login and token-based alternatives. For a temporary or
non-interactive session, export an authorized token before launch instead:

```bash
export HF_TOKEN=<token>
```

Do not commit tokens to the repository.

## 7. Start GridShot for the first time

Start on localhost without requiring Tailscale:

```bash
scripts/up --no-tailscale
```

On Apple Silicon, `scripts/up` delegates to the native launcher. It:

- loads persistent settings from `.env`, if present;
- installs the locked web dependencies and builds the web application;
- installs the locked Python dependencies into `.venv`;
- verifies that PyTorch selected `mps` with FP32 and CPU fallback disabled;
- creates `config/`, `projects/`, cache, log, and PID directories; and
- starts the web service on port 8800 and inference service on port 8801.

Open [http://localhost:8800](http://localhost:8800). Dependency installation can
take several minutes on the first run. Model weights download into
`config/cache/` when each model is first used, so the first readiness, capture,
or optional-model request can also take longer than later requests.

Verify the services from another Terminal window:

On a fresh installation, the first readiness request starts the SAM 2.1 weight
download. The web probe can return HTTP `503` because it times out after 30
seconds while that download is still in progress. This is expected; wait and
rerun the readiness command until it succeeds.

```bash
curl --fail http://127.0.0.1:8800/api/health/live
curl --fail http://127.0.0.1:8800/api/health/ready
curl --fail http://127.0.0.1:8800/api/health/capabilities
```

The web endpoint nests the inference report under `segserver`. The capabilities
response should report `segserver.runtime.selected` as `mps`,
`segserver.runtime.dtype` as `float32`, and
`segserver.runtime.mps_cpu_fallback_enabled` as `false`.

Stop the native services with:

```bash
scripts/down-macos
```

## 8. Register and verify a calibration mat

The public CLI wrapper also detects macOS automatically:

```bash
scripts/gridshot mat new --paper a4
scripts/gridshot mat verify <mat-id> --measured-x <mm> --measured-y <mm>
```

Use `a3` or `letter` instead of `a4` for another bundled mat size. The CLI stores
the registration in the same `config/` directory used by the web application.

## 9. Optional Tailscale access

Tailscale is not needed for workstation-only use. To reach GridShot from a phone,
follow the official [macOS installation guide](https://tailscale.com/docs/install/mac),
authenticate the client, and enable its
[CLI integration](https://tailscale.com/docs/reference/tailscale-cli?tab=macos)
so `tailscale` is available in the Terminal. Then launch without
`--no-tailscale`:

```bash
tailscale version
scripts/up
```

The launcher keeps both services bound to loopback and publishes only the web
service through Tailscale Serve.

## Developer validation

The real interactive SAM, SAM 3 checkpoint, and SAM 3 architecture Metal suites
are opt-in because they require Apple GPU access and may download model weights.
Their `GRIDSHOT_RUN_MPS_*` flags and exact commands are documented at the top of
[tests/test_segserver_mps.py](../tests/test_segserver_mps.py). Re-run the relevant
suite after changing inference dependencies, accelerator behavior, or pinned
model revisions.

## Troubleshooting

### A required command is missing

The native launcher checks `uv`, `npm`, `curl`, and `lsof`. Verify each command:

```bash
command -v uv
command -v npm
command -v curl
command -v lsof
```

`curl` and `lsof` are supplied by macOS; reinstall or update the external tool if
`uv` or `npm` is missing.

### MPS is unavailable

Confirm macOS and process architecture first, then query the installed PyTorch
build:

```bash
uname -m
.venv/bin/python -c 'import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'
```

Both values must be `True`. Ensure the Mac is Apple Silicon, macOS is current,
and the Terminal and Node/Python tools are arm64. GridShot intentionally sets
`PYTORCH_ENABLE_MPS_FALLBACK=0`; enabling CPU fallback can hide unsupported Metal
operations and is not the supported launch configuration.

### SAM 3 reports unavailable or access denied

Confirm access was granted for the exact Hugging Face account and run:

```bash
uvx hf auth whoami
```

Reauthenticate if needed, then retry the concept request. The interactive SAM 2.1
lane remains usable independently.

### Port 8800 or 8801 is already in use

Inspect the listener before stopping anything:

```bash
lsof -nP -iTCP:8800 -sTCP:LISTEN
lsof -nP -iTCP:8801 -sTCP:LISTEN
```

If an earlier GridShot launch owns the ports, run `scripts/down-macos`. The
launcher refuses to replace an unrelated process.

### A service exits during startup

Inspect the native logs:

```bash
tail -n 100 config/logs/web.log
tail -n 100 config/logs/segserver.log
```

## Maintenance

For storage cleanup, preview the project cleanup command before applying it:

```bash
scripts/prune --dry-run
```
