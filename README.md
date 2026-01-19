# Insta360 autostitcher
A small tool that monitors a "dump" directory for `*.insv` files and stitches them into a 360 video.

Note that this tool uses Insta360 MediaSDK, which requires you to apply for the SDK.

I am not a software engineer so the code might be bad. Use it at your own risk. PRs are welcomed :)

<img width="1162" height="1257" alt="image" src="https://github.com/user-attachments/assets/9ff865c8-8757-4a32-8043-907f63196ebe" />

## Usage
The backend service lives in `backend/`.

1. Get the MediaSDK by applying [here](https://www.insta360.com/sdk/home). Insta360 does not allow redistribution, so each user must download the `.deb` themselves.
2. Place the downloaded `libMediaSDK-dev_2.0-0_amd64_ubuntu18.04.deb` (or whichever version you received) somewhere under `backend/` (e.g. `backend/vendor/libMediaSDK-dev_...deb`) and set `MEDIA_SDK_DEB` in `.env` (or pass `--build-arg MEDIA_SDK_DEB=vendor/your-file.deb`) so Docker knows which file to copy during the build.
   - Start by copying `.env.example` to `.env` and adjust the paths/ports for your environment.
3. Build the docker images (from the repo root):
```bash
docker compose build
```
4. Run the stack
```bash
docker compose up
```

> The `MEDIA_SDK_DEB` build arg (wired through `.env` and `docker-compose.yml`) must point to the relative path of the Insta360 SDK `.deb` within the `backend/` directory. If you keep the file at `backend/vendor/libMediaSDK-dev_2.0-0_amd64_ubuntu18.04.deb`, set `MEDIA_SDK_DEB=vendor/libMediaSDK-dev_2.0-0_amd64_ubuntu18.04.deb` in `.env` before running `docker compose build`.

All runtime data (SQLite DB, RAW uploads, stitched output, logs, thumbnails) now lives under a single storage root controlled by the `APP_STORAGE_DIR` environment variable (defaults to `/app`). In Docker we bind-mount the host paths from `.env` (`APP_STORAGE_DIR`, `RAW_DIR`, `OUT_DIR`) into `/app`, `/app/raw`, and `/app/stitched` respectively, while thumbnails are saved under `/app/thumbnails`.

The compose stack now starts two services (both running on the host network, so the chosen ports must be free on the host):
- `backend` (Flask REST API + stitching worker + thumbnail generator) listening on `${AUTO_STITCHER_PORT}` (default 8000)
- `frontend` (Vite React build served via nginx) exposed on `${FRONTEND_PORT}` (default 3000 via `NGINX_PORT`) and reverse-proxying `/status` + `/tasks` to `${BACKEND_API_URL}`. Browsers only need to reach the frontend host; nginx forwards API calls over WireGuard/LAN to the backend.

For production builds the frontend now defaults to relative API calls (so `VITE_API_BASE` can be left empty). The proxy target is controlled via `BACKEND_API_URL` in `.env`, and must be reachable from the frontend host (e.g. `http://10.253.0.8:8000`). For local development you can still override `VITE_API_BASE` when running `npm run dev` if you want the browser to talk directly to another controller instance.

For local testing you can point the backend to the built-in fixtures by overriding the storage dir:

```bash
APP_STORAGE_DIR=backend/test/appdata python backend/auto-sticher.py --storage-dir backend/test/appdata serve
```

If you run the controller directly outside of Docker you can still override the RAW and stitched directories via `--raw-dir` / `--output-dir`, but inside the container they are fixed to `/app/raw` and `/app/stitched`. Generated thumbnails live under `/app/thumbnails`; trigger them via the “Generate Thumbnails” UI button or the `generate_thumbnails` CLI/REST action.

## Web dashboard
A minimal React + TypeScript dashboard lives in `frontend/` to show job progress and trigger scans or stitch runs without touching the CLI.

The job table supports pagination, sorting, and multi-select checkboxes. Select one or more rows and click “Stitch Selected” to queue only those jobs; thumbnails (generated via the backend action) are displayed alongside each entry when available.

Backend stitching respects a configurable parallelism limit (default `1`). Use the “Settings” button in the UI (or POST to `/settings/parallelism` with `{"max_parallel_jobs": N}`) to update it live; the value is persisted in SQLite and can also be seeded via the `MAX_PARALLEL_JOBS` environment variable.

### Getting started
```bash
cd frontend
npm install
npm run dev
```

The dev server proxies `/status` and `/tasks` to `http://localhost:8000` by default. If your controller runs elsewhere, start the UI with `VITE_API_BASE=http://your-controller:8000 npm run dev` (or set the same variable when building for production).
