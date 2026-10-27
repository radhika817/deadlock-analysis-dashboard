# Curator

**Every conclusion, traced to its evidence.**

AI-assisted security investigation for SOC analysts: correlates alerts and logs,
reconstructs attack timelines, maps them to MITRE ATT&CK, and links every generated
sentence to the raw log line that supports it.

- `system_design.md` — architecture, data model, API contracts, conventions
- `DESIGN.md` — interface specification, authoritative for anything visual

**Status: Step 1 of 7.** Three containers, the full schema, `GET /health`, and a
placeholder page. No parsers, pipeline, or AI yet.

---

## Setup on Kali Linux (rootless Docker)

### 1. Install Docker Engine, Compose v2, and the rootless extras

Kali is Debian-based; use Docker's Debian `bookworm` repository.

```sh
sudo apt update
sudo apt install -y ca-certificates curl make openssl uidmap dbus-user-session slirp4netns
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin \
  docker-compose-plugin docker-ce-rootless-extras
```

Compose must be the v2 plugin (`docker compose`), not the legacy `docker-compose` binary.

### 2. Switch to rootless mode

```sh
# Your user needs subordinate ID ranges (Kali normally creates them).
grep "^$(whoami):" /etc/subuid /etc/subgid

# Stop the rootful daemon so the setup tool doesn't refuse, and so nothing talks to it by accident.
sudo systemctl disable --now docker.service docker.socket

dockerd-rootless-setuptool.sh install
systemctl --user enable --now docker
sudo loginctl enable-linger "$(whoami)"     # keep the daemon running after logout

docker context use rootless                 # created by the setup tool
```

Optional but recommended: membership in the `docker` group is root-equivalent on a
rootful socket. Once you are on rootless, remove it with `sudo gpasswd -d "$(whoami)" docker`.

Confirm:

```sh
docker info --format '{{.SecurityOptions}}'   # must include name=rootless
docker compose version                        # v2.x or later
```

### 3. Start Curator

```sh
make up
```

On the first run this creates `api/.env` from `.env.example` with a random
`POSTGRES_PASSWORD`, builds the API and web images, starts all three containers, and
returns once all three pass their healthchecks.

- Web: <http://localhost:5173> (shows `connected` when the API and database are up)
- API: <http://localhost:8000/health>

Add `ANTHROPIC_API_KEY` to `api/.env` before Step 4. Nothing calls the Anthropic API
in Step 1, and `DRY_RUN=true` stays the default until you change it.

---

## Make targets

| Target           | What it does                                                          |
| ---------------- | --------------------------------------------------------------------- |
| `make up`        | Build and start everything, detached; waits until all three are healthy |
| `make down`      | Stop and remove containers; the database volume is kept               |
| `make reset`     | `down`, **delete the database volume**, `up` — re-runs `db/init/`     |
| `make seed`      | Load the dataset (not implemented yet — Step 2)                       |
| `make eval`      | Run the accuracy harness (not implemented yet — Step 6)               |
| `make logs`      | Follow logs from all three services                                   |
| `make shell-api` | Bash inside `curator-api`                                             |
| `make psql`      | `psql` inside `curator-db`, using the container's own credentials     |

## Services

| Service       | Image / base              | Host port          | Notes                                        |
| ------------- | ------------------------- | ------------------ | -------------------------------------------- |
| `curator-db`  | `pgvector/pgvector:pg16`  | none               | volume `pgdata`; `db/init/` runs on first boot |
| `curator-api` | `python:3.11-slim`        | `127.0.0.1:8000`   | FastAPI + uvicorn `--reload`; `./api` mounted |
| `curator-web` | `node:20-alpine`          | `127.0.0.1:5173`   | Vite dev server; `./web` mounted             |

## Notes

- **Ports bind to 127.0.0.1 only**, so the dev stack isn't exposed on shared (e.g.
  hackathon) Wi-Fi. To reach it from another device, change the `ports:` entries in
  `docker-compose.yml` to drop the `127.0.0.1:` prefix.
- **Postgres has no host port.** Kali's own PostgreSQL (used by `msfdb`) often holds
  5432. Use `make psql`.
- **`db/init/` runs only on an empty volume.** For schema changes, add a numbered file
  (`002_*.sql`) and run `make reset`. The same applies to changing `POSTGRES_*` in
  `api/.env`: Postgres stores the credentials when it initialises the volume.
- **`api/.env` is the only real env file** and is gitignored. `.env.example` is the
  committed template.
- **Web dependencies** live in the image, not on the host. `make up` passes
  `--renew-anon-volumes`, so the container picks up changes to `package.json`.
- Under rootless Docker, container root maps to your user, so anything the containers
  write into `./api` or `./web` stays owned by you.

## Troubleshooting

| Symptom                                               | Cause and fix                                                            |
| ----------------------------------------------------- | ------------------------------------------------------------------------ |
| `env file .../api/.env not found`                     | You ran `docker compose` directly before the first `make up`. Run `make up`, or `cp .env.example api/.env`. |
| `permission denied ... /var/run/docker.sock`          | The CLI is pointed at the rootful socket. Run `docker context use rootless`. |
| Page shows `database unreachable`                     | The API is up but can't reach Postgres. `make logs`, check `curator-db`. |
| `password authentication failed` after editing `api/.env` | The volume was initialised with the old password. `make reset`.      |
| `port is already allocated` on 8000 or 5173           | Something else is listening: `ss -ltnp \| grep -E ':(8000\|5173)'`.        |
