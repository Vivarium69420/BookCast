# BookCast on a VPS

This deploys the real Gradio studio — uploads, cast analysis, rendering, Activity, and downloads — behind Caddy with HTTPS and HTTP basic auth.

The static marketing site can stay on GitHub Pages. Put the app on a subdomain such as `app.example.com`.

## What you need

- A VPS: **2 vCPU / 2 GB RAM minimum**, 4 GB if you plan to use Kokoro or a local LLM. Audio files add up, so start with 20 GB+ disk.
- A domain or subdomain you can point at the VPS.
- Ports **80** and **443** open. Do **not** expose port 7860.
- Docker Engine with the Compose plugin.

## 1. DNS

Create an `A` record:

```text
app.example.com  ->  YOUR_VPS_IPV4
```

Start with **DNS only** (no Cloudflare orange cloud) until Caddy has issued the certificate. Caddy handles HTTPS automatically once ports 80/443 reach the VPS. Stop any existing web server on ports 80/443 first.

## 2. Install Docker

Use Docker's official instructions for your distribution. On Ubuntu, the short version is:

```bash
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

Then add Docker's repository and install `docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin` using the current instructions at <https://docs.docker.com/engine/install/>. Verify:

```bash
docker --version
docker compose version
```

## 3. Clone and configure

```bash
sudo mkdir -p /opt/bookcast
sudo chown "$USER":"$USER" /opt/bookcast
git clone git@github.com:Vivarium69420/BookCast.git /opt/bookcast
cd /opt/bookcast

cp .env.example .env
chmod 600 .env
```

Edit `.env`:

```dotenv
DOMAIN=app.example.com
BASIC_AUTH_USER=bookcast
BASIC_AUTH_HASH='$2a$14$...'
TZ=UTC
```

Generate the password hash:

```bash
docker run --rm caddy:2-alpine \
  caddy hash-password --plaintext 'choose-a-long-password'
```

Paste the output as `BASIC_AUTH_HASH`, keeping the single quotes. The app has **no built-in user accounts**, so basic auth is the minimum protection for a public IP.

## 4. Create persistent data and start

```bash
cd /opt/bookcast
./deploy/setup.sh
```

Run it as a user in the `docker` group. If your user is not in that group, use `sudo ./deploy/setup.sh` (or add the user with `sudo usermod -aG docker "$USER"` and log in again).

The script creates:

```text
data/books/   uploaded EPUB/PDF files
data/runs/    casts, chunks, progress, logs, rendered audio
data/models/  optional Kokoro models
```

Those directories are mounted into the container. They are the only data you need to back up.

If you prefer to do it manually:

```bash
mkdir -p data/books data/runs data/models
sudo chown -R 10001:10001 data
docker compose up -d --build
docker compose ps
```

The app container runs as UID/GID `10001`. If you skip the `chown`, uploads and renders will fail with permission errors.

## 5. Verify

```bash
docker compose ps
docker compose logs -f caddy
docker compose logs -f app
```

Open:

```text
https://app.example.com
```

You should get a browser basic-auth prompt, then the BookCast studio.

Once the studio is reachable, point the public landing page CTAs at it:

```js
// site/assets/config.js
window.BOOKCAST_CONFIG = {
  repoUrl: "https://github.com/Vivarium69420/BookCast",
  appUrl: "https://app.example.com"
};
```

Commit and push `site/` to redeploy GitHub Pages. With `appUrl` empty, the CTAs keep going to the quickstart section instead.

## 6. Start on boot

Compose already has `restart: unless-stopped`, but a systemd unit makes the whole stack start reliably after boot:

```bash
sudo cp deploy/bookcast.service /etc/systemd/system/bookcast.service
sudo systemctl daemon-reload
sudo systemctl enable --now bookcast
sudo systemctl status bookcast
```

Adjust `WorkingDirectory` in the unit if you cloned somewhere other than `/opt/bookcast`.

## Firewall

Only SSH, HTTP, and HTTPS should be public:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 443/udp
sudo ufw enable
```

Do **not** add `7860`; Caddy reaches it over the private Compose network.

## Sharing from the VPS

The studio's **Phone** share button can build a public link when you set **Base URL** to `https://app.example.com`. The **Detect** button may report a container or private address, which is expected behind Caddy; override it manually. The link is still protected by the same basic-auth prompt.

## Sizing the render queue

The studio is network-bound on Edge TTS, but a small VPS still needs conservative limits:

- **Parallel renders:** 1 on a 2 GB VPS, 2 on a 4 GB VPS
- **TTS concurrency:** 4–6
- Keep `parallel renders × TTS concurrency` around 12 or less

Rendering roughly 11× realtime was measured on a 45-chapter book at concurrency 6, but that assumes a stable network and enough CPU for ffmpeg. Start low, watch `docker stats`, and raise only if the VPS is comfortable.

## Updating

Wait for running jobs to finish or cancel them (finished chunks are kept), then:

```bash
cd /opt/bookcast
./deploy/backup.sh
git pull --ff-only
docker compose up -d --build
docker compose ps
```

Builds are cached, so only changed dependencies/code cause a long rebuild.

## Backups

```bash
cd /opt/bookcast
./deploy/backup.sh
```

This writes `backups/bookcast-data-<timestamp>.tar.gz` containing books, casts, progress, and audio. Copy backups off the VPS. They may contain copyrighted books and personal audio, so protect them.

Restore:

```bash
cd /opt/bookcast
tar -xzf backups/bookcast-data-<timestamp>.tar.gz
sudo chown -R 10001:10001 data
docker compose restart app
```

## Security notes

- The app is designed as a **single-user studio**, not a multi-tenant service. Basic auth protects one shared workspace; it does not isolate users from each other. Do not hand out accounts until per-user workspaces and ownership checks exist.
- Use a long, unique password and keep `.env` at mode `600`.
- Keep the host and Docker updated.
- If you use Cloudflare in front, set SSL/TLS to **Full (strict)** after Caddy has a certificate.
- Only upload books you have the right to convert.

## Troubleshooting

**Caddy keeps restarting:** check `.env` exists and `DOMAIN`, `BASIC_AUTH_USER`, `BASIC_AUTH_HASH` are set. Run `docker compose logs caddy`.

**Certificate does not issue:** make sure the DNS record points at the VPS, ports 80/443 are open, and the DNS record is not proxied yet.

**Uploads or renders fail with permission errors:** run `sudo chown -R 10001:10001 data` and retry.

**The app is healthy but Caddy waits:** `docker compose ps` should show `app` as healthy. The first start can take up to 90 seconds.

**Everything is slow:** lower TTS concurrency, check free disk with `df -h`, and check memory with `docker stats`.
