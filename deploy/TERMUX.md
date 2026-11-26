# BookCast on Android with Termux

Yes — Termux can host the studio as a website on your phone. Start it once,
open `http://127.0.0.1:7860` in Chrome, and add it to your home screen.

This is a **personal single-user setup**. The app has no built-in multi-user
isolation; use the optional basic auth and keep it on a trusted network.

## Which install route?

- **Native Termux** is smaller and starts faster. Gradio's dependency chain is
  the only tricky part; some Android builds need Rust/C tools.
- **proot-distro Ubuntu** is more reliable if native `pip install` fails. It
  uses the normal aarch64 Linux wheels and is the recommended fallback.

## 1. Native Termux

Install Termux from **F-Droid**, not the Play Store. Then:

```bash
pkg update
pkg upgrade
pkg install git
git clone https://github.com/Vivarium69420/BookCast.git
cd BookCast
./deploy/termux-setup.sh
```

If the setup script succeeds, start the studio:

```bash
./run_termux.sh
```

It prints something like:

```text
BookCast Studio
  This phone:  http://127.0.0.1:7860
  Same Wi-Fi:  http://192.168.1.42:7860
```

Open the first URL on the phone. To use another phone or laptop on the same
Wi-Fi, open the second URL.

The script also tries to open Chrome automatically. If Termux:API is installed,
it holds a wake lock so Android is less likely to kill a render when the screen
turns off.

### If native pip fails

Gradio can pull in Rust/C packages such as `pydantic-core` or `orjson`. Try:

```bash
pkg install -y rust clang make binutils
python -m pip install -r requirements-termux.txt
```

If that is slow or still fails, use the proot route below. Don't fight it.

## 2. proot-distro Ubuntu fallback

```bash
pkg install proot-distro
proot-distro install ubuntu
proot-distro login ubuntu
```

Inside Ubuntu:

```bash
apt update
apt install -y python3 python3-pip python3-venv ffmpeg git espeak-ng qrencode
git clone https://github.com/Vivarium69420/BookCast.git
cd BookCast
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-ui.txt
./run_ui.sh --host 0.0.0.0 --port 7860 --no-browser
```

Open `http://<phone-ip>:7860` from Chrome. The `run_termux.sh` helper is not
needed inside proot; it is for native Termux.

## 3. Add basic auth

The studio is open to anyone who can reach the port. For a shared Wi-Fi:

```bash
BOOKCAST_USER=bookcast \
BOOKCAST_PASSWORD='choose-a-long-password' \
./run_termux.sh
```

Then Chrome asks for that username/password. The same environment variables
work with `./run_ui.sh` and the VPS deployment.

## 4. Open it from another device

Same Wi-Fi:

```text
http://<phone-ip>:7860
```

Find the phone IP with:

```bash
ip -4 addr show wlan0
```

If it does not connect:

- Make sure both devices are on the same Wi-Fi.
- Some routers enable **AP/client isolation**; turn it off.
- Check that you started with `--host 0.0.0.0` (the helper does).
- Check Android's battery optimization has not killed Termux.

Remote access without port forwarding:

- **Tailscale** is the easiest. Install the Tailscale app on the phone and the
  other device, then open `http://<phone-tailscale-ip>:7860`.
- A temporary **Cloudflare Tunnel** works too, but it is not a permanent
  service and exposes the app without auth unless you set the env vars above.

Do **not** port-forward `7860` from your router.

## 5. Keep it alive

- Install the **Termux:API** app from F-Droid and run `pkg install termux-api`
  if you want the helper's wake lock.
- Run `termux-wake-lock` manually if needed.
- Disable battery optimization for Termux in Android settings.
- Keep the phone charging during long renders.
- Keep Termux in the foreground when possible. Android can still kill
  background apps on aggressive battery settings.

## 6. Storage

`termux-setup-storage` gives Termux access to shared storage, but BookCast keeps
its working data inside the repo:

```text
books/   uploaded books
runs/    casts, chunks, progress, logs, audio
```

Rendered books can be several hundred MB. Keep at least a few GB free. Download
finished files from the Activity panel; you do not need to dig through Termux
folders.

## 7. Troubleshooting

**`No module named gradio`** — run `./deploy/termux-setup.sh` again, or use the
proot route.

**`ffmpeg: command not found`** — `pkg install ffmpeg`.

**`Installing pip is forbidden`** — do not run `pip install --upgrade pip` in
native Termux. Termux patches pip and blocks self-upgrades. Pull the latest
`deploy/termux-setup.sh` and rerun it; it installs packages with the packaged
pip without upgrading it.

**`Address already in use`** — use another port:

```bash
PORT=7861 ./run_termux.sh
```

**Phone cannot connect from another device** — confirm the IP, the same Wi-Fi,
and that the app is bound to `0.0.0.0`.

**Termux dies during a render** — enable the wake lock, disable battery
optimization, and keep the charger connected.

**Native install keeps compiling Rust** — use proot Ubuntu. It is the more
predictable path for Gradio on Android.
