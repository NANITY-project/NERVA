# NERVA desktop overlay (Wayland / Hyprland)

A transparent, click-through, always-on-top surface with the full-body humanoid on it. Window tops are floors it can step onto.
It is driven by text lines on stdin, so a model, a skill layer or a human can control it. **It never touches the cursor or keyboard.**

```
sudo pacman -S wayland wayland-protocols      # Arch; build also needs g++ and make
make overlay                                   # or: cd overlay && make       -> overlay/build/nerva-overlay
./overlay/build/nerva-overlay --debug --demo walk --speed 4
# Hyprland autostart:   exec-once = /path/to/nerva-overlay      (layer namespace: "nerva-overlay", for layerrules)
make overlay-test                              # unit checks (ASan+UBSan) + integration checks (the Sway / AT-SPI ones need sway, grim, dbus, python3-gobject, at-spi2-core)
```

| Piece | How |
|---|---|
| Surface | `wlr-layer-shell`, OVERLAY layer, anchored to all edges, exclusive zone -1, no keyboard focus, **empty input region** (clicks pass through) |
| Render | CPU capsules into a `wl_shm` ARGB buffer, double-buffered, damage = changed rectangle only, no commit when nothing moved (~1% of a core idle, ~4% walking) |
| Perception | Hyprland request socket: only `j/monitors` and `j/clients` are ever sent. Class names only unless `--titles` |
| View | side view: `screen_x = W/2 + z*ppm`, `screen_y = H - floor_off - y*ppm`; the body faces +z (screen right); floor = bottom edge |
| Stays out of the way | hides and **pauses** while a window is in real fullscreen (games, video; `--over-fullscreen` to disable) and fades to a ghost when the pointer is near it (`--no-ghost` to disable). Both need Hyprland (`j/clients`, `j/cursorpos`) |
| Protocol | body commands (`HAND FOOT PELVIS FACE LOOK TORSO GRIP STAND BALANCE SURFACE`) plus `PERCEIVE STATUS SPEED W2PX PX2W QUIT`; replies `OK`/`ERR`, feedback as `EVENT` lines |

Commands are held until the surface knows the real output size; closing stdin does not stop the overlay (SIGTERM / `QUIT` does).

## What the model writes and reads
[`docs/ACTIONS.md`](../docs/ACTIONS.md) is the **action language** (`<click>838 351</click>`, `<find>save</find>`, `<hand>right 0.3 1.2 0.4</hand>`, ...), the observation format, the prompts, the behaviour table and the training-data format. `tools/nerva_actions.py` parses, validates (`lint`), compiles and renders it, and `Session` runs a model turn against a real overlay to record true observations. `tools/examples/episodes.jsonl` has 9 episodes recorded from the real thing.

## The AI pointer (off by default)
`--mouse` enables it, and even then **nothing works until you grant one window**. The model cannot grant: grants come only from the control
socket (`$XDG_RUNTIME_DIR/nerva-overlay.sock`, mode 0600), never from stdin.

```
nerva-overlay --mouse                         # starts "enabled but not granted"
nerva-overlay --ctl grant                     # grants the FOCUSED window   (also: --ctl revoke, --ctl status)
# Hyprland keybinds:
#   bind = SUPER SHIFT, G, exec, nerva-overlay --ctl grant
#   bind = SUPER SHIFT, R, exec, nerva-overlay --ctl revoke
```
A thin outline is drawn around the granted window the whole time (dim while it is not focused).

| Rule | |
|---|---|
| Where | only inside the granted window, minus an 8 px margin (title bar / resize border) |
| When | only while that window is **focused** and visible; the grant moves with the window |
| How long | `--grant-ttl` seconds (default 300), then the human must grant again |
| Ends by itself | window closes / hides / leaves the workspace, a fullscreen window appears, Hyprland unreachable, overlay exits |
| What | move, **left** click, scroll (1-5 notches). No keyboard, no typing, no drag; right click only with `--allow-right-click` |
| Never granted | password managers, polkit/auth prompts, keyrings, pinentry, ssh-askpass (class deny-list) |
| Limits | 4 clicks/s; pointer speed capped; every move verified against the real cursor |
| You win | if you touch the mouse, the AI's pointer pauses for 3 s and its click is refused |
| Device | a uinput device created at grant and destroyed at revoke; it has **no keyboard key bits** (two buttons + wheel + absolute position) |

Model commands: `MOUSE STATUS | MOVE px py | TOHAND left|right | CLICK [left] | SCROLL n` (screen pixels, same coordinates as `W2PX`).
`TOHAND` moves the pointer to where that hand's fingertip is drawn.

Setup (once): uinput needs permission, e.g. `/etc/udev/rules.d/99-nerva-uinput.rules`:
```
KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"
```
then add your user to `input`, run `sudo udevadm control --reload && sudo udevadm trigger`, and log in again.
**How the compositor maps an absolute virtual pointer onto your screens is not known in advance**, so at each grant the overlay probes
three positions, reads the real cursor back from Hyprland, fits the mapping, and puts your cursor back. If it cannot get a consistent
answer (device ignored, non-linear, cursor unreadable) it refuses to grant and says why.

`--pointer dry-run` simulates all of this without touching anything (used by the tests).

## Reading the screen (accessibility, `--a11y`, off by default)
The model can READ what is on screen through AT-SPI (the Linux accessibility bus), in **every window except the ones you hide**. Reading is
separate from acting: it may only *point* inside the one window you granted.

```
sudo pacman -S at-spi2-core python-gobject                   # Arch
nerva-overlay --a11y [--mouse]
nerva-overlay --ctl "hide discord"                           # the model can no longer read or even name that app (substring of its class)
nerva-overlay --ctl "unhide discord"      nerva-overlay --ctl hidden       # saved in ~/.config/nerva/perception.json (0600)
```
Model commands (replies arrive later as `A11Y <id> {json}` lines; the immediate reply is `OK a11y <id> queued`):

| Command | Answer |
|---|---|
| `A11Y TREE [target] [depth]` | the visible elements: `role`, `name`, `rect` and `c` (centre) **in screen pixels** (same as `MOUSE MOVE`), `st` (focused, checked, editable...), `act` (e.g. `click`), `text` (editable text, documents, terminals) |
| `A11Y TEXT [target]` | everything readable as text, in order, size-capped |
| `A11Y FIND <words> [IN target]` | elements whose name or text contains the words, with the centre to point at (default: all readable windows) |
| `A11Y FOCUSED [target]` | the element that has keyboard focus |
| `A11Y APPS` | which visible windows can be read at all (accessibility not enabled in an app = unreadable) |

`target` = `focused` (default for TREE/TEXT/FOCUSED), `granted`, `all`, a window address, or part of a window class.

- **Never read:** password fields (by role) and text fields *named* like a secret (password, token, PIN, API key...). Their position is shown, never their content.
- **Never readable, whatever the hide-list says:** password managers, authentication prompts, keyrings (same list as the pointer's deny-list).
- **Hidden apps** are reported as `(hidden)` geometry in `PERCEIVE` (no name, no title) and are refused everywhere, including as "the focused window". Only you change the list; the model has no command for it.
- The reader is a separate read-only Python process (`tools/a11y_helper.py`, tested to contain no call that clicks, types, sets values or edits text). A hung or crashed reader is killed and restarted; the overlay never waits on it.
- Rectangles come from the app's own accessibility geometry plus Hyprland's window position. If they sit a few pixels off (client-side shadows), correct with `--a11y-offset X Y`.

What is and is not readable depends on the *application*, not on this program: GTK apps work out of the box (tested with GTK3); Qt needs
`QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1`; Chromium/Electron need `--force-renderer-accessibility`; Firefox usually exposes its pages once the accessibility
bus is running. **Many terminals expose nothing** (kitty, foot, alacritty); VTE-based ones (GNOME Terminal etc.) do. Use `A11Y APPS` to see what you get.

## Surfaces
Only window tops the legs can reach count as floors (`max_step_up`, 0.45 m, from the height the foot stands on). A maximised window's top, far
above the body, is *not* a floor: the body walks in front of it on the desktop floor. (Before this, such a window made every step fail with
"step up of 9.27 m".) Windows are perches, not walls.

## Known limits
- **The real `/dev/uinput` path is compiled but untested** (no uinput in my environment): calibration, the pointer mapping and the click timing have only been tested against a simulated compositor. Try `--mouse` on a harmless app first (a text editor with the pointer on its sidebar).
- **Accessibility was tested with a real GTK3 window under a headless Sway** (positions match the real widgets exactly), not with Qt, GTK4, Firefox, Electron or on real Hyprland: coordinates for client-side-decorated windows may need `--a11y-offset`.
- The model sees text only through accessibility, no pixels (no vision model yet): anything an app does not expose (canvas, games, images of text) is invisible.
- **Not yet run on real Hyprland.** Tested against a fake Hyprland socket and a headless Sway (surface, transparency, input-region request, stdin protocol, demo walk). Hyprland's real window geometry (borders/gaps: use `--top-off`), real click pass-through and multi-monitor are unverified.
- One output only. On a scaled monitor the buffer is scale 1, so the compositor upscales it (slightly soft).
- Window occlusion is ignored: a window hidden behind another still offers its top. A window moved out from under the body is not handled specially.
- Walking is slow (0.31 m/s, the body's safe default) and only to the right in `--demo walk`; there is no turn-around yet. `--speed` fast-forwards time.
- Perception is polled every 250 ms (no event socket).
