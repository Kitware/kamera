# Per-system deployment configs

`<system>/config.json` (taiga, nayak) is the singular source of truth for a
deployment's static configuration. More static configuration should be
handled with compose.yml files and .launch files.

It is read in two ways, both of which expect plain JSON:

- shell: `kamera-cfg <query>` (`src/run_scripts/inpath/kamera-cfg`), a
  plain `jq -r <query> config.json`
- python: `kamcore/scripts/seed_redis_config.py` and `wxpython_gui.cfg`
  load it and seed the static keys into redis (config.json always wins for
  its keys; operator-mutable session state stays owned by the GUI)

These notes used to live as comments inside `config.yaml` and moved here
when the file became JSON.

## Field notes

**Fields of view.** FOV names describe the position of the camera array's
reference frame: `{left, right, center, NONE}` (use `NONE` with
`ROLE=NONE`). Looking along the direction of flight:

```
 Port     Starbrd
       _
      /^\
    x | | x
 <==U=| |=U==>
      | |
      \ /
     <<V>>
 [L]  [C]  [R]   Fields of View
```

**`.arch`** — camera information and configuration. `max_frame_rate` is the
max fps we can request of the cameras. `max_mpix: 200000` is 0.2e6.

**`.interfaces`** — maps channels to ethernet interface names (this is
going to usurp the locations mapping).

**`.locations`** — maps a DEV_ID to a specific FOV. Use this section to
swap which physical cameras map to a software FOV.

**`.devices`** — configuration for each physical device, keyed by DEV_ID
(e.g. `ir_n0`). A DEV_ID is an abstract label with no absolute correlation
to FOV; that indirection makes it easy and less error-prone to reroute
camera defs. Device blocks appear in groups: cooled FLIR cameras, IR
"backup" cameras, rgb prosilica backup cameras, rgb phase one cameras, uv
prosilica cameras. `model` names an entry in `.models`, `prefer_ip` is the
address the camera driver connects to, and `mac` identifies the physical
unit.

**`.models`** — the sensor resolution (`specs`) of each camera model, keyed
by model name. The GUI sizes its image panels from it.

**`.launch`** — expert configs; launch params passed through to the camera
launch files.
