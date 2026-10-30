# Credits and licences

What this project is built on, and what each part's licence allows. Checked 2026-10-07 by reading each source's own
repository. Not legal advice.

**This project is licensed AGPL-3.0-or-later** (`LICENSE`, chosen 2026-10-07). No car or track data ships with it.

## Software this project links to or was ported from

| what | whose | licence | how it is used here |
|---|---|---|---|
| [AssettoServer](https://github.com/compujuckel/AssettoServer) | Niewiarowski, compujuckel | AGPL-3.0, plus "you must preserve the legal notices and author attributions present in the server". Extra permission only for Steamworks and for plugins published on the AssettoServer Patreon. | `GradientDriftPlugin/` is compiled against it and loaded by it. `drift_rl/ai_spline.py` is a direct port of its `FastLaneParser`, `AiSpline` and `SplinePointOperations`; `drift_rl/fast_sim.py` re-implements the same lookups. |
| [KN5 Blender Tools](https://github.com/nazariyKul/kn5-blender-tools) | KN5 Importer contributors, 2026 | GPL-3.0-or-later (licence and the original MIT notice shipped as `tools/kn5_reader_LICENSE.txt`, `tools/kn5_reader_MIT.txt`) | `tools/kn5_reader.py` is its `kn5.py`, with two marked changes (runs on Python 3.7+, skips texture data). `tools/track_triangles.py` (ours) uses it. |
| [acd_extractor](https://github.com/bovis/acd_extractor) | Bovis | MIT (stated in its gemspec; no licence file) | `drift_rl/cars/acd_to_ini.py` follows its decryption and embeds its 160-byte test archive as a self-test (MIT notice in the file). |
| [Assetto Corsa Telemetry Recorder](https://www.overtake.gg/downloads/) v0.1.1 (OverTake, AC Apps) | markelitxu21, Dec 28, 2025 | **None stated.** Its terms of service say only "I am the sole creator". | Not in this repository. Recorded the human driving reference the simulator was compared against. |

## Images

* **Project Logo:** Generated using Google Nano Banana 2. (`docs/logo.jpg`, shown at the top of
  the README.)
* The AssettoServer logo in the README's AssettoServer badge is AssettoServer's own, used to link
  to that project.

## Libraries

**The trainer** (`drift_rl/`, run from source; installed by the user, not shipped here): numpy
(BSD-3-Clause), numba and numba-cuda (BSD-2-Clause), pandas (BSD-3-Clause), scipy
(BSD-3-Clause), matplotlib (Matplotlib licence, BSD-style), PyTorch (BSD-3-Clause). **The plugin**:
.NET 8 (MIT), and AssettoServer above.

**The packaged setup program** (`gradient-drift-setup`, built by `.github/workflows/build-setup.yml`)
carries these inside it, and ships each one's licence text beside the program (`licenses/`, listed
in `THIRD_PARTY_NOTICES.txt`, which the build writes; it refuses to build without them):

| what | licence | why it is in there |
|---|---|---|
| Python | PSF-2.0 | the program runs on it |
| PyInstaller's bootloader | GPL-2.0-or-later with the bootloader exception | starts the program |
| Qt for Python (PySide6, shiboken6) | LGPL-3.0 (its libraries left as separate, replaceable files) | the window (Windows build only) |
| ruamel.yaml | MIT | editing `extra_cfg.yml` with its comments kept |
| numpy, scipy, pandas (with python-dateutil, six, tzdata) | BSD-3-Clause (numpy and scipy also carry OpenBLAS, BSD-3-Clause, and the GCC Fortran runtime, GPL with the GCC Runtime Library Exception: in their licence files) | making a track's data |
| numba, llvmlite (with LLVM) | BSD-2-Clause; LLVM Apache-2.0 with LLVM exception | making a track's data (compiled code) |
| KN5 Blender Tools (`tools/kn5_reader.py`) | GPL-3.0-or-later | reading the track's model files |
| this project's own code, including `drift_rl/ai_spline.py` (ported from AssettoServer) | AGPL-3.0-or-later | |

The Linux build also carries system libraries from the Debian 11 machine it is built on (OpenSSL,
libffi, expat, zlib, bzip2, xz, ncurses, libstdc++), each under its own licence, listed in its
`THIRD_PARTY_NOTICES.txt`. GNU readline is kept out on purpose. Every licence above lets an
AGPL-3.0 program carry the work, given the notices.

## Game content (none of it is in this repository)

* **Assetto Corsa** - Kunos Simulazioni. The in-game Python API (`ac`, `acsys`) is the game's.
* **Tracks** the policy was trained or tested on:
  * **Spirit Peaks Raceway** - by Mitch Rosaaen ([RiceFieldGaming](https://www.facebook.com/RiceFieldGaming/)),
    version 1.0, 2020. Trained on, and the main test server.
  * **Drift Playground 2021** - by PJ Edulian, P3ngu1n and Project Kaido. The second test server;
    trained on since gen11 (the earlier policies drove it as a track they had never seen).
  * **Magione** - Kunos Simulazioni (ships with the game). Trained on.
  * **House of RWD** (`kb3k__House_of_RWD`, version 1.0) - by killball3000 (KB3K;
    [OverTake.gg](https://www.overtake.gg/downloads/kb3k__house_of_rwd.70754/),
    [Patreon](https://www.patreon.com/kb3kAssettocreations)), a recreation of the RC drift circuit
    the YouTube channel RC Supremacy built in real life. Trained on.
  * **RiverSide Drift** (version 1.0) - by fanolamas11
    ([OverTake.gg](https://www.overtake.gg/downloads/riverside-drift.51422/)). Trained on.
  * **Ridge Motorsports Park** (layout ridge-a) - a recreation of The Ridge Motorsports Park,
    Shelton, Washington. Trained on (gen10 and later).
  * **Klutch Kickers Drifters Paradise** (`adc_klutch_kickers_drifters_paradise`) - the official
    track at Emerald Coast Dragway, by ADC_VEGGIEBOI of The Aussie Drift Co
    ([Facebook](https://www.facebook.com/aussiedriftco)). Trained on (gen11 and later).
* **Cars** whose numbers the policy was trained and tested with (it drives any rear-drive car;
  these are the ones it learned in):
  * **Gravy Garage** street and drift cars: Miata, AE86, E36 Compact and Touring, E46, GS300,
    JZX90, JZX100 Mark II, S13s, S14s and 180SX.
  * **Kunos Simulazioni** cars from the game and its DLC, including the BMW Z4 Drift, M3 E92
    Drift, M3 E30 and E30 Drift, AE86 Drift and Tuned, Supra Mk IV Drift, MX-5 NA and ND, GT86,
    RX-7 Tuned, 370Z, Skyline R34, Mustang 2015 and M4.
  * **World Drift Tour** (2021): Nissan Silvia S15 and Toyota Supra A80.
  * **Euros**: the "twodogs" Corvette C6 V2.

  Nothing from them ships: the numbers read from their data files stay on the developer's
  machine, and each user extracts their own from the cars they have.

## Ideas and methods

* PPO - Schulman et al., "Proximal Policy Optimization Algorithms", 2017. The trainer is our
  own implementation.
* The drift line and what a run is judged on - Formula Drift's 2026 judging regulations and
  D1 Grand Prix's scoring criteria.
* Cai et al., "High-speed Autonomous Drifting with Deep Reinforcement Learning", 2020 (reward
  ideas); the analysis of Ken Gushi's drift in Stanford's MARTY (what a held drift looks like).
* The brush tyre curve (`RoundTireCurve`) - E. Fiala, "Seitenkräfte am rollenden Luftreifen",
  VDI-Zeitschrift 96, 1954; the form Stanford's drifting work uses (Hindiyeh and Gerdes, "A
  Controller Framework for Autonomous Drifting: Design, Stability, and Experimental Validation",
  ASME Journal of Dynamic Systems, Measurement, and Control, 2014). Its full-sliding angle is
  the car's own tyres.ini `FRICTION_LIMIT_ANGLE` plus `FLEX` x the tire's load.

