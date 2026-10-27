# Experimente starten und auswerten (FR3, MuJoCo)

Diese Anleitung beschreibt den **Experiment-Workflow** (Stand 2026-10-26). Hintergrund zur Regelung,
zu Gains und zur Reibungsproblematik steht in [`README.md`](README.md); der wissenschaftliche Plan in
[`RESEARCH_PLAN.md`](RESEARCH_PLAN.md).

**Was verifiziert ist und was nicht:** Alles unten Beschriebene wurde in einer Linux-Umgebung
(Python 3.11.15, MuJoCo 3.17.0) ausgeführt. **Unter Windows wurde nichts davon getestet.**
Die Windows-Befehle sind aus der Linux-Struktur abgeleitet; Abschnitt 7 listet wahrscheinliche
Probleme.

---

## 1. Installation unter Windows (PowerShell)

Voraussetzungen: Python 3.11 (3.12 ist ungetestet), Git. Kein WSL, kein Docker nötig.
Die Modelle (FR3, S2NNDS-Sine) liegen bereits im Repository unter `vendor/`.

```powershell
git clone https://github.com/philma17/lfd-sim-pic.git
cd lfd-sim-pic
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Falls `Activate.ps1` wegen der Ausführungsrichtlinie blockiert wird: entweder
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (gilt nur für dieses Fenster) oder ohne
Aktivierung direkt `.\.venv\Scripts\python.exe ...` statt `python ...` verwenden.

Die Installation (v. a. `torch`) lädt einige hundert MB. Wenn die unpinned Versionen Probleme machen:
`python -m pip install -r requirements-tested.txt`.

**Beim allerersten Start** lädt `pyLasaDataset` einmalig den LASA-Datensatz aus dem Internet
(GitHub) und legt ihn im Benutzerverzeichnis ab. Dafür wird eine Internetverbindung gebraucht;
danach läuft alles offline.

### Installation prüfen

```powershell
python -B -m unittest discover -s tests
```

Erwartet (so in der Linux-Umgebung): `Ran 35 tests`, **ein Fehler** `test_vendored_files_unchanged`
und **ein Skip** `test_start_and_servo_match_baseline_results`. Beide sind bekannt und erklärt
(Abschnitt 8). Dauer ca. 20–40 s. Weitere Fehler wären ein echtes Problem (siehe Abschnitt 7).

---

## 2. Ein Befehl: nominaler FR3-Lauf

```powershell
python -B experiments/run_experiment.py nominal_fr3
```

Das lädt `configs/experiments/nominal_fr3.yaml` (erbt von `configs/picds.yaml`), führt
**Positions-Servo** und **PIC-DS** auf dem unveränderten Menagerie-FR3 aus, speichert Daten und Plots
und schreibt Berichte. Dauer ca. 10 s.

Erwartete Konsolenausgabe (in Linux gemessen; Zeiten hängen vom Rechner ab):

```
batch 2026….._nominal_fr3  ->  …\results\runs\2026….._nominal_fr3
  running nominal / servo ...
    converged       final    0.0002 mm  t_reach 6.21  peak applied tau/lim 0.307  sat steps 0  wall 1.0s
  running nominal / pic ...
    stalled         final   94.1185 mm  t_reach None  peak applied tau/lim 0.220  sat steps 0  wall 5.5s
runs 2, errors 0, finished but not converged 1
```

Das ist **das erwartete Ergebnis, kein Fehler**: Es reproduziert die historischen Werte
(`EXPERIMENT_LOG.md`, `results/pic/metrics.json`): Der Servo konvergiert (2.0e-7 m, 6.21 s), PIC-DS
bleibt mit den Paper-Gains durch die Haftreibung des Menagerie-FR3 am Start stehen (94.1 mm).
Zum Vergleich ohne Reibung (DIAGNOSTISCHER Fall, nicht die Baseline):

```powershell
python -B experiments/run_experiment.py diagnostic_frictionless
```

Hier konvergieren beide (Servo 6.128 s, PIC-DS 6.924 s, wie historisch).

**Exit-Codes:** `0` = alle Läufe fertig und gespeichert (auch wenn ein Lauf nicht konvergiert ist, das ist ein
Ergebnis); `1` = mindestens ein Lauf ist mit einem Fehler abgebrochen (`error.txt` im Laufordner);
`2` = fehlerhafte Konfiguration/Kommandozeile (z. B. Tippfehler im Schlüssel); `3` = nur mit `--strict`: ein Lauf
hat nicht konvergiert.

---

## 3. Was wird gespeichert?

Jeder Aufruf legt einen **neuen** Ordner an, nichts wird überschrieben (auch `results/pic/` bleibt unberührt):

```
results/runs/<Zeitstempel>_<experiment>/
    batch_report.md  batch_summary.csv  batch.json     Überblick aller Läufe
    compare/…                                           Vergleichsplots (siehe unten)
    <bedingung>__<regler>/                              z. B. alpha2.0__pic
        manifest.json     Herkunft: Git-Commit (+ "dirty"), Hashes von Config/Modell/Code,
                          Bibliotheksversionen, Startkonfiguration, Seed, angewendete Physikparameter
        config_resolved.yaml   die tatsächlich verwendete Konfiguration
        metrics.json      Kennzahlen + Status (failure_class)
        traj.npz          alle Zeitreihen, volle Auflösung (maßgebliche Messdaten)
        trajectory.csv    dieselben Zeitreihen, jede 5. Probe (10 ms), z. B. für Excel
        report.md         Kurzbericht, plots/*.png
```

`manifest.json` enthält `"status": "incomplete"`, solange ein Lauf nicht fertig ist: ein abgebrochener
Lauf bleibt damit als unvollständig erkennbar. Ein Lauf braucht ca. 4–6 MB, `results/runs/` wächst also;
alte Ordner kannst du selbst löschen.

### Gespeicherte Größen (`traj.npz`, Zeilen = Zeitpunkte, `t` in s)

| Schlüssel | Bedeutung | Art |
|---|---|---|
| `q`, `qd` | Gelenkpositionen [rad], -geschwindigkeiten [rad/s] | gemessen (MuJoCo-Zustand) |
| `tcp`, `tcp_ref`, `goal` | TCP-Position, ideale DS-Referenz (gleiches α), Ziel [m] | gemessen / berechnet |
| `err_sync` | zeitsynchroner Fehler `‖p(t) − p_ref(t)‖` [m] | berechnet |
| `err_geom` | Abstand zum idealen *Pfad* (zeitunabhängig) [m] | berechnet |
| `dist_goal` | Abstand zum Ziel [m] | berechnet |
| `tau_cmd`, `tau_app` | **angeforderte** bzw. **angewendete** Gelenkmomente [Nm] | gemessen/geloggt |
| `tau_limit`, `sat_joint`, `sat_any` | Momentgrenzen, Sättigungsflags (`|tau_cmd| > Limit`) | berechnet |
| `tcp_vel` | TCP-Geschwindigkeit `J_p(q)·q̇` [m/s] | exakt aus Messwerten |
| `sigma_min_Jp`, `manipulability_Jp` | Konditionierung des Positions-Jacobi | berechnet (nur Kinematik) |
| `d_tcp_acc`, `d_tcp_jerk`, `d_qdd`, `d_qddd` | Beschleunigung / Ruck | **abgeleitet (numerisch)** |
| `pic_s`, `pic_z`, `pic_beta_R`, `F_task`, … | PIC-DS: Tank, Kopplung, Kraft | Reglergrößen |
| `q_des`, `q_gap` | Servo: Sollposition und `q_des − q` | Reglergrößen |

Die **letzte Zeile** von `tau_cmd`/`tau_app` und der PIC-Reglergrößen (`z`, `beta_*`, `F_task`) ist
**NaN**: Im Endzustand wird kein Regelschritt mehr berechnet. NaN heißt „nicht vorhanden", nicht 0.

### Beschleunigung und Ruck: Methode und Grenzen

Beschleunigung und Ruck werden **nicht gemessen**. Sie entstehen durch Savitzky-Golay-Differentiation
(Fenster 0.05 s, Polynomgrad 3) der gemessenen Geschwindigkeit (`pic/analysis.py`). Das glättet
numerisches Rauschen, schwächt aber echte schnelle Vorgänge ab und ist an den Rändern und bei
Haft-Gleit-Übergängen unzuverlässig. Der Start-Transient (erste 0.1 s, z. B. Servo-Sag-Sprung)
dominiert die Spitzenwerte; deshalb gibt es `…_peak_after_startup_…` ohne diese Zeit. Ruck ist ein
qualitativer Indikator, keine Präzisionsmessung. Die Parameter stehen im Abschnitt `analysis:` der YAML.

### Plots (automatisch pro Lauf, `plots/`)

`01_tcp_3d` (3D-Pfad + Draufsicht, Start/Ziel/Referenz) · `02_tracking_error` · `03_joints`
(mit Gelenkgrenzen) · `04_kinematics` (Geschwindigkeit exakt, Beschleunigung/Ruck abgeleitet, log-Skala) ·
`05_torques` (angefordert vs. angewendet, ±Limit, rot = Sättigung) · `06_controller` (PIC: Tank/Kopplung/
Kräfte, Servo: Soll-Ist-Lücke, σ_min des Jacobi).
Vergleichsplots (`compare/…`, `cmp_01…04`): Legende zeigt nur die Parameter, die sich unterscheiden;
gemeinsame Parameter stehen im Titel.

---

## 4. Eigene Versuche, ohne Python zu ändern

**a) Einzelwert auf der Kommandozeile ändern** (gilt für alle Bedingungen der Datei):

```powershell
python -B experiments/run_experiment.py nominal_fr3 --set ds_speed_factor=2.0
python -B experiments/run_experiment.py nominal_fr3 --set plant.dof_frictionloss_scale=0.5 --set sim.T_max=15
python -B experiments/run_experiment.py nominal_fr3 --controllers servo
python -B experiments/run_experiment.py nominal_fr3 --tag mein_test
```

Tippfehler werden abgefangen (`pic.lamda` → Fehler, Exit 2) statt stillschweigend ignoriert.

**b) Eigene YAML anlegen**, am einfachsten durch Kopieren von `configs/experiments/speed_scaling.yaml`:

```yaml
extends: picds.yaml            # Basisparameter
sim: {T_max: 20.0}             # überschreibt nur diesen Wert
plant: {dof_frictionloss_scale: 0.0}
experiment:
  name: mein_versuch
  controllers: [servo, pic]    # servo | pic | pic_pulse
conditions:                    # jede Bedingung = ein Satz Überschreibungen
  - {label: langsam, set: {ds_speed_factor: 0.5}}
  - {label: schnell, set: {ds_speed_factor: 2.0, plant.torque_limit_scale: 0.5}}
```

Starten mit `python -B experiments/run_experiment.py configs/experiments/mein_versuch.yaml`.
Labels müssen eindeutig sein. Nur ein paar bewusst kontrollierte Parametersätze, kein Sweep-Framework.

### Konfigurierbare Parameter

| Schlüssel | Bedeutung | nominal |
|---|---|---|
| `ds_speed_factor` | α in `ẋ = α f(x)`, α > 0 | 1.0 |
| `plant.dof_frictionloss_scale` | Faktor auf Coulomb-Gelenkreibung (1.137/0.763/0.44/0.248 Nm) | 1.0 |
| `plant.dof_damping_scale` | Faktor auf viskose Gelenkdämpfung (0.21 Nms/rad) | 1.0 |
| `plant.gravity_scale` | Faktor auf die Erdbeschleunigung | 1.0 |
| `plant.mass_scale` | Faktor auf alle Linkmassen **und** -trägheiten | 1.0 |
| `plant.inertia_scale` | zusätzlicher Faktor nur auf die Trägheitstensoren | 1.0 |
| `plant.torque_limit_scale` | Faktor auf die dokumentierten Limits 87/87/87/87/12/12/12 Nm | 1.0 |
| `task.q_start_rad` | explizite Startkonfiguration (7 Werte) statt Multi-Start-IK | nicht gesetzt |
| `sim.T_max`, `sim.dt`… | Simulationsdauer usw. (`dt` muss 0.002 bleiben = MuJoCo-Zeitschritt) | 10 s |
| `pic.lambda`, `pic.s0`, `pic.null_space.d_q`, … | PIC-DS-Parameter (siehe `configs/picds.yaml`) | Paper-Werte |
| `analysis.*` | Filterfenster, CSV-Stride, Stall-Schwelle | siehe `pic/analysis.py` |

Welche MuJoCo-Größen genau geändert werden, steht in `pic/plant.py`; das MJCF wird nie verändert, alle
Änderungen gelten nur im Speicher, und `manifest.json → plant.applied` enthält die **tatsächlich im Modell
stehenden Werte**.

---

## 5. Zwei Bedingungen vergleichen (kleines Beispiel)

Der Ordner `results/compare/…_pic_friction0_vs_1/` entstand so (nach `friction`-Lauf):

```powershell
python -B experiments/run_experiment.py friction
python -B experiments/compare_runs.py results/runs/<batch>/fric0.0__pic results/runs/<batch>/fric1.0__pic --name pic_friction0_vs_1
```

`compare_runs.py` rechnet nichts neu. Es legt `results/compare/<Zeit>_<name>/` mit Overlay-Plots und
`comparison.md` an. Ergebnis dieses Beispiels: PIC-DS ohne Reibung konvergiert (6.924 s), mit
Nominalreibung `stalled` (94.1 mm). Mehrere Bedingungen **derselben** Konfiguration vergleicht der Runner
bereits selbst (`compare/` im Batch-Ordner). Plots eines alten Laufs neu erzeugen:
`python -B experiments/plot_run.py <laufordner>` (schreibt nach `plots_regenerated/`).

---

## 6. Ergebnisse interpretieren

**`failure_class`** (in `metrics.json`, `batch_report.md`):

| Klasse | Bedeutung |
|---|---|
| `converged` | Endabstand < 1 % von L = 1 mm bei `T_max` |
| `stalled` | nicht konvergiert **und** TCP-Geschwindigkeit in den letzten 0.5 s < 1 mm/s: der Roboter steht (z. B. Haftreibung) |
| `timeout_moving` | nicht konvergiert, aber noch in Bewegung bei `T_max` (zu langsam, oder weggelaufen/gefallen, Plots ansehen!) |
| `nonfinite` | numerisch ungültig |

Die Klassen sind eine grobe, feste Schwelle und keine Ursachendiagnose. Ob ein Lauf z. B. wegen Sättigung
oder wegen Reibung scheitert, zeigen `n_saturated_steps`, `05_torques.png` und `peak_cmd_torque_ratio`.

**Wichtige Unterscheidungen:**
- `err_sync` (zeitsynchron) bestraft auch reines Zu-Spät-Sein, `err_geom` misst nur die Pfadabweichung.
  Ein DS ist zeitfrei; für „weicht der Pfad ab?" ist `err_geom` die richtige Größe.
- `tau_cmd` kann das Limit weit überschreiten (beim Servo wächst der Soll-Wert `q_des` weiter, solange er
  saturiert); `tau_app` ist das, was die Physik tatsächlich sieht (auf das Limit begrenzt).
- Die Referenz `tcp_ref` ist der ideale DS-Rollout **mit demselben α und derselben Startposition**,
  nicht mit dem Roboter-Ist-Zustand neu verankert.
- **`T_max` konstant halten**, wenn α variiert: Bei α = 0.5 braucht der ideale DS ca. 12 s. Mit `T_max = 10`
  würde er nur am Zeitbudget scheitern (`speed_scaling.yaml` nutzt deshalb 20 s für alle α).
- Ein einzelner Lauf ist eine **Beobachtung**, keine Allgemeinaussage (siehe `AGENTS.md`/`RESEARCH_PLAN.md`).
  Diagnostische Bedingungen (Reibung 0) sind nicht die Baseline.
- PIC-DS und Servo unterscheiden sich in Schnittstelle (Drehmoment vs. Positionssollwert). Das Verhalten ist
  Folge von Regler **und** Plant-Gains. Es ist kein Reglerwettbewerb.

---

## 7. Fehlersuche

| Symptom | Wahrscheinliche Ursache / Abhilfe |
|---|---|
| `Activate.ps1 … cannot be loaded` | Ausführungsrichtlinie: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` oder `.\.venv\Scripts\python.exe` direkt aufrufen |
| `ModuleNotFoundError: mujoco/torch/…` | venv nicht aktiv oder `pip install -r requirements.txt` nicht gelaufen (Prompt zeigt `(.venv)`?) |
| Fehler beim ersten Start beim Laden von LASA | Internet nötig für den einmaligen `pyLasaDataset`-Download; Proxy/Firewall prüfen |
| `UnicodeEncodeError` | Neue Dateien werden als UTF-8 geschrieben; Fehler bei eigenen Dateien: UTF-8 speichern (Umlaute in YAML-Kommentaren vermeiden, `common/scenario.py` liest ohne Encoding) |
| `configuration error: unknown config key …` (Exit 2) | Tippfehler im Schlüssel; die Meldung listet die gültigen Schlüssel |
| `sim.dt … != model timestep` | `sim.dt` muss 0.002 bleiben |
| `non-finite qpos` / Lauf `nonfinite` | Instabile Parameter (extreme Skalierungen); Plots/Manifest ansehen; Lauf bleibt als Fehler dokumentiert |
| `mj_fullM` o. ä. `TypeError` | MuJoCo-API-Änderung: `requirements-tested.txt` (MuJoCo 3.17.0) verwenden |
| Plots fehlen, Lauf ok | Matplotlib läuft headless (`Agg`), benötigt kein Display; Fehlertext in `error_compare.txt` bzw. `error.txt` |
| Pfad zu lang (Windows, 260 Zeichen) | Repository nahe am Laufwerkswurzelverzeichnis klonen (z. B. `C:\src\lfd-sim-pic`) |
| PIC bewegt sich nicht (`stalled`, 94 mm) | **kein Bug**: Haftreibung > Regelkraft, siehe README „Gains and friction" |

---

## 8. Stand der Physikvariationen (FR3)

„Getestet" = Unit-Test in `tests/test_experiment.py` prüft, dass sich die **Physik** ändert.
„Gelaufen" = durchgängiger Lauf mit Daten/Plots; die Ergebnisse stehen in `EXPERIMENT_LOG.md`.

| Variation | Status |
|---|---|
| A. Geschwindigkeit α (0.5 / 1 / 2) | **funktioniert**: getestet + gelaufen (`speed_scaling`). α skaliert Reglergeschwindigkeit und Referenz-Rollout, die Geometrie bleibt gleich. |
| B. Reibung: ×0 / ×1 / ×2 (Coulomb), Dämpfung | **funktioniert**: getestet + gelaufen (`friction`). Bei ×1 und ×2 steht PIC-DS am Start (Haftreibung). |
| C. Drehmomentgrenzen (1.0 / 0.5 / 0.25) | **funktioniert**: getestet + gelaufen (`torque_limit`). Angefordert/angewendet/Sättigung wird für beide Regler geloggt. Servo: Limit via MuJoCo-`forcerange`; PIC-DS: Clipping. |
| D. Gravitation, Masse, Trägheit | **funktioniert technisch**: getestet + je ein Lauf (`physics_probe`, Machbarkeitsprobe, kein Studienergebnis). |
| D. COM-Verschiebung, Armatur, Gravitationsrichtung | **nicht implementiert** (Erweiterungspunkt in `pic/plant.py`: `model.body_ipos`, `dof_armature`, `opt.gravity`). |
| D. Mehrere Startkonfigurationen | **Schnittstelle vorhanden** (`task.q_start_rad`), getestet (Validierung, Übernahme); kein Mehrfachstart-Lauf durchgeführt. Der DS-Startpunkt bleibt der Demo-Mittelwert; abweichende Startposen werden als TCP-Offset in `start_ik` protokolliert. |
| D. Singularitäten | Konditionierung (`sigma_min_Jp`) wird geloggt; **kein** Singularitäts-Experiment gemacht. |
| Störpuls (`pic_pulse`) | nur für PIC-DS (die Baseline-Servofunktion hat keinen Kraft-Hook). |

**Zur PIC-Gravitationskompensation:** Sie verwendet immer das **Nominalmodell** des Roboters. `gravity_scale`
und `mass_scale` ≠ 1 sind für PIC-DS daher *nicht modellierte* Last bzw. Gravitation. Der Servo hat keine
Kompensation. (Beobachtung im Probelauf: PIC-DS konvergiert dann nicht, der Servo schon. Als Erklärung ist
nur die *Hypothese* belegt, dass ein reiner DS-Dämpfungsregler ohne Positionssteifigkeit eine
Restgravitation nicht ausregelt; sie wurde nicht separat geprüft.)

**Bekannte Abweichung (Provenance):** `common/ds.py` und `common/robots.py` weichen von den in
`common/VENDORED.txt` eingetragenen Hashes ab (Referenz-Commit `a06ed55`). Der Diff gegenüber `a06ed55`
(per `git show`, Referenzrepo unverändert) besteht nur aus Pfad-Defaults (`S2NNDS_DIR`, `MENAGERIE_DIR` →
`vendor/…`) und der Konstante `MENAGERIE_COMMIT` (`d7b481b…` → `cda5236…`). Andere vendorte Dateien sind
hash-identisch. Der Test zeigt nur die erste abweichende Datei (`ds.py`), `robots.py` weicht ebenfalls ab. Hashes
wurden **nicht** angepasst; der Vorschlag steht im Abschlussbericht und braucht deine Freigabe. Beachte außerdem:
das eingebettete Menagerie-FR3-Modell (`cda5236`) ist nicht der Commit der Baseline (`d7b481b`); ein
Modellunterschied wurde nicht untersucht, die historischen Zahlen wurden aber reproduziert.

---

## 9. UR5e und weitere Aufgaben

**UR5e: nicht integriert**, aber geprüft. In einem Wegwerfskript (nicht im Repo) mit Menagerie-Revision
`f3b6013` liefen die *unveränderten* Funktionen: Modell lädt, `position_servo_map` verifiziert 6 PD-Positionsservos,
Startkonfiguration ist identisch mit der Baseline (`LfD_baseline@a06ed55`), Servo konvergiert (6.858 s,
Spitzenmoment 0.12 des Limits) und PIC-DS mit den **unvalidierten** FR3-Gains konvergiert ebenfalls
(6.298 s, Nullraum-Dimension 3, kein Clipping). Das ist ein Ausführungsnachweis für **einen** Lauf, keine
Validierung, und er ist nicht reproduzierbar abgelegt. Für eine echte Integration fehlt:
1. UR5e-Modell vendoren (`vendor/mujoco_menagerie/universal_robots_ur5e`, Lizenz beachten) und
   `Menagerie`-Revision gegen die Baseline klären,
2. `build_case` (FR3-Sperre, `TAU_MAX_DOC["FR3"]`, Robotername in `experiment.py`) auf `cfg["robot"]` verallgemeinern,
3. Das UR5e-MJCF hat **keine** Gelenkreibung/-dämpfung (`frictionloss = 0`, `damping = 0`): die multiplikativen
   Reibungsfaktoren wären wirkungslos, es bräuchte absolute Werte,
4. Gains/Nullraum (`d_q`) für den UR5e (3-dim. Nullraum statt 4) validieren, UR5e-Momentgrenzen (150/28 Nm sind im Modell
   bereits erzwungen) gegen das Datenblatt prüfen.

**Weitere Erweiterungspunkte** (nicht implementiert): Orientierungsaufgaben (`Target("axis5"/"full6")`
existiert in `common/ik.py`, PIC-DS ist nur Position), weitere Startpositionen/Multistart (über Bedingungen mit
`task.q_start_rad`), 3-DoF-Modell, 2D/3D-Vergleiche (`check_dims` unterstützt 3D-Embedding), PIC-Zerlegungen
`linear`/`lyapunov`, Servo mit Störpuls.
