# AGENTS.md

## Project

Standalone live 7-microphone array direction estimator

## Repository

- Source repository path: `C:\simul`
- Vault path: `C:\Vault\Codex`
- Original source artifact: `code.txt`

## Scope

- Keep changes inside `C:\simul` unless a future prompt explicitly says otherwise.
- Treat this as a standalone local project.
- Preserve `code.txt` unless a future prompt explicitly authorizes editing it.
- Keep the microphone array at 7 microphones.
- The project estimates direction of arrival from one array only.
- Do not claim or visualize precise 3D position, distance, triangulation, target lock, or impact point.
- Do not add launcher, firing, projectile, weapon automation, GPIO, Arduino, serial, servo, or real hardware-control logic.
- Do not add GUI, web app, API, deployment files, or external services unless explicitly scoped.
- Do not claim real-world accuracy or hardware validation without evidence.

## Commands

- Self-check: `python uma8_music.py --self-check`
- Simulated direction estimation: `python uma8_music.py --simulate`
- Simulated direction visualization: `python uma8_music.py --visualize --simulate --duration 3`
- Optional live estimate: `python uma8_music.py --live`
- Optional live visualization: `python uma8_music.py --visualize --live`
- Validation commands: `python -m py_compile .\uma8_music.py`; `python -m unittest`

## Vault Write-Back

Vault write-back is forbidden unless a future prompt provides an explicit Obsidian write-back target and rule.
