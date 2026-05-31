# DTS-Simulation Repository Instructions

## Project

- Project name: `DTS-Simulation`
- Source-code repository path: `C:\simu`
- Obsidian vault path: `C:\Vault\Codex`
- Obsidian project documentation folder: `C:\Vault\Codex\02-Projects\DTS-Simulation`

## Repository Commands

- Run command: `python -m dts_simulation`
- Typecheck command: unavailable - no typecheck tool configured
- Test command: `python -m unittest discover -s tests -v`
- Build command: unavailable - Python package scaffold has no build command required yet

## Sensitive Files

- No secrets known.
- `.env`, credential files, service keys, and other secret material remain forbidden.

## Vault Context Policy

When running inside this source-code repository, Codex may read explicitly referenced Obsidian vault files for context. Vault wiki links do not grant permission to read extra vault files; Codex must not follow them unless the current prompt explicitly permits and bounds it.

Codex must not modify the vault unless the current prompt provides a specific write-back target. If write-back is allowed, Codex may only append or update the specified section in the specified file.

Project folders under `02-Projects/` are documentation and workflow folders only. They are not application source-code repositories.

## Allowed Write-Back Behavior

- Default: vault writes are forbidden.
- Allowed only when the prompt names an exact vault file and write-back rule.
- Report every vault file read or changed.

## Scope Boundaries

- Team 1 source scope is one-array direction-of-arrival processing unless a later prompt expands it.
- Do not implement multi-array localization, final source-position estimation, graphical simulation, launcher/Nerf behavior, real hardware control, broadband drone handling, or multi-bin fusion unless explicitly authorized.
- Do not claim real-world accuracy or validation from simulation-only tests.

## Final Report Format

```text
Status: DONE / PARTIAL / BLOCKED

Summary:
- ...

Files inspected:
- ...

Vault files read:
- ...

Code files changed:
- ...

Vault files changed:
- ...

Validation performed:
- ...

Documentation updated:
- ...

Assumptions made:
- ...

Risks or unresolved issues:
- ...

Recommended next action:
- ...
```
