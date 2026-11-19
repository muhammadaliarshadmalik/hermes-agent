# Team B decision log

## D2 — Evolving artifact

The candidate artifact is a `policy.yaml` containing instruction rules in each candidate directory. Candidate IDs come from directory names. `candidate-000` is the empty-rules baseline (`schema_version: 1`, `rules: []`). B1's runner accepts a candidate directory and loads it through an isolated per-task `HERMES_HOME`, injecting policy rule text into Hermes instructions without modifying Hermes source.

The proposed limits (30 rules, 400 characters per rule, 8 KB per policy file) remain assumptions pending explicit B1 confirmation.

## V1/V2 — Artifact versioning

Candidate directories are immutable snapshots. `evolve/versioning.py` hashes sorted relative paths and file contents with CRLF normalized to LF for configured text types. Candidate IDs use `candidate-NNN` (at least three digits). Sealing removes write permission as a guard; recorded hashes and `verify_snapshot` provide integrity checks. Verify a snapshot before evaluation.

## Shared ScoreReport contract

B1 is updating `interfaces/score_report.v1.schema.json`. Until that update lands, `MockScorer`'s report shape remains provisional and should be reconciled against the schema before the interface freeze.

## Open decisions

- Confirm policy rule limits and exact YAML validation requirements with B1.
- Decide whether all generated candidate snapshots, including rejected candidates, are committed to Git. The lineage/evaluation evidence should retain candidate hashes.
- Confirm shared model/provider settings and the agreed Hermes baseline commit before official measurements.

## Hermes loading verification

B1 identified `agent/skill_bundles.py` and `hermes_cli/config_home.py` as the loading path. Inspection confirms `agent/skill_bundles.py` reads bundle YAML from `<HERMES_HOME>/skill-bundles/` and uses its `instruction` text when invoking a bundle. `hermes_cli/config_home.py` initializes the supplied home directories and ensures the default `SOUL.md`; it does not itself resolve `HERMES_HOME` or load policy YAML. The runner must stage/inject the candidate policy through its isolated home or other agreed adapter path. B1 should confirm the exact mapping from `policy.yaml` rules into runner instructions.

## Baseline tag and diff

The local baseline tag is `baseline-hermes`, pointing at `d1b46d718ffe675427928eff7e5fff69bcb4519b`. Comparing `baseline-hermes..origin/main` shows only `TEAM_B_README.md` added.

## Scope and patch defaults

The guard and patch API do not name `policy.yaml`; they accept relative paths and configurable extension/size arguments. Their current defaults (`.yaml`, 8192 bytes) are the current B2 policy artifact assumptions, so callers should override them if B1 changes the candidate format or confirms different limits.