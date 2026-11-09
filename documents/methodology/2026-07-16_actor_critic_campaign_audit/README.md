# Actor-critic campaign audit, 2026-07-16 to 2026-07-17

Working material of the audit that preceded the H1 campaign and the AAAI-2027
abstract. The refutation audit is the source of
`../2026-07-17_campaign_launch_blockers.md`; the briefing checks the abstract
notes in `documents/paper/2026-07-17_*` against the artefacts.

| File | Content |
|---|---|
| `AAAI_ABSTRACT_BRIEFING.md` | Audit of the claims prepared for the abstract against the artefacts in `data/`: the record of the actor-critic, which H1 verdicts are measured and which are not |
| `CELL_SELECTION_OPTIONS.md` | Decision material for choosing the dynamical systems of the experiments |
| `check_cell_viability.py` | Viability gate: one seed per campaign cell, asking whether any learned controller reduces the closed-loop cost |
| `measure_clip_binding.py`, `measure_divergence_firing.py` | Measurements of when the clipping and divergence guards bind |
| `maths-vs-plumbing-refutation-wf_23fe6377-3da.js`, `refutation_journal_wf_23fe6377-3da.jsonl` | Workflow and journal of the maths-versus-plumbing refutation audit |
| `policy-gradient-estimator-audit-wf_312b8f77-06c.js`, `pg_audit_journal_wf_312b8f77-06c.jsonl` | Workflow and journal of the policy-gradient estimator audit |
| `audit_items_union.json` | Union of the audit items of both workflows |

Provenance: these files were committed by mistake to the author's personal
repository `claude-workspace-rho` (commit `5e55392`, 2026-09-12) and moved here
on 2026-11-09. Paths inside them are relative to this repository's root.
