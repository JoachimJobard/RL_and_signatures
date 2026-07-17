# AAAI-2027 submission logistics — deadlines, requirements, eligibility

*Recorded 2026-07-17, read off the AAAI 2027 OpenReview submission form. Companion to
`2026-07-16_aaai_rl_paper_audit.md` (scope) and `2026-07-17_aaai_abstract_candidates.md` (abstract).
This file tracks **submission mechanics only** — nothing scientific.*

Venue: The Forty-First AAAI Conference on Artificial Intelligence, Montréal, 16 Feb 2027.
Submission runs through **OpenReview** (not EasyChair, not email).

---

## 1. Deadlines (authoritative — from the OpenReview form)

| Date (UTC) | Item | Note |
|---|---|---|
| Jun 30 2026 11:59**AM** UTC-0 | Submission start | open |
| **Jul 21 2026 AoE** | **Reciprocal Reviewer Nomination locks** | *"cannot be changed after Jul 21 AoE"* — irreversible. **Not a live hazard here** (see §4: no author qualifies) |
| **Jul 22 2026 11:59AM UTC-0** | **Abstract Registration** | Title + Authors + Abstract + Topics. **No PDF.** |
| **Jul 29 2026 11:59AM UTC-0** | **Submission Deadline** | PDF + Reproducibility Checklist + supplements, on the *same* record |

**In Paris time (CEST, UTC+2 — both dates fall inside EU summer time): 13:59, i.e. 1:59 PM.**
Computed, not assumed: `2026-07-22 11:59 UTC -> 2026-07-22 13:59 CEST`; `2026-07-29 11:59 UTC ->
2026-07-29 13:59 CEST`.

**The `11:59AM UTC` is the AoE convention, not a typo.** Anywhere-on-Earth is UTC−12, so the end of
21 July AoE *is* 22 July 11:59 UTC. Verified: `Jul 21 AoE == Jul 22 2026 11:59AM UTC -> True`, and
`Jul 28 AoE == Jul 29 2026 11:59AM UTC -> True`.

**Two earlier claims in this file were WRONG and are withdrawn:**
1. *"The briefing's 21 July / 28 July are one day early."* **False.** The briefing stated the
   deadlines in **AoE**; the form states the same instants in UTC. They agree exactly.
2. *"The reciprocal nomination lock is the earliest deadline, before the abstract."* **False.** It
   falls at the **same instant** as abstract registration (21 July AoE = 22 July 11:59 UTC).

**The flow is one form, two stages.** The `Abstract*` field is submitted by 22 July; the `PDF` field
is uploaded to the same submission by 29 July.

---

## 2. Desk-rejection risks (each is stated as such on the form)

1. **Reciprocal reviewer not nominated when a qualified author exists.**
   *"If a qualified author is available among the submission's authors but no such author is
   nominated, or if the nominated author fails to complete their assigned reviews, the submission
   may be desk rejected."* The nominee commits to reviewing **up to 6 papers**.
2. **Incomplete OpenReview author profiles.** Every author needs a profile *before* submitting, with
   full publication name, current position, institution-affiliated email, and DBLP URL.
   *"submissions with incomplete author profiles will be subject to desk rejection."*
   As of 2026-07-17 the form lists **only Remy Hosseinkhan Boucher**.
3. **Linking to an external code/data repository — forbidden.**
   *"linking to the paper sources/data in an external code/data repository is forbidden, including
   anonymized repositories like AnonymousGitHub and similar."*
   → `github.com/JoachimJobard/RL_and_signatures` **must not be cited**. It is public and carries an
   author's name, so a link would also break anonymity. Code goes as the **Code and Data Supplement**
   upload (≤50 MB).
4. **Dual submission.** Must affirm no substantially similar version is under review elsewhere, and
   that simultaneous submissions by any author are cited anonymously as "under review".

---

## 3. Required form fields

| Field | Status | Note |
|---|---|---|
| Title | required | |
| Authors | required | all need OpenReview profiles |
| **TL;DR** | optional | *"a short sentence describing your paper"* — **not yet drafted** |
| Abstract | required | candidates in `2026-07-17_aaai_abstract_candidates.md` |
| Primary Topic | required | **not yet chosen** |
| Secondary Topics | optional, ≤5 | **not yet chosen** |
| Country of Institutions | required | |
| PDF | by 29 Jul | |
| **Reproducibility Checklist** | required | answered standalone version from the Author Kit (`AuthorKit27/ReproducibilityChecklist.tex`, already in `paper_draft_aaai_template/`) |
| Technical Supplement | optional | proofs/derivations; main paper must stand alone |
| Media Supplement | optional | ≤50 MB |
| Code and Data Supplement | optional | ≤50 MB; **no external links** |
| Reciprocal Reviewer Nomination | see §2.1 | locks 21 Jul AoE |
| Submission Policies Acknowledgement | required | |
| License | CC BY 4.0 | |

**Author Kit:** `~/Downloads/AuthorKit27.zip` — verified byte-size identical to the copy already in
`latex_documents/.../paper_draft_aaai_template/` (`aaai2027.sty` 16915, `.bst` 30207,
`CameraReady2027.tex` 57156, `ReproducibilityChecklist.tex` 8729, all three PDFs). The local
`AnonymousSubmission2027.tex` (36299) and `aaai2027.bib` (52455) differ deliberately — boilerplate
stripped, bibliography added. **No template update is needed.**

---

## 4. Reciprocal reviewer eligibility — the author's record (measured)

Rule: not an SPC, AC or Organiser, **and** either **≥2 first-author** or **≥5 co-authored**
publications in peer-reviewed **archival** venues related to AAAI. **Workshop papers do not count.**

Source of truth: DBLP **PID 379/5221** (`https://dblp.org/pid/379/5221`), read 2026-07-17. Seven
entries:

| Entry | Type | Role | Counts? |
|---|---|---|---|
| *On Learning-Based Control of Dynamical Systems* (2025) | `phdthesis` | first | ✗ a thesis is not a publication |
| *Learning non-Markovian Dynamical Systems with Signature-based Encoders* (2025, **ML-DE**) | `inproceedings` | co-author | ✗ **workshop** |
| *Evidence on the Regularisation Properties…* (2025, CoRR) | `article` | first | ✗ arXiv, not peer-reviewed |
| *Increasing Information for MPC…* (2025, CoRR) | `article` | first | ✗ arXiv |
| *Learning non-Markovian…* (2025, CoRR) | `article` | co-author | ✗ arXiv |
| ***Increasing information for MPC with semi-Markov decision processes* (2024, L4DC)** | `inproceedings` | **first** | ✓ PMLR, peer-reviewed archival |
| ***Evidence on the Regularisation Properties of Maximum-Entropy RL* (2024, OLA)** | `inproceedings` | **first** | ✓ Springer proceedings — peer-reviewed, smaller venue |

The three CoRR entries are the arXiv twins of the L4DC and OLA papers, not additional works.

### RESOLVED 2026-07-17 by the author, against the Checker: **NO author qualifies.**

- **OLA is not a recognised venue** ("not known at all" to the Checker) → it does **not** count.
- **L4DC 2024 counts**, and is therefore the author's **only** qualifying publication → **1
  first-author against the required 2** → **not qualified**.
- The ≥5 co-authored route is not close (≈1–2 archival works in total).
- **Lionel Mathelin and Onofrio Semeraro also do not qualify** (checked by the author). Their records
  are outside the AAAI-related venue list, so the earlier recommendation to nominate one of them is
  **withdrawn**.

**Consequence — and it is benign.** With no qualifying author, the form's second option,
*"We declare that no author qualifies"*, is a **true** attestation and is the correct selection. The
desk-rejection clause fires only on a **false** declaration (a qualified author existing and not being
nominated), so this route carries no risk here. **No nomination is needed, and the 21 July lock
therefore ceases to be a live hazard** — only the declaration must be ticked.

An optional confirmation email to `workflowchairs@aaai.zendesk.com` is drafted but not required.

---

## 5. Open items

1. **Decide the reciprocal reviewer and enter it — before 21 July AoE.** Irreversible.
2. Run the qualification checker on the DBLP profile (and on the intended nominee's).
3. Add the remaining authors and confirm each has a complete OpenReview profile.
4. Choose Primary Topic (+ up to five secondary).
5. Draft the TL;DR.
6. Answer the Reproducibility Checklist.
7. Assemble the Code and Data Supplement as an **upload** — no repository link.
