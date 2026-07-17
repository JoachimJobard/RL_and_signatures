"""Warn-once reporting for numerical clamps, floors, truncations and guards.

A numerical device that can alter a value at runtime -- a divergence cut, a positivity
projection, a denominator epsilon, a gradient clip, a NaN replacement -- must announce itself
when it actually *binds*. The distinction this module exists to serve:

* A clamp is **non-binding** (inactive) on a step when the raw value already satisfies it, so the
  clamp is the identity map and provably cannot change any reported number. Such a clamp is
  mathematically inert and needs no report.
* A clamp is **binding** (active) when it replaces the raw value with a different one. It has then
  changed the answer, and a number computed downstream of it is not the number it appears to be.

The presence of a clamp in the source therefore proves nothing about a run; only its *binding*
does, and binding cannot be read off a configuration file. A clamp that binds without saying so
regularises the very regime it touches, and can mask the phenomenon under study -- a solver looks
stable, an optimiser looks convergent, precisely *because* the clamp is hiding the singular corner
where the interesting behaviour lives.

Reporting convention, applied uniformly by this module:

1. The **first** activation of each named clamp is reported at WARNING level, via ``warnings.warn``,
   so that it surfaces at default verbosity and in captured pytest output.
2. **Subsequent** activations are reported at DEBUG level, so that a hot loop stays quiet.
3. Every report carries **which** clamp bound, **how many** elements it affected, and the **most
   extreme raw (pre-clamp) value**, so the log shows how far into the truncated regime the inputs
   went rather than merely that a boundary was touched.

Only a clamp that is *provably* inert -- an autograd-safety epsilon whose magnitude is below any
value the computation can produce -- may be left unreported, and then only with a comment stating
why it is inert.
"""

from __future__ import annotations

import logging
import warnings

_LOGGER = logging.getLogger(__name__)

# Names of clamps whose first activation has already been reported at WARNING level.
_CLAMPS_ALREADY_WARNED: set[str] = set()

# Number of activations recorded per clamp name, for end-of-run summaries and for tests.
_CLAMP_ACTIVATION_COUNTS: dict[str, int] = {}


def report_clamp_activation(
    clamp_name: str,
    *,
    code_location: str,
    bound_description: str,
    most_extreme_raw_value: float,
    number_of_affected_elements: int = 1,
    additional_context: str = "",
    alters_the_value: bool = True,
) -> None:
    """Report that a numerical clamp has bound, i.e. has actually changed a value.

    Call this only when the clamp *binds*. Calling it on every evaluation, including the
    non-binding ones, would defeat the purpose: the signal being reported is precisely that the
    raw value left the admissible region.

    Parameters
    ----------
    clamp_name
        Stable identifier of the clamp, used to decide whether this is the first activation.
        Include the quantity and the agent/solver, e.g. ``"divergence_threshold/value_gradient"``.
    code_location
        ``file:line`` of the clamp itself, so a reader can go straight to it.
    bound_description
        The bound in the mathematics' own terms, e.g. ``"||x|| <= 100.0"``.
    most_extreme_raw_value
        The most extreme *pre-clamp* value seen in this activation. This is the number that says
        how far outside the bound the computation actually went.
    number_of_affected_elements
        How many elements the clamp altered in this activation.
    additional_context
        Free text pinning the activation to a point in the run, e.g. the episode and time.
    """
    count = _CLAMP_ACTIVATION_COUNTS.get(clamp_name, 0) + 1
    _CLAMP_ACTIVATION_COUNTS[clamp_name] = count

    context_suffix = f" {additional_context}" if additional_context else ""
    if alters_the_value:
        kind, consequence = "Clamp", (
            "A quantity computed downstream of a clamp that binds is not the quantity it appears "
            "to be: the clamp has replaced the value the mathematics prescribed."
        )
    else:
        # A FAILURE GUARD replaces nothing — it detects that the computation has already left the
        # domain where its result means anything (a non-finite state, say). Reporting it with the
        # clamp boilerplate would misdescribe it as an intervention, which is exactly the sort of
        # imprecision this module exists to prevent.
        kind, consequence = "Guard", (
            "This guard alters no value: it reports that the computation had already left the "
            "domain in which its result is meaningful."
        )
    message = (
        f"{kind} '{clamp_name}' FIRED at {code_location}: the condition is {bound_description}, "
        f"and the most extreme raw value was {most_extreme_raw_value!r}, affecting "
        f"{number_of_affected_elements} element(s).{context_suffix} {consequence} "
        f"Activation number {count} for this {kind.lower()}."
    )

    if clamp_name not in _CLAMPS_ALREADY_WARNED:
        _CLAMPS_ALREADY_WARNED.add(clamp_name)
        warnings.warn(message + " (First activation; further activations are logged at DEBUG.)",
                      RuntimeWarning, stacklevel=2)
    else:
        _LOGGER.debug(message)


def get_clamp_activation_counts() -> dict[str, int]:
    """Return a copy of the per-clamp activation counts recorded so far in this process.

    Intended for an end-of-run summary and for tests. A run whose report lists a non-zero count
    for a clamp must be read with that clamp in mind.
    """
    return dict(_CLAMP_ACTIVATION_COUNTS)


def reset_clamp_activation_state() -> None:
    """Forget every recorded activation and warn-once flag.

    Intended for tests, which must be able to assert on the first activation independently of the
    order in which they run.
    """
    _CLAMPS_ALREADY_WARNED.clear()
    _CLAMP_ACTIVATION_COUNTS.clear()
