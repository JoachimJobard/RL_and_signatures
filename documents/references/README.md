# Reference documents

Primary references for the internship of Joachim Jobard on signature-based
reinforcement learning for non-Markovian (delayed) dynamics.

- **`K. Doya - Reinforcement Learning In Continuous Time and Space (2000).pdf`**
  Foundational paper on continuous-time, continuous-space reinforcement learning
  (continuous-time TD error, advantage updating, value-gradient based policies).
  The internship extends this framework to the signature-based, non-Markovian
  setting.

- **`J. Jobard - Continuous Reinforcement Learning for Delayed Dynamical Systems:
  A Path Signature-Based Representation Approach (2026).pdf`**
  Master thesis describing the work carried out during the internship: the
  signature-based representation of delayed (non-Markovian) state histories, the
  continuous-time actor-critic algorithms implemented in this repository, the
  experimental methodology, and the results obtained so far (some experiments
  remain to be completed).

- **`I. Perez Arribas - Derivatives pricing using signature payoffs (2018).pdf`**
  Source of the path-signature representation used in this codebase. Its
  Definition (p.5) gives the augmented path $\widehat{X}_t = (t,\,X_t,\,(X_0/T)\,t)$
  — a monotone time channel, the state path, and a linear basepoint ramp — and its
  Theorem 4.2 (p.6) is the universal-approximation result (linear functionals of
  the signature of the augmented path are dense in continuous payoffs), with the
  monotone time channel making the signature injective. The signature feature map
  in `src/utils/dynamic_signature.py` implements this augmentation.
