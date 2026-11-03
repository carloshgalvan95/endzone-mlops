# ADR-005: Acceptance rule for a Phase 2 feature set

**Date**: 2026-11-03
**Status**: Accepted

## Context

Phase 2 compares a candidate to a reference on rolling-origin CV, seasons 2016-2023. A small log-loss change can look real once the interval is known. The bar has to be fixed before the run.

## Decision

Accept a feature set only when both are true on the pooled out-of-fold games:

1. Pooled log loss improves on the reference by at least 0.003. That is `log_loss_diff` <= -0.003.
2. The 95% paired bootstrap interval lies entirely below 0. The bootstrap draws the same game indexes for both models, 1000 resamples, seed 42.

If either condition fails, the set is rejected. A missing interval is inconclusive. The threshold is not moved after a result.

The ridge setup is a separate rule: switch the reference to `logistic_l2` on `h2_full` only when its `log_loss_diff` is <= 0. That run did not. The reference stays logistic `h2_full`.

## Consequences

H5 (`h5_qb` 0.6355, difference -0.0022, interval -0.0042 to -0.0003) and H5b (`h5_qb` 0.6356, difference -0.0021, interval -0.0039 to -0.0002) are rejected. The intervals exclude zero, which is a positive signal, and both are short of -0.003. The market can be labeled accepted by the same numeric rule. That label is a baseline comparison, not a promotion.
