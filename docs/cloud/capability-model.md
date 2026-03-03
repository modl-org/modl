# Cloud Capability Model

> **STATUS: NOT IMPLEMENTED** — Future spec for `modl --cloud` gating. See [plan.md](plan.md) for cloud roadmap.

## Purpose

Define a provider-neutral cloud gating model for `modl --cloud` execution.

Launch goals:
- Fail fast on missing auth
- Fail fast on quota/concurrency limits
- Keep CLI independent of provider-specific APIs

---

## Launch Scope (MVP)

For launch, cloud gating includes only:
1. Auth validation (token/session + account active)
2. Quota validation (remaining credits/minutes + concurrency)

All other capability/policy layers are future work.

---

## Canonical Launch Gate Schema

```json
{
  "schema_version": "v1",
  "provider": "modal",
  "auth": {
    "authenticated": true,
    "account_active": true
  },
  "quota": {
    "remaining_gpu_minutes": 1200,
    "max_concurrent_jobs": 4,
    "current_running_jobs": 1
  },
  "limits": {
    "max_concurrent_jobs": 4,
    "max_job_duration_seconds": 21600
  }
}
```

---

## Validation Pipeline

Before cloud submission:

1. Resolve target/provider from precedence rules
2. Load cached auth/quota snapshot
3. Refresh from provider if stale or forced
4. Evaluate requested `JobSpec` against auth + quota rules
5. Return one of:
   - `Allowed`
  - `Denied(code, message, remediation)`

---

## Denial Codes

- `CLOUD_AUTH_MISSING`
- `SUBSCRIPTION_INACTIVE`
- `MAX_CONCURRENCY_EXCEEDED`
- `QUOTA_EXCEEDED`

Launch keeps denial taxonomy intentionally small.

Each denial must include actionable remediation text.

---

## Launch Behavior

No automatic cloud-side downgrades in MVP.

If auth/quota checks fail, return `Denied` with actionable remediation.

---

## Quota Accounting

Track canonical metrics in local DB:
- submitted jobs
- succeeded/failed jobs
- GPU seconds consumed
- estimated vs actual cost

Provider-specific billing fields may be stored in metadata JSON but must not replace canonical fields.

---

## Refresh Policy

- Auth/quota cache TTL default: 5 minutes
- Forced refresh on:
  - explicit user command
  - auth token change
  - denial due to quota uncertainty

Commands:

```bash
modl cloud gate status
modl cloud gate refresh
modl cloud subscription status
```

---

## Future Scope (Post-Launch)

Deferred items:
- feature capability matrix (GPU types, regions, model restrictions)
- policy engine (org/project rules)
- automatic downgrade policy
- multi-provider failover and cost routing

---

## Provider Adapter Requirements

Each provider must implement:
- `auth_status()` -> canonical auth status mapping
- `quota_status()` -> canonical quota mapping
- `usage()` -> normalized usage metrics

Provider-specific details remain inside adapter modules.

---

## Phase Scope

Phase 3 (Modal MVP):
- support one provider
- support auth checks
- support quota and concurrency gating

Out of scope:
- multi-provider failover
- full capability matrix and policy engine
