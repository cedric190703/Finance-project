# ADR 001: DuckDB with bitemporal observations

## Status

Accepted.

## Context

The project needs a local analytical store that can be rebuilt from archived
provider responses and can distinguish the date a market value describes from
the date the system learned it.

## Decision

Use DuckDB with append-only fact tables. All reads go through an `as_of` query
that selects the latest revision whose `knowledge_date` is not after the
requested point in time.

## Consequences

The project has no database service dependency and its EOD ledger can live with
the market data. It is intentionally not a multi-writer production database;
an operational deployment that needs concurrent writers should replace this
adapter with a transactional warehouse while retaining the same bitemporal
contract.
