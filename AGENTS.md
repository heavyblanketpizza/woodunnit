# Project instructions

- Use [README.md](README.md) as the sole project guide; keep documentation proportional to the codebase.
- Scope: local ingestion, curation, grouped splitting, reviewed export, and deterministic preprocessing. Keep `fungi = 0`, `oomycetes = 1`, with publisher-supported genus/species labels beneath them; do not add diagnoses, treatments, broad categories, or uploads.
- Preserve originals, publisher labels, credits, and permission evidence. Imported labels remain unreviewed; access alone is not usage approval. Do not invent metadata, reviews, or results.
- Group all known relationships before filtering, including excluded bridging records. Keep related images together; freeze splits before fitting or scoring and reserve test data for final evaluation.
- Use portable configuration and identical downstream preprocessing. Keep data, private metadata, local paths, generated artifacts, and weights outside Git and packages.
- Use synthetic fixtures and meaningful integrity/grouping/preprocessing checks. Run checks appropriate to the change; documentation-only changes need link and consistency checks.
- Keep the Git/package allowlists current. Preserve MIT licensing for original code and documentation and separate attribution for upstream sources.
