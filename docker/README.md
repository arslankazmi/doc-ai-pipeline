# Docker Deployment

## Build and run
```bash
docker compose up --build
```

## Services
- Inference API: http://localhost:8000 (health: `/health`, demo: `/demo`)
- HITL Review: http://localhost:8001 (health: `/health`, queue: `/queue`)

## Data persistence
Both services share a named Docker volume `pipeline_data` for `cases.db` and uploads.

## Environment variables
| Variable | Default | Description |
|---|---|---|
| CASES_DB_PATH | /app/data/cases.db | SQLite database path |
| CONFIDENCE_THRESHOLD | 0.85 | Routing threshold |

## Rebuilding after code changes
```bash
docker compose up --build --force-recreate
```
