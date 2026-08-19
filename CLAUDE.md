# Code standards

- SOLID. Single responsibility per module/class; depend on abstractions (protocols), not implementations.
- DRY. One function per behavior, reused everywhere. No copy-pasted logic.
- Small, readable, typed functions. Type hints on every signature.
- Separation of concerns: `evidence/`, `agents/`, `debate/`, `trace/`, `api` communicate ONLY through `models/` and injected dependencies. No cross-package internal imports.
- No magic numbers or strings in code. Secrets in `.env`; every constant/threshold/model name in `committee/config.py` (`from committee.config import settings`).
- Minimal comments: only non-obvious whys and gotchas.
- Commits: one commit per concern (tools together, pub/sub separate, etc.). Never add Claude co-author lines.
- No CI. Tests live in `tests/`, run with `poetry run pytest`.
