# Third-party vendoring

## pykt-toolkit (git submodule)

Upstream: [pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit).

After cloning this repo, fetch the pinned revision:

```powershell
git submodule update --init --recursive
```

Optional dependency `[pykt]` in the root `pyproject.toml` installs `pykt-toolkit` from `third_party/pykt-toolkit` (same commit Git records for the submodule), so PyPI is not used for that package.
