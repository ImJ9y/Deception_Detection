# Public Release Checklist

- [ ] Remove any secrets or credentials from code/configs.
- [ ] Remove private data files and checkpoints.
- [ ] Confirm all paths in README are generic (no private machine paths).
- [ ] Verify every script in this sample compiles:
  - `python3 -m py_compile $(find . -name '*.py')`
- [ ] Confirm authorship/attribution sections are accurate.
- [ ] Add LICENSE (for example MIT) before publishing.
- [ ] Push this folder to a **public** GitHub repository.
- [ ] Verify GitHub link and README render correctly.
