# benchmarks_archive

Archive zone for v1's 30+ `ausnut_benchmark_*` folders.

Once v2 reaches Phase 4 (evaluator rewrite), zip up everything from
`machine_learning/dataset_process/ausnut_benchmark_*` into a single archive
here and delete the originals. The history is preserved without polluting the
active `dataset_process/` directory.

Suggested command (PowerShell from the repo root):

```powershell
$dest = "machine_learning_v2/benchmarks_archive/2026-03_to_2026-05.zip"
Compress-Archive -Path machine_learning/dataset_process/ausnut_benchmark_* -DestinationPath $dest
```

Then in v1: `Remove-Item machine_learning/dataset_process/ausnut_benchmark_* -Recurse`

Do this only after v2 evaluator runs are stable so we can cite the historical
context if a regression ever needs to be investigated.
