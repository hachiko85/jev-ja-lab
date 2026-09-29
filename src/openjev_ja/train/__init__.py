"""Training: the YAML-configured, architecture-per-library counterpart to `openjev_ja.eval`.

    corpus.py         raw training corpora -> BenchmarkItem (parallel to eval.local_data)
    cv.py             k-fold split, data-agnostic
    architectures.py  registry of trainable architectures (parallel to eval.orchestrate's
                      scorer dispatch) -- add a new library's architecture here
    runner.py         one architecture's k-fold cross-validation training loop
    config.py         YAML workflow: one architecture + corpus per model entry
    cli.py            `jev-ja-lab-train-workflow --config <path>`
"""
