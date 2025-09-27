# Wakeifier
Want to write like Joyce's Finnegans Wake? 

wakeify/
  data/
    raw/
      wake_source.txt
      seed_prompts.txt
    pairs/
      train.jsonl
      dev.jsonl
      test.jsonl
    style_clf/
      wakeish.txt
      not_wakeish.txt
  scripts/
    bootstrap_pairs.py
    train_lora.py
    eval_semantics.py
    train_style_clf.py
    infer_pipeline.py
    post_edit.py
  configs/
    lora.yaml
    gen.yaml
  models/
    base/                # symlink or HF cache
    lora_wakeify/        # output dir for adapters
    style_clf/           # tiny classifier weights
    embedder/            # sentence-transformers cache
  notebooks/
    00_data_exploration.ipynb
    01_error_analysis.ipynb
  server/
    app.py               # FastAPI for a local demo
  README.md
  requirements.txt
