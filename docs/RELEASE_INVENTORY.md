# Release inventory

Generated from the files on disk and checked against the sealed plan (`seal-c6f58862d63b586a0df8dbcf140db0fd`) and the rehearsal ledger event.

- **In git:** tracked files ship with the repository.
- **Local:** checkpoints and weights are approved artifacts kept outside git. Their hashes are recorded here and in the sealed plan.
- **Images:** no dataset image is released.

| Artifact | Path | SHA-256 | Check | In git |
|---|---|---|---|---|
| flowers102 split manifest | `data/manifests/flowers102-v1.csv` | `886cf59e92473be726096c16975e82804df0ebe18045274da891ad3747651283` | matches the sealed plan | yes |
| flowers102 class map | `data/manifests/flowers102.class_map.json` | `a6022208db65646efb7a335d9e40e54bc3cf906028124aaa39160e378d24580f` | canonical hash matches the sealed plan | yes |
| neu_cls split manifest | `data/manifests/neu_cls-v1.csv` | `2f4740b0b0167ed3ef95421b53da671724755ed918ed09f53912c51b4aa477ca` | matches the sealed plan | yes |
| neu_cls class map | `data/manifests/neu_cls.class_map.json` | `bf75c27cf955ae0202bed8db366d7dcbb7d1984da040bb80c5444422a89c2f60` | canonical hash matches the sealed plan | yes |
| support manifests (40) | `data/manifests/supports/*.json` | `ea7509ca4921a4170a9b3983377c3e053ce4a94b0b4d97f5be67038622722036` | 40 of 40 content hashes match the sealed plan | yes |
| uv.lock | `uv.lock` | `114a0511f2600595b185fd1c5b60f7f626d11dfdc48d66d40f69ae4e8d3f9bd5` | matches the sealed plan | yes |
| requirements.lock | `requirements.lock` | `3b3d00a64dcb29672d5acd23d628437b24bcb701b5e82ba92cf8607f13b9d499` | matches the sealed plan | yes |
| sealed plan | `reports/fewshotlab-v1-full/sealed_plan.json` | `f813009ed6940984d96830317ac775002ce134da370d9a8d5f365abd37e22222` | recorded here | yes |
| evaluation ledger | `reports/fewshotlab-v1-full/evaluation_ledger.jsonl` | `7412242554516b280cb08894db3e82cd1ef9b7a0bd86d7d08b21d46cf36ce5a9` | recorded here | yes |
| replay manifest | `reports/fewshotlab-v1-full/replay_manifest.csv` | `44cdb21bbbf3f49cf9b9901cae8233a494b7234e12a115e6bdf4bdc58190e546` | matches the sealed plan | yes |
| report aggregate_val.md | `reports/fewshotlab-v1-full/aggregate_val.md` | `66b4af001e0a115b66e6d0549fe1af44ac153e8feb567a02aaec2a355c286cf0` | recorded here | yes |
| report aggregate_val.json | `reports/fewshotlab-v1-full/aggregate_val.json` | `fc8a635fa40f99a4a2b0bc72a3639c5f6725e74f091e7350d9c43abc07de2fc5` | recorded here | yes |
| report test/aggregate_test.md | `reports/fewshotlab-v1-full/test/aggregate_test.md` | `7a7a22bc5ad0149863ca01dd153be1529971ca185f9005120adb66732be4c79c` | recorded here | yes |
| report test/aggregate_test.json | `reports/fewshotlab-v1-full/test/aggregate_test.json` | `ae9ff2c21f42ee1ca6a8cdd61250c8ef3f3c12217abb2e21ddea54a253ab2608` | recorded here | yes |
| report test/robustness.md | `reports/fewshotlab-v1-full/test/robustness.md` | `ae32c2b0340db10722c6d2f6385010820717f6b99ae57905f886cb4456b8124e` | recorded here | yes |
| report test/robustness.json | `reports/fewshotlab-v1-full/test/robustness.json` | `4b7530a68329112cbb879e2904994154b746e3b5e31c20bb7d5c24b1062ffa8d` | recorded here | yes |
| rehearsal rehearsal.json | `demo_assets/fewshotlab-v1-full/rehearsal.json` | `2bfa256ec7e609e1bddac66d6b815d347af7ef2616c493f0c8c5e1885e93f82b` | matches the ledger's replay_rehearsal event | yes |
| rehearsal rehearsal.md | `demo_assets/fewshotlab-v1-full/rehearsal.md` | `0cd8b9e3c9972c807d4e6a397d35a6e5fa56ee0f3b130b43a4abb25ad02e9d97` | matches the ledger's replay_rehearsal event | yes |
| demo checkpoint (NEU partial K=10 seed 11) | `runs/artifacts/fewshotlab-v1__neu_cls__partial__k10__s11__cfg085b60aa/a001/best.ckpt` | `830dbab5018f9bb1dd7d943cae64fcfee0eae080439289c1e64edee8a1bb2cdd` | matches the sealed plan | no (local) |
| all sealed run checkpoints | `runs/artifacts/<run>/<attempt>/` | `per run in sealed_plan.json` | 156 of 156 match the sealed plan | no (local) |
| pretrained weights IMAGENET1K_V1 | `.cache/torch/checkpoints/resnet18-f37072fd.pth` | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` | matches the sealed plan | no (local) |
