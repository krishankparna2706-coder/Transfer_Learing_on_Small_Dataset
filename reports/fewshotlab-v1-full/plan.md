# Study plan: fewshotlab-v1-full

160 logical slots, 156 unique runs (4 deterministic duplicates counted once). Category `core`. Generated 2026-11-02T06:24:05Z.

## Pre-flight checks

- PASS unique_run_ids: 160 unique of 160
- PASS shared_support_ids: every method at a dataset/K/seed reads the same support manifest
- PASS initialization_policy: scratch starts from random init; frozen/partial/prototypes from IMAGENET1K_V1
- PASS paired_init_seed: init seed depends only on train_seed (= support_seed)
- PASS identical_training_transform_per_dataset: {'neu_cls': ['9d3ac91523ecdbe9'], 'flowers102': ['b5bbd7ad654b4be4']}
- PASS single_recipe: 1 distinct recipe(s) across learned slots

## Slots

| dataset | K | method | seed | status | run ID |
|---|---:|---|---:|---|---|
| flowers102 | 1 | scratch | 11 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k1__s11__cfgcbfc24cb` |
| flowers102 | 1 | scratch | 22 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k1__s22__cfg009060ef` |
| flowers102 | 1 | scratch | 33 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k1__s33__cfg6898b5f8` |
| flowers102 | 1 | scratch | 44 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k1__s44__cfg1e3865e0` |
| flowers102 | 1 | scratch | 55 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k1__s55__cfg46d17632` |
| flowers102 | 1 | frozen | 11 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k1__s11__cfg184249bf` |
| flowers102 | 1 | frozen | 22 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k1__s22__cfg629e861b` |
| flowers102 | 1 | frozen | 33 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k1__s33__cfga662c00e` |
| flowers102 | 1 | frozen | 44 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k1__s44__cfg873d917a` |
| flowers102 | 1 | frozen | 55 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k1__s55__cfga6adc8cf` |
| flowers102 | 1 | partial | 11 | COMPLETED | `fewshotlab-v1__flowers102__partial__k1__s11__cfg13f9d385` |
| flowers102 | 1 | partial | 22 | COMPLETED | `fewshotlab-v1__flowers102__partial__k1__s22__cfg73f174ec` |
| flowers102 | 1 | partial | 33 | COMPLETED | `fewshotlab-v1__flowers102__partial__k1__s33__cfgc38d168f` |
| flowers102 | 1 | partial | 44 | COMPLETED | `fewshotlab-v1__flowers102__partial__k1__s44__cfg0c4d4558` |
| flowers102 | 1 | partial | 55 | COMPLETED | `fewshotlab-v1__flowers102__partial__k1__s55__cfg6fcb5d78` |
| flowers102 | 1 | prototypes | 11 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k1__s11__cfg252cb28c` |
| flowers102 | 1 | prototypes | 22 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k1__s22__cfga8ff7d81` |
| flowers102 | 1 | prototypes | 33 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k1__s33__cfgbab095d6` |
| flowers102 | 1 | prototypes | 44 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k1__s44__cfg712802ab` |
| flowers102 | 1 | prototypes | 55 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k1__s55__cfgbc76f4ce` |
| flowers102 | 3 | scratch | 11 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k3__s11__cfgae535514` |
| flowers102 | 3 | scratch | 22 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k3__s22__cfg4a7af7e0` |
| flowers102 | 3 | scratch | 33 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k3__s33__cfg78c018ad` |
| flowers102 | 3 | scratch | 44 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k3__s44__cfgf3400cb3` |
| flowers102 | 3 | scratch | 55 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k3__s55__cfgbd400fc6` |
| flowers102 | 3 | frozen | 11 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k3__s11__cfg71c509e0` |
| flowers102 | 3 | frozen | 22 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k3__s22__cfg47d62064` |
| flowers102 | 3 | frozen | 33 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k3__s33__cfgcb851adf` |
| flowers102 | 3 | frozen | 44 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k3__s44__cfgde1baedb` |
| flowers102 | 3 | frozen | 55 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k3__s55__cfga6253b1d` |
| flowers102 | 3 | partial | 11 | COMPLETED | `fewshotlab-v1__flowers102__partial__k3__s11__cfg44ba7894` |
| flowers102 | 3 | partial | 22 | COMPLETED | `fewshotlab-v1__flowers102__partial__k3__s22__cfg542c51e9` |
| flowers102 | 3 | partial | 33 | COMPLETED | `fewshotlab-v1__flowers102__partial__k3__s33__cfg55d590bc` |
| flowers102 | 3 | partial | 44 | COMPLETED | `fewshotlab-v1__flowers102__partial__k3__s44__cfg2a5d7535` |
| flowers102 | 3 | partial | 55 | COMPLETED | `fewshotlab-v1__flowers102__partial__k3__s55__cfg2349ffb4` |
| flowers102 | 3 | prototypes | 11 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k3__s11__cfga865ee8f` |
| flowers102 | 3 | prototypes | 22 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k3__s22__cfg2265c993` |
| flowers102 | 3 | prototypes | 33 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k3__s33__cfgd1a60d2d` |
| flowers102 | 3 | prototypes | 44 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k3__s44__cfg238362f7` |
| flowers102 | 3 | prototypes | 55 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k3__s55__cfga04f1c2c` |
| flowers102 | 5 | scratch | 11 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k5__s11__cfg6081216d` |
| flowers102 | 5 | scratch | 22 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k5__s22__cfg72bca28a` |
| flowers102 | 5 | scratch | 33 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k5__s33__cfge79cad4a` |
| flowers102 | 5 | scratch | 44 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k5__s44__cfg308bed7a` |
| flowers102 | 5 | scratch | 55 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k5__s55__cfgd4e2bc05` |
| flowers102 | 5 | frozen | 11 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k5__s11__cfg1c8b0d96` |
| flowers102 | 5 | frozen | 22 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k5__s22__cfg2651d8eb` |
| flowers102 | 5 | frozen | 33 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k5__s33__cfg8b8e7c1c` |
| flowers102 | 5 | frozen | 44 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k5__s44__cfg09e0e800` |
| flowers102 | 5 | frozen | 55 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k5__s55__cfg11980ba7` |
| flowers102 | 5 | partial | 11 | COMPLETED | `fewshotlab-v1__flowers102__partial__k5__s11__cfg3ee1aca7` |
| flowers102 | 5 | partial | 22 | COMPLETED | `fewshotlab-v1__flowers102__partial__k5__s22__cfgdcad806d` |
| flowers102 | 5 | partial | 33 | COMPLETED | `fewshotlab-v1__flowers102__partial__k5__s33__cfg2145f765` |
| flowers102 | 5 | partial | 44 | COMPLETED | `fewshotlab-v1__flowers102__partial__k5__s44__cfg50f029c9` |
| flowers102 | 5 | partial | 55 | COMPLETED | `fewshotlab-v1__flowers102__partial__k5__s55__cfg46d2aac3` |
| flowers102 | 5 | prototypes | 11 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k5__s11__cfga0963029` |
| flowers102 | 5 | prototypes | 22 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k5__s22__cfgfe0ab699` |
| flowers102 | 5 | prototypes | 33 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k5__s33__cfg31920d10` |
| flowers102 | 5 | prototypes | 44 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k5__s44__cfgc3be5ca5` |
| flowers102 | 5 | prototypes | 55 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k5__s55__cfg218b5890` |
| flowers102 | 10 | scratch | 11 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k10__s11__cfg58a96d8c` |
| flowers102 | 10 | scratch | 22 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k10__s22__cfga9cf001f` |
| flowers102 | 10 | scratch | 33 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k10__s33__cfg1b023fa2` |
| flowers102 | 10 | scratch | 44 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k10__s44__cfg5a0f6821` |
| flowers102 | 10 | scratch | 55 | COMPLETED | `fewshotlab-v1__flowers102__scratch__k10__s55__cfg8adceb0c` |
| flowers102 | 10 | frozen | 11 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k10__s11__cfg74c47f5d` |
| flowers102 | 10 | frozen | 22 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k10__s22__cfgdf7400cf` |
| flowers102 | 10 | frozen | 33 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k10__s33__cfgefd43ac6` |
| flowers102 | 10 | frozen | 44 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k10__s44__cfg1a2fb687` |
| flowers102 | 10 | frozen | 55 | COMPLETED | `fewshotlab-v1__flowers102__frozen__k10__s55__cfgf1f5a13f` |
| flowers102 | 10 | partial | 11 | COMPLETED | `fewshotlab-v1__flowers102__partial__k10__s11__cfg6a40d8f4` |
| flowers102 | 10 | partial | 22 | COMPLETED | `fewshotlab-v1__flowers102__partial__k10__s22__cfg29298795` |
| flowers102 | 10 | partial | 33 | COMPLETED | `fewshotlab-v1__flowers102__partial__k10__s33__cfg40565516` |
| flowers102 | 10 | partial | 44 | COMPLETED | `fewshotlab-v1__flowers102__partial__k10__s44__cfg75dec688` |
| flowers102 | 10 | partial | 55 | COMPLETED | `fewshotlab-v1__flowers102__partial__k10__s55__cfge4f3af2e` |
| flowers102 | 10 | prototypes | 11 | COMPLETED | `fewshotlab-v1__flowers102__prototypes__k10__s11__cfg086a51d2` |
| flowers102 | 10 | prototypes | 22 | DEDUPLICATED (= fewshotlab-v1__flowers102__prototypes__k10__s11__cfg086a51d2) | `fewshotlab-v1__flowers102__prototypes__k10__s22__cfgf42f3c75` |
| flowers102 | 10 | prototypes | 33 | DEDUPLICATED (= fewshotlab-v1__flowers102__prototypes__k10__s11__cfg086a51d2) | `fewshotlab-v1__flowers102__prototypes__k10__s33__cfg868cbce4` |
| flowers102 | 10 | prototypes | 44 | DEDUPLICATED (= fewshotlab-v1__flowers102__prototypes__k10__s11__cfg086a51d2) | `fewshotlab-v1__flowers102__prototypes__k10__s44__cfgcbb2d506` |
| flowers102 | 10 | prototypes | 55 | DEDUPLICATED (= fewshotlab-v1__flowers102__prototypes__k10__s11__cfg086a51d2) | `fewshotlab-v1__flowers102__prototypes__k10__s55__cfgc214ca50` |
| neu_cls | 1 | scratch | 11 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k1__s11__cfg00f23917` |
| neu_cls | 1 | scratch | 22 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k1__s22__cfg3707aefc` |
| neu_cls | 1 | scratch | 33 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k1__s33__cfgdcf0a720` |
| neu_cls | 1 | scratch | 44 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k1__s44__cfgc2bb3e6f` |
| neu_cls | 1 | scratch | 55 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k1__s55__cfg3e5fc8a8` |
| neu_cls | 1 | frozen | 11 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k1__s11__cfgae8bd4ad` |
| neu_cls | 1 | frozen | 22 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k1__s22__cfg49f99899` |
| neu_cls | 1 | frozen | 33 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k1__s33__cfg106820fd` |
| neu_cls | 1 | frozen | 44 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k1__s44__cfg5658e323` |
| neu_cls | 1 | frozen | 55 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k1__s55__cfgbd0058f5` |
| neu_cls | 1 | partial | 11 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k1__s11__cfg71c55691` |
| neu_cls | 1 | partial | 22 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k1__s22__cfg903fab37` |
| neu_cls | 1 | partial | 33 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k1__s33__cfg76594641` |
| neu_cls | 1 | partial | 44 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k1__s44__cfgea07d7c5` |
| neu_cls | 1 | partial | 55 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k1__s55__cfg9ff11d10` |
| neu_cls | 1 | prototypes | 11 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k1__s11__cfg08cf953c` |
| neu_cls | 1 | prototypes | 22 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k1__s22__cfgd2451cf3` |
| neu_cls | 1 | prototypes | 33 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k1__s33__cfg7d12aab4` |
| neu_cls | 1 | prototypes | 44 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k1__s44__cfg8aa75f9c` |
| neu_cls | 1 | prototypes | 55 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k1__s55__cfgab0be004` |
| neu_cls | 3 | scratch | 11 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k3__s11__cfga219b3c7` |
| neu_cls | 3 | scratch | 22 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k3__s22__cfg79678373` |
| neu_cls | 3 | scratch | 33 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k3__s33__cfge40d7524` |
| neu_cls | 3 | scratch | 44 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k3__s44__cfgb6899e7c` |
| neu_cls | 3 | scratch | 55 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k3__s55__cfg38077ef0` |
| neu_cls | 3 | frozen | 11 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k3__s11__cfg73b450cb` |
| neu_cls | 3 | frozen | 22 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k3__s22__cfg1ac02882` |
| neu_cls | 3 | frozen | 33 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k3__s33__cfg261859f1` |
| neu_cls | 3 | frozen | 44 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k3__s44__cfg8651d763` |
| neu_cls | 3 | frozen | 55 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k3__s55__cfg8a42cd92` |
| neu_cls | 3 | partial | 11 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k3__s11__cfgafc3b95d` |
| neu_cls | 3 | partial | 22 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k3__s22__cfg44397afe` |
| neu_cls | 3 | partial | 33 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k3__s33__cfge5ff02ad` |
| neu_cls | 3 | partial | 44 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k3__s44__cfg051dc6e0` |
| neu_cls | 3 | partial | 55 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k3__s55__cfgd82f6a25` |
| neu_cls | 3 | prototypes | 11 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k3__s11__cfg12da9145` |
| neu_cls | 3 | prototypes | 22 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k3__s22__cfg5bc72632` |
| neu_cls | 3 | prototypes | 33 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k3__s33__cfg1637213f` |
| neu_cls | 3 | prototypes | 44 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k3__s44__cfga7ce7f8a` |
| neu_cls | 3 | prototypes | 55 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k3__s55__cfg8c8be8dd` |
| neu_cls | 5 | scratch | 11 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k5__s11__cfg13338879` |
| neu_cls | 5 | scratch | 22 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k5__s22__cfgd30a00bc` |
| neu_cls | 5 | scratch | 33 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k5__s33__cfg11f0c196` |
| neu_cls | 5 | scratch | 44 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k5__s44__cfg5f5c2da7` |
| neu_cls | 5 | scratch | 55 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k5__s55__cfg7174afb7` |
| neu_cls | 5 | frozen | 11 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k5__s11__cfg85db7fc7` |
| neu_cls | 5 | frozen | 22 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k5__s22__cfg3e77c348` |
| neu_cls | 5 | frozen | 33 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k5__s33__cfg029ffbc4` |
| neu_cls | 5 | frozen | 44 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k5__s44__cfge829fe1c` |
| neu_cls | 5 | frozen | 55 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k5__s55__cfg8b5ac2db` |
| neu_cls | 5 | partial | 11 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k5__s11__cfgc6a81028` |
| neu_cls | 5 | partial | 22 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k5__s22__cfg5cff60bd` |
| neu_cls | 5 | partial | 33 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k5__s33__cfga703faff` |
| neu_cls | 5 | partial | 44 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k5__s44__cfgb6c89b8e` |
| neu_cls | 5 | partial | 55 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k5__s55__cfg96be9f01` |
| neu_cls | 5 | prototypes | 11 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k5__s11__cfge209274b` |
| neu_cls | 5 | prototypes | 22 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k5__s22__cfg0a3c8de4` |
| neu_cls | 5 | prototypes | 33 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k5__s33__cfg44001a92` |
| neu_cls | 5 | prototypes | 44 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k5__s44__cfg5efb0626` |
| neu_cls | 5 | prototypes | 55 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k5__s55__cfgef01c21d` |
| neu_cls | 10 | scratch | 11 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k10__s11__cfg24bd0ca4` |
| neu_cls | 10 | scratch | 22 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k10__s22__cfgf118e71c` |
| neu_cls | 10 | scratch | 33 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k10__s33__cfgec69b167` |
| neu_cls | 10 | scratch | 44 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k10__s44__cfga18ab348` |
| neu_cls | 10 | scratch | 55 | COMPLETED | `fewshotlab-v1__neu_cls__scratch__k10__s55__cfg0c0aa8a2` |
| neu_cls | 10 | frozen | 11 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k10__s11__cfg2d81b62f` |
| neu_cls | 10 | frozen | 22 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k10__s22__cfgffc4a0ba` |
| neu_cls | 10 | frozen | 33 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k10__s33__cfgc4b4ee48` |
| neu_cls | 10 | frozen | 44 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k10__s44__cfga5f386fb` |
| neu_cls | 10 | frozen | 55 | COMPLETED | `fewshotlab-v1__neu_cls__frozen__k10__s55__cfg88f46d31` |
| neu_cls | 10 | partial | 11 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k10__s11__cfg085b60aa` |
| neu_cls | 10 | partial | 22 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k10__s22__cfg885fb69f` |
| neu_cls | 10 | partial | 33 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k10__s33__cfg3d073c2d` |
| neu_cls | 10 | partial | 44 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k10__s44__cfgfddac134` |
| neu_cls | 10 | partial | 55 | COMPLETED | `fewshotlab-v1__neu_cls__partial__k10__s55__cfg8a5f793f` |
| neu_cls | 10 | prototypes | 11 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k10__s11__cfg90c8a345` |
| neu_cls | 10 | prototypes | 22 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k10__s22__cfg752034ee` |
| neu_cls | 10 | prototypes | 33 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k10__s33__cfg2fc51792` |
| neu_cls | 10 | prototypes | 44 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k10__s44__cfg4deb0563` |
| neu_cls | 10 | prototypes | 55 | COMPLETED | `fewshotlab-v1__neu_cls__prototypes__k10__s55__cfgee1a1d8c` |
