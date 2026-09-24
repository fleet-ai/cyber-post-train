# Qwen3.8 teacher3k 262K SFT history

Observed on 2026-09-24. This is a sanitized, read-only reconstruction of the
`chris-q38-t3k262` v1-v14 lineage. It uses retained RayJob objects, Jobs API
identities, SFS JSON receipts, and Git history. No private trainer logs,
prompts, traces, answers, flags, tensor values, or credentials were read.

## Result

- v1-v11 did not leave SFS proof of a completed optimizer update.
- v12 completed one optimizer update and wrote a native checkpoint. That is a
  successful **mechanics canary**, not an accepted scientific checkpoint: it
  has no seal, export, independent GPU reload, or `ACCEPTED.json`.
- v13 reached progress step 12. Step 10 is the only durable checkpoint. It was
  sealed, exported to BF16, reopened on CPU, and exactly restored by v14 on all
  64 ranks. It was not independently reloaded on GPU or accepted for serving
  and capability evaluation.
- v14 restored the exact step-10 optimizer, scheduler, and sampler state, then
  OOMed while attempting logical step 11. It wrote no new checkpoint.
- No v1-v14 artifact is an accepted scientific result. The lineage proves
  useful long-context training, save, export, and resume mechanics only.

All 14 RayJobs are terminal. On 2026-09-24, no matching RayCluster or Pod
existed. Retained Kueue Workloads all had `Finished=True`; this lineage held no
GPU allocation.

## Immutable source identities

| Item | Exact identity |
|---|---|
| Model | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Model weights | `sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352` |
| Image | `661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4` |
| Split manifest | `sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c` |
| Capacity corpus | `/mnt/sfs/jobs/chris-q38-study-corpora-v1/teacher3k-262k-v1/capacity-v5/train.parquet`, `sha256:2359c54e5c5cd756761a0f6e8c250ec8b32b87c8f84f0888252ddac932cfc5ec` |
| Capacity corpus manifest | `sha256:c5c7d82127d524593ed7e1cf5375fb3457e36c0d92376531808b6eeaacda10f5` |
| Capacity corpus shape | 112 rows, 32 sessions, 9,374 assistant responses, 3,022,959 supervised tokens |
| Full corpus | `/mnt/sfs/jobs/chris-q38-study-corpora-v1/teacher3k-262k-v1/data-v5/train.parquet`, `sha256:86452a28af5c77b48c1e53e13a6d8fd9d483eec350d22214cd7717127502a7b1` |
| Full corpus manifest | `sha256:d0641cc7baf082c0b1d9bd494112ad8562b9c332cb5be5a7696249d9101a96a8` |
| Full corpus shape | 3,679 rows, 2,886 sessions, 176,654 assistant responses, 57,384,881 supervised tokens |

## Exact jobs and immutable bundles

| v | RayJob / API run ID / RayJob UID | Nodes | Plan / bundle / runtime SHA-256 | Terminal evidence |
|---|---|---:|---|---|
| 1 | `chris-q38-t3k262-can-v1-e208afe9` / `e208afe9-29d2-44d3-bb62-fa63aa353c3f` / `8f443f46-2fac-499f-95b7-53e06918fbc6` | 2 | `4c780015b4ac6bafc2a1fae36a2bfd504b578a2060c1a46ac7e5cc6b7bc2c8c2` / `cb0059c65aa66e768edf6e188c761a2ea0f54abeb55ad5d0b0146ffdf53c54be` / `c5ed919dd3b08c68ad1213547949e36df7645f81a24113aeb0123ca1d02c2a08` | OOM; no completed-update receipt |
| 2 | `chris-q38-t3k262-can-v2-5d72cb07` / `5d72cb07-f954-48c5-8f37-8f48a6aab311` / `09413178-541d-4203-8031-5cf9872e31f2` | 2 | `08372105b3864149b478445e360afcdbca0e98b9db4ea6a5410d478bf2e3d000` / `074bd7a5a769be6ad33d93469d8896ebbac8f7b9ebaf966183075447f781b914` / `4586f2b2c69d2c9510864cfab541553cc210ff8bedfc3a927e88e837cef9113b` | runtime error; no completed-update receipt |
| 3 | `chris-q38-t3k262-can-v3-bb3c42b4` / `bb3c42b4-869c-4d8d-82e7-11176d71c2fc` / `ce8440b0-6766-453b-a02e-5242a54c129b` | 2 | `2c45bd701c6a193cda3c1f2f1af79f09cfb679a59a64c236a585af86acf3ddcc` / `1937f8cc98c1913317fc6ebf81e6939fe9d35a19a53cb5b60b0fba7bdd9b30ea` / `76412a741961ca145a7481e01d8d8bb5474a5b235271c8bc1f14a4493f59e332` | runtime error; no completed-update receipt |
| 4 | `chris-q38-t3k262-can-v4-b7a54616` / `b7a54616-4013-4c96-b458-9c954a92e40a` / `60bc91fb-5b9f-4712-9641-1d9f56ba0bf2` | 2 | `1f1247d422c61c4e532dda46d757484ab7388e5e3536f4a55890c85f6c0ac30f` / `7e6b7a0b27393d3fd78a73ccc247c1e75467d349a70a17fb0f28e1f7da9f3837` / `3f1da2d709a7c470d6ef8990cc5570853ad2fd34f3eacc9b42881cb80dcfb4f4` | OOM; no completed-update receipt |
| 5 | `chris-q38-t3k262-can-v5-4a1c0e94` / `4a1c0e94-dffa-47a0-9cea-d846c47ceb10` / `ff74ab3f-bd1c-4b19-8da4-7677f7d96e7c` | 2 | `2f80f03a7fdd02d87d83e1730df33e0db634d0db71824afcf2d676d79624b943` / `c93b6c876ba4fe406a525631013a0258a3701c6bac15b2f8c0824f95746cce32` / `eb9eb1838ef6b9ef6abbea0902de0a3ffd3e0350d8733f40efc5c7500432d1ab` | actor died; no completed-update receipt |
| 6 | `chris-q38-t3k262-can-v6-438fcf54` / `438fcf54-fa6c-4176-ac8e-c2d62ad7baca` / `34818443-f6fc-45f1-8ccd-40eb9ff8331e` | 2 | `3b22e1b7988596ada8656b34893542b83bc36bca1db6668bc70b15b8aa48f6ae` / `28119e6641d4e5a9111498aba4f9455b5dcd331cbc002d7027ced23cf5b52c73` / `ccb3631b84197ef576cd97b2a134ac0a94756fe347976e5b18cc101dfc45374b` | OOM; no SFS-proven completed update |
| 7 | `chris-q38-t3k262-can-v7-bc18f683` / `bc18f683-a2cf-48f9-a495-3a503432b255` / `1617be3f-2d97-4429-bd30-5419f9dad919` | 2 | `e21bbb95636ef339a196b4304972b6e573e387942464a0db4de1e5e28a214826` / `bf22d50d28207c313ed5550104a10eb1b21b09497c9f82703b654eaaa394da82` / `177099b9dab5d3e8b9f59da96e366e9dd94d220ed3af097636d05be8e35fdab6` | OOM; no completed-update receipt |
| 8 | `chris-q38-t3k262-can-v8-3919e0ac` / `3919e0ac-37ae-460f-b005-b9680d081db2` / `e143e6a3-9720-4ee0-9f94-9649c37076a8` | 2 | `11c0cbdc225275dcbbe5eaec78c9e00a6f7a25cd0a146ab1a415d7ff7699287a` / `9daa959aef026a68057e19423c6581ad5fe0b876d7222ddd3840388f3c400b01` / `26128df9f5da858be74cca627b08fe052616853ebe730c1b1d70f562f3e89529` | OOM; no completed-update receipt |
| 9 | `chris-q38-t3k262-can-v9-0154095e` / `0154095e-cb8e-4b7b-be1b-4a9792bdfe4a` / `6ecacaac-7ec5-4fc9-b71c-43abe44f6e0f` | 4 | `ca903f9023177fa591ef2cb75e705ce08b57dd541759d080380fc68c9ed8b035` / `6a97654695f89efbcd575804115191070ba4ff28a251472c71d14de35bc2b0b2` / `26128df9f5da858be74cca627b08fe052616853ebe730c1b1d70f562f3e89529` | group-2 OOM; no completed-update receipt |
| 10 | `chris-q38-t3k262-can-v10-aec33804` / `aec33804-9537-4c38-b09c-045299d08a6e` / `8f564225-35d2-458f-adb1-3e6dd1ba5bcf` | 8 | `b4dc777713fb3385f7c276f4168223201a7d2b86559dd1f5f064105bdefc8c2b` / `c5fa02bc293209874c53b17df79aecb8cd32759527990d2f2653fc139e01bb9e` / `ee53668cf733cadf209dfac42fcfaae1c855314f6a355dfbc6595927a76a3c45` | OOM; no SFS-proven completed update |
| 11 | `chris-q38-t3k262-can-v11-73fa80d4` / `73fa80d4-203f-4d9d-8405-9831c1955d5e` / `d96add76-e22c-4330-92c6-5f353f80ab47` | 8 | `1b54c0cadbc240c3c709615256ee3651ea48332cfec7d1b3de2a5584a5d4a833` / `81280fcf1c501ddead3f8b0ec52d4474f5416f5d8fb30cb6ff0f19634380e833` / `b6b81c9876ddf8379a0837a0aaf5f0426acf8ccc4388d79bd93bad94c59e38d3` | OOM; no completed-update receipt |
| 12 | `chris-q38-t3k262-can-v12-a421f0a3` / `a421f0a3-5ed2-48cb-99ee-f830fa395cb7` / `36f915ca-d882-46b3-9d8b-5e4e2bfddae7` | 8 | `3b9e81acb301327c40007a40ab13c3bf5c164152655fb49f96a8db52eb117690` / `e1eb27e77aa42088feac8913de75414910fe919e1f818dfbe5a086e4a210b5b8` / `b6b81c9876ddf8379a0837a0aaf5f0426acf8ccc4388d79bd93bad94c59e38d3` | one update and native checkpoint; mechanics only |
| 13 | `chris-q38-t3k262-full-v13-e5a9b2a8` / `e5a9b2a8-8156-4aa8-ab56-16e02e893796` / `b43ad3aa-3a4a-4054-9822-64217b755a66` | 8 | `74082500b8c66a557d00b19fcc6d877a6c9e7ac9eed9ef7bcc9a687740ca190b` / `20fee00da15ddf472481342f47b7eab8925e92bab1d51fb6cdac57140dc4b02c` / `b6b81c9876ddf8379a0837a0aaf5f0426acf8ccc4388d79bd93bad94c59e38d3` | progress step 12; durable sealed/exported step 10; run failed |
| 14 | `chris-q38-t3k262-resume-v14-b91e5d53` / `b91e5d53-fbef-48ff-9004-da200a5eacb5` / `d40119db-dd8c-466d-a8a4-a4180c8753f3` | 8 | `a225b148a7d91f8a154c09f4afce787a54d17189ba125061fe2535ad3deb5b2a` / `fd76c974af5b6fc820e8b77aeab54787d2aff1af5b665521a1da8ff9bd9d1651` / `365d1b5260acf358181a56504f620e7af86a3a0ad29cda46ea1607f3d31f9f4e` | exact step-10 restore; OOM attempting step 11 |

The v12 parent plan was never committed as a run config. Its authoritative
copies are the retained RayJob bundle and
`/mnt/sfs/jobs/chris-q38-t3k262-can-v12/.runtime/plan.json`. Git commit
`519da7040ef93ecbd606281fed997a221edc620a` documents its plan SHA. The exact
runtime is Git blob `674a8db43ee09c1af8eebee71400f82592f00ad7` at commit
`45c04d709f855e20d931456233b85f427558525f`.

## Sanitized receipt inventory

v1-v11 each contain exactly `STARTED.json`, `DEVICE_INITIALIZED.json`,
`WANDB.json`, `WATCHDOG.json`, `RUNTIME_STAGE.json`, `FAILURE_STAGE.json`, and
`FAILED.json`. None contains `PROGRESS.json`, a checkpoint receipt, a checkpoint
payload, an export, a seal, a reload receipt, or an acceptance receipt.
`FAILURE_STAGE.optimizer_step=1` identifies the logical step being attempted;
it does not prove an optimizer update completed.

| v | Failure-stage file / canonical self-digest | Failed-receipt file / canonical self-digest |
|---|---|---|
| 1 | `cbdaf490af7ef9061bf055cd2008cb8e57aa8b4515097b0efac89158912cd494` / `28cb82773dd8c8032b4fcdc99cfa70046b864aeca6530f3c213a24ede2b868f3` | `8b3cfc8ea3776fcb3deb54257b26a0f58c50836c2728b0dadf823bfbbdf9e4dd` / `7c08516904de70335adc9016e477a8079018eb5787b832c27c7d38c47108622e` |
| 2 | `cf2680442bea091340b8f594d9fb01f6e6eb2e72e51d421bdc48aec161341741` / `ef568cd0b60c160895ab4a41424e01e946fd5a33c2192afed02b1cf21223f8ce` | `434b27e31df5ba4296e6218e8ba882f47b27269b8cb8ce1f6bdd291d72b30ad2` / `0de10cb58619040a26d59f7ba1f44aebb8e85704b914218f750e25b1ad1ac3d6` |
| 3 | `f6f197adaa71f23d148de26ac862a13934d4b69beddf63048ecc889c46ff44b8` / `6c130166f63d63684d040da5eb83a5c9f7379d439ce28703321655be41bdf113` | `e62ba26eb935e3d8312612b547ec7fcfe8f85534b353ac8bfe1cc85a4c7dbcc3` / `fadd834da794133af0e6e85b418b319141517089fc662306eb37f2ca65b6364d` |
| 4 | `4ac740532cbfdb07ffe72b68dacdcb95702444560f4b6ba72bd33a626c7feda4` / `6e2a647417eb0ebb12db61add1d75b06dc1d97c535c117b437ad348cc20ce625` | `4ec548a118b1eb0797da3131c57e5d8cbb7e10aaeac3af426e3b5c2aae51bae4` / `011dab8c7750dffe70688063ff2bfdcf881ac1dcf5e3984c90b862a68769a902` |
| 5 | `4aee24919ce8585e70c3a292a5f778a338b3d9309d1432306b1ac42bfa248946` / `6606060a4dd207adbce1c483f646f95a7c2c35276adc1e85e51c56e473e5eaef` | `9f5ce2f5fbd9e04cbae2c9fad8433277e520636c54c3e28665ae7751cf82fb86` / `26df226038c360bfd54acf37c96d5257a95ffd8119cd2f4e1b5a8ac34845e7fe` |
| 6 | `61a3364fbb68e86e5112a328e3d687c24bd50b06343c912c8aa679b176d7a7ef` / `fa514453e4ffa02344def61e4e3f032e0c0e03c2431ca90381e2b212a4be8056` | `aac360e86242144c39929ebe9bc9ae21338ec4f489258d0338e9f07a24c2cf7e` / `340f35cdc415bf26473880113e03e15f1c2bf9090ee50a193a1dc560dca95605` |
| 7 | `88bc80edcbfa3109b708e7d340e75c33323a1b80dbce6148ad635f3c6c3cd9e3` / `b11d172705f46714cddafa1f8f89079dc3f2fb649f40327694568b69d3cf4d02` | `211c402c4c4f3e4a58e9e26a6094a756c079bb9a2fd7060c5ddf2e51c8797d10` / `f872720946ac043ae074ecd50807554b4114a5fdd7744ecd1f8c738cb60c9ed7` |
| 8 | `c2dd868261dc73420722d54f15d5628fa8a09e9b1fcdd99d0fdccd7e6538bdf6` / `13e8258d4f767ddab6e75e7d173728090e7389e0640961698d8b04907cdd14bc` | `fd40f34b1153bd96a650a9379a487b723b824ff2c84d5602bc82c3d8717af80d` / `1df5da4f8ef7bf32688ff7129a15cc2e7f699b0e44643a2b6c4d41f6653090f9` |
| 9 | `76b33d01eb96d1a20bd6136aca09dfb39b1a257379f3c83dc76724b4c1849172` / `af8b440ca7aab0c1d20bb3ef5457780f9d8c0cf392143ee2c903d4a8fd6c21cc` | `ec358991d4b81802e554aafef5c4d3e92884d7577cab949b7a68dcbaa71aea37` / `55c23253ee7c5f9855f9f26feed5a518b39ee19c29e8d8e41485667feb538377` |
| 10 | `e613f7fad45cfea3a6f89c8af386ac8414d7c4d772a23129607f4a06039fe991` / `9da9863fc2fdf4a192ee8eafe4127b5b63847f95495b56c9fac6438e6374d352` | `40dcd6b8350d309dfbe8faa0311ff1a4deb6f9c8e90b3e4fd249aa41f4551bdc` / `6b5dc1533d82d59f9de662db089406a937e82ac9b2f2a5aac42acebb43383752` |
| 11 | `1f13984752dab0084823c966b9c8a0d1baece12c1e8a341a19ad79e4efde880c` / `04573f276eb5a2cc015fafb9cccf2819ea0f46b5a749b1ffe9b85865ac603900` | `3f0c48380c09d3c24271361cb70551148da67a859a46c2eb51ff0d36e34383c2` / `e9d049e25a1760711543c954c4243a32349c00d8e1449beaf122ed18b9e38297` |

The stronger receipts are:

| Run | Receipt | Stored-file SHA-256 | Canonical self-digest | Meaning |
|---|---|---|---|---|
| v12 | `PROGRESS.json` | `8d652eba11d22d99a1e8422c8cf3c3278850da565c5b3553718253591bc1dc2f` | `11762466435d2afe0a5ba32b56c919ce0d02198ee7dac80c4793529c5c5ae1f2` | optimizer step 1 |
| v12 | `TRAINING_PAUSED.json` | `399b18c57e03c9c6c6429a4d64bf4c31da24ed88e050eff1facf54e5886d1c5a` | `0b00877a5c40ec25fc1da622177ede0c4c35e3e0febe7e07563416b3ff42e919` | one update, 1,870,343 supervised tokens, export pending |
| v12 | `checkpoint_receipts/step-000001.json` | `834b1c11c57c64806b71b583e9c9b916f73bdc8fca868a60f0be0bd0e866f6af` | `3f0cbcd958121499859560f521a8b2f262ffc5d20c6be3cdc472c68f65706758` | native step-1 checkpoint: 201 files, 324,750,377,772 bytes |
| v13 | `PROGRESS.json` | `6063751e4a8d54a00b461adf2484f7e74861988555725af1acbec17dac58eab3` | `a27ab95768112873b785766015d9e9bca5736c9e7893901079c845cb37067ee8` | progress reached step 12 |
| v13 | `checkpoint_receipts/step-000010.json` | `276662ce318fb46858567f327c57bffad743c384b4723b6ac86fb31dc3344ac2` | `188fcd19d9c3aa57e2aceeceab25b919fc1f7fdc48ea01015c5f56f6aa9c4ab4` | durable step 10; 10,440,020 supervised tokens |
| v13 | `checkpoint-seals-step10-v1/step-10.json` | `caa5ee506185af522c81f8f2d7a353deebba6fee56acf98f55d6087603590b0c` | `0c8488a94afe900c42c749ba536ab24df1ede5634f4c8ea1e37b38ac1817a090` | sealed 64-rank checkpoint; GPU reload false |
| v13 | `hf-export-step10-v1/EXPORT.json` | `184e6e6ad1b073e0974ef4820a63c5ef2bbfb8b6acf999ad3bb6a0f692ac9d66` | `841f37d0cc4627b549d97352bef9ca636096d9f7f34f9de7241ad189957f9ae6` | zero-update BF16 export; tensors reopened equal; GPU reload false |
| v13 | `hf-export-check-step10-cpu-v1.json` | `1fc02f641cb0e5dc6439956dbf0d8e30818b636b986c218edbf87acb970d363f` | `3616d39d5ace43604df19968f6d690377cc9f56b047844dc6a423efeb92ede3c` | CPU reopen passed; synthetic only; not serving-qualified |
| v14 | `recovery-preflight-v14/PREFLIGHT.json` | `4fee12a3ad5fcde76a624e2ac86c70247e17a948fed53dd6ca05e856bf4b0741` | `3418a6e33b7fba97b4734a778beec691ec2d9f5bf42d9e6cfa4b740132639662` | zero-GPU source/checkpoint preflight |
| v14 | `RECOVERED.json` | `96bb2389180a4a6808436d730248ffdb5e6041fda8b3741c2aa1a1b242437ed4` | `12938a4945475490d515e60675118eb63097bba10573f5930e733b3b8f07f4b7` | all 64 ranks restored step 10, optimizer, scheduler, sampler |
| v14 | `FAILURE_STAGE.json` | `216880812e998424654fe6f2b8ccef9ccb81fac3bef7b88e0a1f6383ff98f914` | `fce8825d4f02556fe1e3f9384178d8301f1e8638d5efef4d1da553f35dd4e813` | OOM attempting logical step 11 |

All canonical self-digests above were independently recomputed from the
sanitized JSON.

## Minimal four-node qualification hypothesis

The smallest useful experiment is an exact derivative of v12:

| Field | v12 parent | Four-node hypothesis |
|---|---:|---:|
| Nodes | 8 | 4 |
| GPUs per node | 8 | 8 |
| Global batch | 64 | 32 |
| Microbatch per GPU | 1 | 1 |
| Max steps for 112 rows and one epoch | 2 | 4 |
| Pause after step | 1 | 1 |
| Layer checkpoint group | 1 | 1 |
| GDN chunk tokens | 512 | 512 |
| LM / MLP / RMSNorm chunk tokens | 1024 | 1024 |

Model, image, data, split, learning rate, seed, 262,144-token context,
checkpoint interval 1, retention 2, and sequence parallel size 1 stay exact.
Batch 32 is derived from one microbatch on each of 32 ranks; `max_steps=4` is
`ceil(112 / 32)` for one epoch. Keeping batch 64 would introduce two gradient
accumulation rounds and therefore would not be this minimal hypothesis.

This is **not** a qualified or launchable recipe. v9 established that a
different four-node/group-2 shape OOMs; it did not test this group-1 derivative.
The missing evidence is one finite four-node/group-1 optimizer update followed
by a sealed native checkpoint, zero-update export, independent GPU reload and
parity check, and confirmed resource release.

## Current-main implementation gap

At this observation, `origin/main` was
`9859a27d09b4fd881dfe04bbf9a1dcf7413083b8`.

- `training/sft.py` SHA-256
  `447dcaac2b610c1b6c124a13e7d541edc4c3145d26d8ff31577d75b37d8dd67d`
  does not accept or materialize the historical sequence-parallel, GDN chunk,
  LM/MLP/RMSNorm chunk, or layer-checkpoint-group controls.
- `training/sft_runtime.py` SHA-256
  `8cb671f377d089e1303248e237f386c256b212b15e41dadc07ac49cf2695ee17`
  does not implement those historical mechanisms.
- The exact v12 runtime is not patch-equivalent to current main. Its key
  long-context chain is `e940c76ded273fb1a3d097516b0b3bf82c61cb06`,
  `7e1356a516c350dabbae07d5ebbe80690de254e1`,
  `4b0b0b41416f67287e92a4cd551588d794e82503`, then
  `45c04d709f855e20d931456233b85f427558525f`.

Before any qualification request can be prepared, current main must either
deliberately port and test those mechanics or explicitly support a reviewed,
immutable historical runtime bundle. It also needs a durable current-main data
manifest for the exact capacity-v5 corpus and a run config that states only the
hypothesis deltas above. A later server preview must prove the exact root
annotation `fleet.ai/failure-alerts: "off"`, c1 priority, four 8-GPU nodes,
requeue disabled, and a create-once output. This report authorizes no launch.

## Serving evidence caveat

Git commit `3165af38da301e718acee65d240eee5e9f873b9e` records that historical route
`chris-q38-t3k262-s10-web-v1` loaded weights and served with zero restarts. The
record does not cryptographically bind that route's payload revision to the v13
seal and export receipts. It is additional mechanics evidence, not an
independent checkpoint-reload or scientific-acceptance gate.
