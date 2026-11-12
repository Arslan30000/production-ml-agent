# Stage 1 DVC run

I initialized DVC in the Git repository, generated the synthetic equipment dataset, added it to DVC, configured a local DVC remote, ran the training pipeline, and pushed the DVC objects. I kept the [terminal screenshot](https://chatgpt.com/library?file_id=libfile_486adac6c1ac8191ae1715e632790a13&account_id=personal) with this record.

The run reported three files pushed and `Data and pipelines are up to date.` The holdout results were accuracy 0.85, precision 0.8113, recall 0.8431, and F1 0.8269 on 120 test rows. The dataset SHA-256 recorded by training was `ed16efff36a8916b235b446bd7b73a9f64bd385360ef57526c92703dc8b659c2`. The training process started at Git HEAD `97a84d908f69e0a385b448d190cbce52b1560287`.

The DVC metadata records MD5 `2c05374ccfea6e4cf247f99d0ef3aa19` for `data/raw/equipment.csv`. The remote is a folder beside the repository, so the dataset is reproducible on this WSL machine while that folder is preserved. A new clone on another machine would need a shared remote or a copy of this folder.

I still need to commit the DVC metadata, create a second committed dataset version, compare the two versions, and verify checkout in an isolated worktree. I will record those outputs here after running them.
