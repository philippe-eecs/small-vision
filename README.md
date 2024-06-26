# Unified Auto-Encoding with Masked Diffusion
This is the offical Jax implementation of [Unified Mask Diffusion](https://arxiv.org/abs/2406.17688). You can use this codebase to train MAE, UMD, and DiT. It also includes auto-evaluation for few-shot linear probing and FID/IS scores for generation.

This codebase is largely pulled from [Big Vision](https://github.com/google-research/big_vision) codebase. It implementes a generic auto-encoder class that was developed on imagenet-2012 and meant to be used for self-supervised learning on the 100-shot imagenet probing. It also allows self-supervised methods to be fine-tuned on labels for class-conditional generation.

Please visit the [Big Vision](https://github.com/google-research/big_vision) repo for more details on how to setup TPUs and the workflow of the codebase. This experience will be best if you have access to TPUs, but the original experiments done in this paper were tested on 64x64 imagenet with GPUs.

Weights + Colab to play around coming soon! 

Feel free to bring up a github issue or send me an email if you have any questions or comments! philippehansen@utexas.edu 

# Important File Locations

- Main Config File `big_vision/configs/ae_i1k.py`
- Auto-Encoder Implementation `big_vision/models/ae.py`
- Auto-Encoder Trainer `big_vision/trainers/train_ae.py`
- FID/IS Implementation `big_vision/evaluators/fid.py`

# Setting things up

You will need to download imagenet-2012 and preprocess it into the [tfds format](https://github.com/google-research/big_vision?tab=readme-ov-file#preparing-tfds-data) onto a google bucket. Then you will need to setup your [TPU pod](https://github.com/google-research/big_vision?tab=readme-ov-file#cloud-tpu-vm-setup). 

You can access TPUs easily through the [TRC program](https://sites.research.google/trc/about/) if you are an academic looking to do research!

This code base was also extensively tested on GPUs with 64x64 Imagenet. Simply modify the TFDS_DATA_DIR/workdir to the one that is local to your machine/cluster. Instead of `bash big_vision/run_tpu.sh big_vision.train_tpu` use `python3 -m big_vision.train`. I've included an ibrun script that was tested with an distributed multiprocess HPC with A100s, but it hasn't been tested in a while.

# Pre-Training Commands for Self-Supervised Experiments

Commands to reporduce self-supervised experiments in UMD. 100-Shot Linear probing is automatically evaluated during pre-training every 10k gradient steps. Also evaluated are validation mask and denoising errors with visualizations. Each reported number was taken at 800 Epochs.

- UMD-B/4 `TFDS_DATA_DIR=gs://{BUCKET_NAME}/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train_tpu --config big_vision/configs/ae_i1k.py:variant=:B/4,size=64  --workdir gs://{BUCKET_NAME}/checkpoints/UMD_B_4 --key {WANDB_KEY_HERE}` (100 Shot LP - 31.8%)
- MAE-B/4 `TFDS_DATA_DIR=gs://{BUCKET_NAME}/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train_tpu --config big_vision/configs/ae_i1k.py:variant=:B/4,size=64,batch_size=4096,adaln=False  --workdir gs://{BUCKET_NAME}/checkpoints/MAE_B_4 --key {WANDB_KEY_HERE}` (100 Shot LP - 36.6%)
- DiT-B/4 `TFDS_DATA_DIR=gs://{BUCKET_NAME}/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train_tpu --config big_vision/configs/ae_i1k.py:variant=:B/4,size=64,no_noise_prob=0.0,mask_ratio=0.0  --workdir gs://{BUCKET_NAME}/checkpoints/DiT_B_4 --key {WANDB_KEY_HERE}` (100 Shot LP - 25.6%)
- MaskDiT-B/4 `TFDS_DATA_DIR=gs://{BUCKET_NAME}/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train_tpu --config big_vision/configs/ae_i1k.py:variant=:B/4,size=64,no_noise_prob=0.0,mask_ratio=0.5  --workdir gs://{BUCKET_NAME}/checkpoints/MaskDiT_B_4 --key {WANDB_KEY_HERE}` (100 Shot LP 26.3%)
- Latent-UMD-L/2 `TFDS_DATA_DIR=gs://{BUCKET_NAME}/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train_tpu --config big_vision/configs/ae_i1k.py:variant=:L/2,latent_diffusion=True,size=256  --workdir gs://{BUCKET_NAME}/checkpoints/latent_UMD_L_2 --key {WANDB_KEY_HERE}` (100 Shot LP 54.4%)
- MAE-L/16 `TFDS_DATA_DIR=gs://{BUCKET_NAME}/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train_tpu --config big_vision/configs/ae_i1k.py:variant=:L/16,size=256  --workdir gs://{BUCKET_NAME}/checkpoints/MAE_L_16 --key {WANDB_KEY_HERE}` (100 Shot LP 51.1%)

# Fine-Tuning Command for Class-Conditional Generation
Commands to reproduce finetuning generation experiments after self-supervised pre-training. 10k-FID and 10k-IS scores are automatically computed during training with 125 sampling steps.

- UMD-B/4 `TFDS_DATA_DIR=gs://{BUCKET_NAME}/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train_tpu --config big_vision/configs/ae_i1k.py:variant=:B/4,size=64,batch_size=256,beta2=0.999,mask_ratio=0.0,no_noise_prob=0.05,use_labels=True,epochs=50,area_min=95,wd=0.0,finetune=True  --workdir gs://{BUCKET_NAME}/checkpoints/UMD_B_4 --key {WANDB_KEY_HERE}` (FID 19.8, IS 46.9)
- MAE-B/4 `TFDS_DATA_DIR=gs://{BUCKET_NAME}/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train_tpu --config big_vision/configs/ae_i1k.py:variant=:B/4,size=64,adaln=False,batch_size=256,beta2=0.999,mask_ratio=0.0,no_noise_prob=0.05,use_labels=True,epochs=50,area_min=95,wd=0.0,finetune=True  --workdir gs://{BUCKET_NAME}/checkpoints/MAE_B_4 --key {WANDB_KEY_HERE}` (FID 26.8, IS 18.5)
- DiT-B/4 `TFDS_DATA_DIR=gs://{BUCKET_NAME}/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train_tpu --config big_vision/configs/ae_i1k.py:variant=:B/4,size=64,batch_size=256,beta2=0.999,mask_ratio=0.0,no_noise_prob=0.00,use_labels=True,epochs=50,area_min=95,wd=0.0,finetune=True  --workdir gs://{BUCKET_NAME}/checkpoints/DiT_B_4 --key {WANDB_KEY_HERE}` (FID 18.9, IS 46.9)
- MaskDiT-B/4 `TFDS_DATA_DIR=gs://{BUCKET_NAME}/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train_tpu --config big_vision/configs/ae_i1k.py:variant=:B/4,size=64,batch_size=256,beta2=0.999,mask_ratio=0.0,no_noise_prob=0.00,use_labels=True,epochs=50,area_min=95,wd=0.0,finetune=True --workdir gs://{BUCKET_NAME}/checkpoints/MaskDiT_B_4 --key {WANDB_KEY_HERE}` (FID 19.0, IS 43.4)
- Latent-UMD-L/2 `TFDS_DATA_DIR=gs://{BUCKET_NAME}/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train_tpu --config big_vision/configs/ae_i1k.py:variant=:L/2,latent_diffusion=True,size=256,batch_size=256,beta2=0.999,mask_ratio=0.0,no_noise_prob=0.05,use_labels=True,epochs=50,area_min=95,wd=0.0,finetune=True  --workdir gs://{BUCKET_NAME}/checkpoints/latent_UMD_L_2 --key {WANDB_KEY_HERE}` (FID 3.96, IS 212.6)

# Eval Commands
To eval a model on a more through generation benchmark and on transfer learning, run:
`TFDS_DATA_DIR=gs://{BUCKET_NAME}/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train_tpu --config big_vision/configs/eval_ae_i1k.py:variant=:{VARIANT_HERE},size={64/256},adaln={True/False},latent_diffusion={True/False},finetune={True/False},sampling_timesteps={62/125/250},samples={True/False},probe={True/False}  --workdir gs://{BUCKET_NAME}/checkpoints/{CHECKPOINT_TO_EVAL}--key {WANDB_KEY_HERE}`

# Training DiT From Scratch
If you want to train DiT from scratch to test generative capabilities, run this command. 

- Latent-DiT-L/2 `TFDS_DATA_DIR=gs://{BUCKET_NAME}/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train_tpu --config big_vision/configs/ae_i1k.py:variant=:L/2,latent_diffusion=True,size=256,batch_size=256,beta2=0.999,mask_ratio=0.0,no_noise_prob=0.0,use_labels=True,epochs=400,area_min=95,wd=0.0  --workdir gs://{BUCKET_NAME}/checkpoints/latent_DiT_L_2 --key {WANDB_KEY_HERE}` (FID 4.31, IS 274.5)

# Current and future contents

We plan to add implementations of newer ideas to this codebase to keep it up to date with the diffusion literature. We also want to optimize this codebase for speed.
Any suggestions are welcoem!

Planned short-term:
- Rectified Flows.
- Preprocessed VAE Latents to Speed up Latent Diffusion.
- Find ways to reduce computational load of AdaLN modulation. 
- Look into quantization (FSQ).
- 64x64 Supervised Imagenet-1k (meant for small scale training)

# Citing the codebase

If you found this codebase useful for your research, please cite:

```
@misc{hansenestruch2024unifiedautoencodingmaskeddiffusion,
      title={Unified Auto-Encoding with Masked Diffusion}, 
      author={Philippe Hansen-Estruch and Sriram Vishwanath and Amy Zhang and Manan Tomar},
      year={2024},
      eprint={2406.17688},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
      url={https://arxiv.org/abs/2406.17688}, 
}
```

Please also cite the big vision repo as well

```
@misc{big_vision,
  author = {Beyer, Lucas and Zhai, Xiaohua and Kolesnikov, Alexander},
  title = {Big Vision},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/google-research/big_vision}}
}
```
