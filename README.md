<div align="center">

# AwareVLN: Reasoning with Self-awareness for Vision-Language Navigation

<p align="center" style="margin:1.4em 0 0.8em;">
  <a href="https://gwxuan.github.io/AwareVLN/"><img src="https://img.shields.io/badge/Project_Page-AwareVLN-2EA44F?style=flat&labelColor=555555" alt="Project Page"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2605.22816"><img src="https://img.shields.io/badge/arXiv-2605.22816-B31B1B?style=flat&labelColor=555555&logo=arxiv&logoColor=white" alt="Paper"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/gwx22/AwareVLN"><img src="https://img.shields.io/badge/Dataset-AwareVLN-FFD63A?style=flat&labelColor=555555" alt="Dataset"></a>
  &nbsp;
  <a href="https://huggingface.co/gwx22/AwareVLN-ck"><img src="https://img.shields.io/badge/Checkpoint-AwareVLN-FFD63A?style=flat&labelColor=555555" alt="Checkpoint"></a>
</p>

<p style="font-size:1.5em;font-weight:600;letter-spacing:0.03em;color:#555;margin:0.75em 0 0;"><strong>CVPR 2026</strong></p>

<p align="center">
  <img src="assets/teaser.png" width="800">
</p>

</div>

## 💡 Introduction

AwareVLN equips VLN with sparse **self-aware reasoning** at key navigation nodes. A unified VLM switches between `[REASON]` and `[ACT]`; an automatic data engine provides scalable supervision.


## 🚀 Training
### Installation
To build the training environment, run:
```bash
./environment_setup.sh awarevln
conda activate awarevln
```

### Dataset
Training annotations of reasoning are produced by our **automatic data engine**, which labels sparse **self-aware reasoning** at key nodes. Download from [Dataset](https://huggingface.co/datasets/gwx22/AwareVLN) and extract `videos.tar.gz` in each subfolder.

* **r2r / rxr:** Trajectories from rollouts of existing policy, with corrections when needed; reasoning annotations from our data engine.
* **r2rfollow / rxrfollow:** Trajectories that follow expert paths; reasoning annotations from our data engine.

* **Human:** Not included. Follow [NaVILA-Dataset](https://huggingface.co/datasets/a8cheng/NaVILA-Dataset): use **[video IDs](https://huggingface.co/datasets/a8cheng/NaVILA-Dataset/blob/main/Human/video_ids.txt)**, download with `yt-dlp`, extract frames via `scripts/extract_rawframes.py` in the [NaVILA repo](https://github.com/a8cheng/NaVILA).

The data should have structure like:
```graphql
AwareVLN-Dataset
├─ reason
|   ├─ r2r
|   |    ├─ _anno_cot
|   |    |    ├─ annotations_shuffle_uni.json
|   |    |    ├─ cot_new.json
|   |    ├─ videos
|   ├─ rxr
|   |    ├─ ...
|   ├─ r2rfollow
|   |    ├─ ...
|   ├─ rxrfollow
|   |    ├─ ...
├─ Human
|   ├─ raw_frames
|   |    ├─ <video_id>
|   |    |    ├─ 0001.jpg
|   |    |    ├─ ...
|   ├─ annotations_shuffled.json
```

### Training
We start from **NaVILA-style VILA** (Llama-3 8B + SigLIP + mm_projector, 8 frames), and fine-tune with our reasoning data to learn **self-aware reasoning**. The pretrained model and our trained **AwareVLN weights** are available [here](https://huggingface.co/gwx22/AwareVLN-ck).

```bash
export AWAREVLN_DATA_ROOT=/path/to/data
bash scripts/train/sft_8frames.sh
```


## 📊 Evaluation

### Installation

This repository builds on [VLN-CE](https://github.com/jacobkrantz/VLN-CE), which relies on older versions of [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7) and [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7).

1. Create conda env `awarevln-eval` (Python 3.10)

```bash
conda create -n awarevln-eval python=3.10
conda activate awarevln-eval
```

2. Build Habitat-Sim & Lab (v0.1.7) from **source**

Follow the [VLN-CE setup guide](https://github.com/jacobkrantz/VLN-CE?tab=readme-ov-file#setup).
To resolve NumPy compatibility issues, apply the following hotfix:
```bash
python evaluation/scripts/habitat_sim_autofix.py # replace habitat_sim/utils/common.py
```

3. Install VLN-CE dependencies
```bash
pip install -r evaluation/requirements.txt
```

4. Install VILA dependencies
```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install -e .
pip install -e ".[train]"
pip install -e ".[eval]"

pip install git+https://github.com/huggingface/transformers@v4.37.2
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/
```

5. Fix WebDataset version
```bash
pip install webdataset==0.1.103
```

### Data
Follow [VLN-CE](https://github.com/jacobkrantz/VLN-CE) and download R2R / RxR annotations and MP3D scenes under `evaluation/data/` (Val-Unseen, monocular RGB):
```graphql
evaluation/data/datasets
├─ RxR_VLNCE_v0
|   ├─ val_unseen
|   |    ├─ val_unseen_guide.json.gz
|   |    ├─ ...
├─ R2R_VLNCE_v1-3_preprocessed
|   ├─ val_unseen
|   |    ├─ val_unseen.json.gz
|   |    ├─ ...
evaluation/data/scene_datasets
├─ mp3d
|   ├─ 17DRP5sb8fy
|   |    ├─ 17DRP5sb8fy.glb
|   |    ├─ ...
```

### Running Evaluation
1. Trained **AwareVLN weights** are available [here](https://huggingface.co/gwx22/AwareVLN-ck), or use your own `outputs/`.
2. Run evaluation on R2R-CE using:
```bash
cd evaluation
bash scripts/eval/r2r.sh
```
Examples:
* Single GPU:
    ```bash
    MODEL_PATH=../ck/awarevln TOTAL_CHUNKS=1 GPU_LIST="0" bash scripts/eval/r2r.sh
    ```
* Multiple GPUs (e.g., 8 GPUs):
    ```bash
    MODEL_PATH=../ck/awarevln TOTAL_CHUNKS=8 GPU_LIST="0,1,2,3,4,5,6,7" bash scripts/eval/r2r.sh
    ```
3. Run evaluation on RxR-CE using:
```bash
MODEL_PATH=../ck/awarevln bash scripts/eval/rxr.sh
```
4. Results are saved under `evaluation/eval_awarevln/<CKPT_NAME>/`. Metrics are aggregated automatically; to re-run:
```bash
python scripts/eval_jsons.py eval_awarevln/awarevln/VLN-CE-v1/val_unseen NUM_CHUNKS
python scripts/eval_jsons.py eval_awarevln/awarevln/RxR-VLN-CE-v1/val_unseen NUM_CHUNKS
```

## 🎬 Demo

AwareVLN performs structured reasoning during navigation—for example, detecting a misinterpreted turn and issuing a corrective plan, or recognizing a completed subtask and planning the next phase aligned with the instruction.

<p align="center">
  <img src="assets/demo.gif" width="600">
</p>


_______________________________________________________________

## 📜 Citation

```bibtex
@article{guo2026awarevln,
      title={AwareVLN: Reasoning with Self-awareness for Vision-Language Navigation}, 
      author={Wenxuan Guo and Xiuwei Xu and Yichen Liu and Xiangyu Li and Hang Yin and Huangxing Chen and Wenzhao Zheng and Jianjiang Feng and Jie Zhou and Jiwen Lu},
      journal={arXiv preprint arXiv:2605.22816},
      year={2026},
      url={https://arxiv.org/abs/2605.22816}, 
}
```
