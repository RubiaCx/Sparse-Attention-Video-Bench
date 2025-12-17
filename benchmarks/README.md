# Benchmarks


## Prepare VBench Prompts

```bash
# clone vbench repo
git submodule update --init --recursive
```

```bash
export OPENAI_API_KEY=sk-proj-1234567890

python prepare_vbench_prompts.py --dimension overall_consistency
python prepare_vbench_prompts.py --dimension temporal_style
python prepare_vbench_prompts.py --dimension subject_consistency
python prepare_vbench_prompts.py --dimension spatial_relationship
python prepare_vbench_prompts.py --dimension appearance_style
```

## Run Benchmarks

```bash
bash ./generate_dense_wan14b.sh overall_consistency
bash ./generate_svg_wan14b.sh overall_consistency
```

```bash
vbench evaluate \
    --dimension <dimension> \
    --videos_path <video-folder> \
    --mode=custom_input
```
