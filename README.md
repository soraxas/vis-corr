# Correspondence visualizer

## Install dependencies

Use Python 3.12+ with `uv`: [link](https://docs.astral.sh/uv/getting-started/installation/):


```bash
uv run main.py
```

## Run the viewer

Or you can replace your src/ref pcd

```bash
uv run main.py \
  --corr corr.pkl \
  --ref pcd_1.ply \
  --src pcd_2.ply
```

## Control viewer

using your <kbd>1</kbd>, <kbd>2</kbd>, <kbd>3</kbd>, <kbd>4</kbd> to control the mode (see on-screen instruction)
