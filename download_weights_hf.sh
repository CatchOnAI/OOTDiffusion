"""Download weights from HF repo."""

pip install --upgrade huggingface_hub
git config --global credential.helper store

huggingface-cli login
git clone https://huggingface.co/levihsu/OOTDiffusion
cp -r ./OOTDiffusion/checkpoints ./