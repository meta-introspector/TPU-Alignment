pip3 install --upgrade transformers datasets sentencepiece peft pandas pyarrow
pip install torch~=2.1.0 --index-url https://download.pytorch.org/whl/cpu 
pip install torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html 
pip uninstall tensorflow -y 
