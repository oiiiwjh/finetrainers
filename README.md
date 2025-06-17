## installation

``` bash
    conda create -n finetrainers python=3.12
    conda activate finetrainers
    pip install -r requirements.txt

    ## leaky
    # pip install av
    # pip install ftfy

    git clone https://github.com/huggingface/diffusers
    cd diffusers
    python setup.py install

    # ln
    ln -s /share/wjh/datasets/ .
    ln -s /share/wjh/checkpoints/finetrainers_ckpts
    ln -s /share/wjh/logs/finetrainers/ ./logs
```