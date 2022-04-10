# Visual Attention Emerges from Recurrent Sparse Reconstruction

This codebase is built upon the official code of "Towards Robust Vision Transformer".

# Usage

Install PyTorch 1.7.0+ and torchvision 0.8.1+ from the official website.

`requirements.txt` lists all the dependencies:
```
pip install -r requirements.txt
```
In addition, please also install the magickwand library:
```
apt-get install libmagickwand-dev
```

## Training

Take RVT-S with VARS-SD for an example. We use single node with 8 gpus for training:

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345  main.py --model rvt_small --data-path path/to/imagenet  --output_dir output/here  --num_workers 8 --batch-size 128 --attention vars_sd
```

To train models with different scales or different attention algorithms, please change the arguments `--model` and `--attention`. 

## Testing

```
python main.py --model rvt_small --data-path path/to/imagenet --eval --resume path/to/checkpoint --attention vars_sd
```

To enable robustness evaluation, please add one of `--inc_path /path/to/imagenet-c`, `--ina_path /path/to/imagenet-a`, `--inr_path /path/to/imagenet-r` or `--insk_path /path/to/imagenet-sketch` to test [ImageNet-C](https://github.com/hendrycks/robustness), [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-R](https://github.com/hendrycks/imagenet-r) or [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch).

If you want to test the accuracy under adversarial attackers, please add `--fgsm_test` or `--pgd_test`.


