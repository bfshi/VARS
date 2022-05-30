# Visual Attention Emerges from Recurrent Sparse Reconstruction (ICML 2022)

### [Baifeng Shi](https://bfshi.github.io), [Yale Song](http://people.csail.mit.edu/yalesong/home/), [Neel Joshi](https://neelj.com/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Xin Wang](https://xinw.ai/)

<img src="VARS.png" alt="drawing" width="600"/>

Codebase of our ICML'22 paper "[Visual Attention Emerges from Recurrent Sparse Reconstruction](https://arxiv.org/abs/2204.10962)".


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

Take RVT-Ti with VARS-D for an example. We use single node with 8 gpus for training:

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345  main.py --model rvt_tiny --data-path path/to/imagenet  --output_dir output/here  --num_workers 8 --batch-size 128 --attention vars_d
```

To train models with different scales or different attention algorithms, please change the arguments `--model` and `--attention`. 

## Testing

```
python main.py --model rvt_tiny --data-path path/to/imagenet --eval --resume path/to/checkpoint --attention vars_d
```

To enable robustness evaluation, please add one of `--inc_path /path/to/imagenet-c`, `--ina_path /path/to/imagenet-a`, `--inr_path /path/to/imagenet-r` or `--insk_path /path/to/imagenet-sketch` to test [ImageNet-C](https://github.com/hendrycks/robustness), [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-R](https://github.com/hendrycks/imagenet-r) or [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch).

If you want to test the accuracy under adversarial attackers, please add `--fgsm_test` or `--pgd_test`.

## Links 

This codebase is built upon the official code of "[Towards Robust Vision Transformer](https://github.com/vtddggg/Robust-Vision-Transformer)".

## Citation
If you found this code helpful, please consider citing our work: 

```bibtext
@article{shi2022visual,
  title={Visual Attention Emerges from Recurrent Sparse Reconstruction},
  author={Shi, Baifeng and Song, Yale and Joshi, Neel and Darrell, Trevor and Wang, Xin},
  journal={arXiv preprint arXiv:2204.10962},
  year={2022}
}
```
