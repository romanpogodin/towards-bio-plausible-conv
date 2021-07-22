# Paper
### Towards Biologically Plausible Convolutional Networks

Roman Pogodin, Yash Mehta, Timothy P. Lillicrap, Peter E. Latham

arXiv: https://arxiv.org/pdf/2106.13031.pdf

# Requirements
Python 3.7.3, PyTorch 1.8, CUDA 10.1.

Note: older versions of PyTorch will not work because we use torch.kron (Kronecker product),
 which was introduced in 1.8. Newer CUDA drivers or Python 3.6+ should work, however.

# Main experiments
The main experiments are run through `experiments.py`. All arguments are provided in `utils/parse_arguments`.

All experiments were run with automatic mixed precision (`--amp` flag).

### CIFAR10/100 and TinyImageNet

To run CIFAR10:
```
python3 experiments.py --dataset CIFAR10 --experiment resnet_cifar --amp --experiment-folder ./  --job-idx final \
    --n-epochs 200 --epoch-decrease-lr 100 150 --optimizer AdamW --batch-size 512 --lr 0.01 --weight-decay 0.0001 \
    --is-locally-connected \
    --share-weights --instant-weight-sharing --weight-sharing-frequency 10 \
    --training-mode test --stat-report-frequency 10 --checkpoint-frequency 50
```
Remove `--is-locally-connected` to make convolutional. 
Remove `--share-weights --instant-weight-sharing --weight-sharing-frequency 10` to remove weight sharing.

The other two datasets need changing of `--dataset CIFAR10` to 
` --dataset CIFAR100` or  `--dataset TinyImageNet`
TinyImageNet also needs `--imagenet-path path`, where `path` is a folder with train and val subfolders.

Note: `--experiment-folder` must exist before executing the script!

### ImageNet

```
python3 experiments.py --dataset ImageNet --resnet18-width-per-group 32 --imagenet-path path \
    --experiment resnet18_imagenet --amp --experiment-folder ./  --job-idx final \
    --n-epochs 200 --epoch-decrease-lr 100 150 --optimizer AdamW --batch-size 256 --lr 0.0005 --weight-decay 0.01 \
    --is-locally-connected \
    --share-weights --instant-weight-sharing --weight-sharing-frequency 10 \
    --training-mode test --stat-report-frequency 10 --checkpoint-frequency 10
```
Remove `--is-locally-connected` to make convolutional. 
Remove `--share-weights --instant-weight-sharing --weight-sharing-frequency 10` to remove weight sharing.
Add `--n-first-conv 1` to make the first layer convolutional.

ImageNet needs `--imagenet-path path`, where `path` is a folder with train and val subfolders.


# Experiments for convergence of weight sharing

Convergence simulations for dynamic weight sharing are done in `dynamic_weight_sharing.py`

`dynamic_weight_sharing.simulate_dynamic` runs the dynamical system that implements SGD. 
`dynamic_weight_sharing.simulate_instant` runs SGD.
