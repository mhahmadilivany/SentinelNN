# SentinelNN: A Framework for Fault Resilience Assessment and Enhancement of CNNs

SentinelNN is an open-source framework to analyze and enhance the fault tolerance of CNN models. The key features of SentinelNN are:

* Resilience analysis: SentinelNN provides a set of methods for the importance analysis of channels in convolutional layers, including: *"l1-norm", "vul-gain", "salience", "deepvigor"*
* Structured channel pruning: SentinelNN can prune the less important channels of CNNs, based on a selected resilience analysis method, to reduce the size of CNNs
* Selective channel duplication and correction: SentinelNN duplicates the more important (or vulnerable) channels and applies a correction mechanism to mitigate the effect of faults
* Range restriction: SentinalNN applies a range restriction method (i.e., Ranger) to the ReLU activation function to reduce the error propagation to the CNN outputs

The framework takes a pretrained CNN model (i.e., .pth file) and outputs pruned or hardened model with the same format. For experiments, sentinelNN logs the results in a directory with the model's name and dataset. The user can also conduct fault injection experiments to observe the effectiveness of model hardening.

![SentinelNN](https://github.com/user-attachments/assets/0f04e1fc-6c6c-405c-8a47-239f6c9d5439)


**Requirements:**

This framework is developed and tested with python 3.10, pytorch 1.10.2+cu102, torchvision 0.11.2+cu102

**How to use:**

Here are the commands that the user can employ to use this framework:

* To test a pretrained CNN model:
  ```
  python main.py --model=model_name --dataset=dataset_name --batch-size=batch-size
  ```

This framework supports loading pretrained models for Cifar-10 and Cifar-100 directly from [pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models/tree/master) and also loads models from pytorch hub for ImageNet.

* To prune a CNN model:

  ```
  python main.py --model=model_name --dataset=dataset_name --batch-size=batch-size --is-pruning \
  --pruning-ratio=0.x --importance=method_name
  ```

> The `pruning_ratio` is a value between 0 and 1. The `importance` method name can be: *"l1-norm", "vul-gain", "salience", "deepvigor" or "channel-FI"*.

> In the case of `importance=deepvigor` the derived vulnerability factors by [DeepVigor ](https://github.com/mhahmadilivany/DeepVigor)are saved, for each layer separately, in the corresponding workspace and can be reused.

* The pruned model is saved and can be loaded as:

  ```
  python main.py --model=model_name --dataset=dataset_name --batch-size=batch-size \
  --is-pruned --pruning-ratio=0.x --pruned-checkpoint=./path/to/saved/model.pth
  ```
* To harden a pretrained model:

  ```
  python main.py --model=model_name --dataset=dataset_name --batch-size=batch-size \
    --is-hardening --hardening-ratio=0.x --importance=method_name --clipping=ranger
  ```
* To harden a pruned model:

  ```
  python main.py --model=model_name --dataset=dataset_name --batch-size=batch-size \
  --is-pruned --pruning-ratio=0.x --pruned-checkpoint=path/to/saved/model.pth --is-hardening --hardening-ratio=0.x --importance=method_name --clipping=ranger
  ```
* To load a pruned+hardened model and conduct fault injection into weights:

  ```
  python3.10 main.py --model=model_name --dataset=dataset_name --batch-size=256 \
  --is-pruned --pruning-ratio=0.1 --pruned-checkpoint=path/to/saved/model.pth \
  --is-hardened --hardening-ratio=0.1 --clipping=ranger --hardened-checkpoint=path/to/saved/model.pth \
  --is-FI --BER=0.000005 --repeat=1000
  ```

> This command performs bitflips in convolutional and linear layers with the *Bit Error Rate (BER)* of `0.000005` and repeats the FI campaign `1000 `times and logs the average results.

> While loading a pruned or hardened model, the `pruning_ratio `and `hardening_ratio` should be equal to their initial applied ratios and the corresponding files should be assigned. The framework first loads the initial model, and then modifies its structure with respect to the pruning and hardening ratios, respectively. And finally, it loads the parameters from the saved model.

> SentinelNN supports Ranger at the moment, in the hardening process in which ReLU is protected. We have implemented multiple state-of-the-art activation restriction methods in another framework, called [RReLU](https://github.com/hamidmousavi0/reliable-relu-toolbox/tree/master).

Check out the published paper [here ](https://ieeexplore.ieee.org/document/10616072)or [here](https://arxiv.org/abs/2405.10658). If you use SentinelNN, please cite:

```
@inproceedings{ahmadilivani2024cost-effective,
  author={Ahmadilivani, Mohammad Hasan and Mousavi, Seyedhamidreza and Raik, Jaan and Daneshtalab, Masoud and Jenihhin, Maksim},
  booktitle={2024 IEEE 30th International Symposium on On-Line Testing and Robust System Design (IOLTS)}, 
  title={Cost-Effective Fault Tolerance for CNNs Using Parameter Vulnerability Based Hardening and Pruning}, 
  year={2024},
  pages={1-7},
  keywords={Accuracy; Error analysis; Computational modeling; Fault-tolerant systems; Redundancy; Neural networks},
  doi={10.1109/IOLTS60994.2024.10616072}}
```

Related papers and frameworks:

* [DeepVigor+](https://github.com/mhahmadilivany/DeepVigor) source code, and [paper1](https://ieeexplore.ieee.org/document/10174133) and [paper2](https://arxiv.org/abs/2410.15742)
* [RReLU](https://github.com/hamidmousavi0/reliable-relu-toolbox/tree/master) toolbox source code and its [paper](https://arxiv.org/abs/2406.06313)
