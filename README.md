# Joint Computing, Pushing, and Caching Optimization for Mobile Edge Computing Networks via Soft Actor-Critic Learning
A Deep-Reinforcement Learning Approach for activity optimization in mobile edge computing (MEC) network

<p align="center"> <img src='docs/system.png' align="center" height="300px"> </p>

> [**Joint Computing, Pushing, and Caching Optimization for Mobile Edge Computing Networks Via Soft Actor-Critic Learning**](https://arxiv.org/pdf/2309.15369.pdf)            
> Xiangyu Gao, Yaping Sun, Hao Chen, Xiaodong Xu, and Shuguang Cui <br />
> *arXiv technical report ([arXiv 2309.15369](https://arxiv.org/abs/2309.15369))*
> 
    @ARTICLE{10275097, author={Gao, Xiangyu and Sun, Yaping and Chen, Hao and Xu, Xiaodong and Cui, Shuguang},
         journal={IEEE Internet of Things Journal}, 
         title={Joint Computing, Pushing, and Caching Optimization for Mobile Edge Computing Networks Via Soft Actor-Critic Learning}, 
         year={2023}, volume={}, number={}, pages={1-1}, 
         doi={10.1109/JIOT.2023.3323433}}

> [**Soft Actor-Critic Learning-Based Joint Computing, Pushing, and Caching Framework in MEC Networks**](https://arxiv.org/pdf/2305.12099.pdf)            
> Xiangyu Gao, Yaping Sun, Hao Chen, Xiaodong Xu, and Shuguang Cui <br />
> *arXiv technical report ([arXiv 2305.12099](https://arxiv.org/abs/2305.12099))*
> 
    @misc{gao2023soft, author={Xiangyu Gao and Yaping Sun and Hao Chen and Xiaodong Xu and Shuguang Cui},
         title={Soft Actor-Critic Learning-Based Joint Computing, Pushing, and Caching Framework in MEC Networks},
         year={2023}, eprint={2305.12099}, archivePrefix={arXiv}, primaryClass={cs.IT}
}

<!-- ## Update
***(Dec. 3, 2023) Release the source code and sample data.*** -->

## Contact
Any questions or suggestions are welcome! 

Xiangyu Gao [xygao@uw.edu](mailto:xygao@uw.edu)

## Abstract
Mobile edge computing (MEC) networks bring computing and storage capabilities closer to edge devices, which reduces latency and improves network performance. However, to further reduce transmission and computation costs while satisfying user-perceived quality of experience, a joint optimization in computing, pushing, and caching is needed. In this paper, we formulate the joint-design problem in MEC networks as an infinite-horizon discounted-cost Markov decision process and solve it using a deep reinforcement learning (DRL)-based framework that enables the dynamic orchestration of computing, pushing, and caching. Through the deep networks embedded in the DRL structure, our framework can implicitly predict user future requests and push or cache the appropriate content to effectively enhance system performance. One issue we encountered when considering three functions collectively is the curse of dimensionality for the action space. To address it, we relaxed the discrete action space into a continuous space and then adopted soft actor-critic learning to solve the optimization problem, followed by utilizing a vector quantization method to obtain the desired discrete action. Additionally, an action correction method was proposed to compress the action space further and accelerate the convergence. Our simulations under the setting of a general single-user, single-server MEC network with dynamic transmission link quality demonstrate that the proposed framework effectively decreases transmission bandwidth and computing cost by proactively pushing data on future demand to users and jointly optimizing the three functions. We also conduct extensive parameter tuning analysis, which shows that our approach outperforms the baselines under various parameter settings.

## Use RAMP-CNN

All radar configurations and algorithm configurations are included in [config](config.py).

### Software Requirement and Installation

Python 3.6, pytorch-1.7.1 (please refer to [INSTALL](requirements.txt) to set up libraries.)

### Download Sample Data and Model
1. From below Google Drive link
    ```
    https://drive.google.com/drive/folders/1TGW6BHi5EZsSCtTsJuwYIQVaIWjl8CLY?usp=sharing
    ```

2. Decompress the downloaded files and relocate them as following directory manners:
    ```
    './template_files/slice_sample_data'
    './template_files/train_test_data'
    './results/C3D-20200904-001923'
    ```

## 3D Slicing of Range-Velocity-Angle Data
For convenience, in the sample codes we use the raw ADC data of each frame as input and perform the [Range, Velocity, and Angle FFT](https://github.com/Xiangyu-Gao/mmWave-radar-signal-processing-and-microDoppler-classification) during the process of slicing. Run following codes for 3D slicing.
    
    python slice3d.py
    

The slicing results are the RA slices, RV slices, and VA slices as shown in below figure.
<p align="center"> <img src='docs/slice_viz.png' align="center" height="230px"> </p>

## Train and Test
1. Prepare the input data (RA, RV, and VA slices) and ground truth confidence map for training and testing. Note that the provided training and testing data is in the post-3D slicing format, so you can skip the last step if you used provided data here:
    ```
    python prepare_data.py -m train -dd './data/'
    python prepare_data.py -m test -dd './data/'
    ```
2. Run training:
    ```
    python train_dop.py -m C3D
    ```
    You will get training outputs as follows:
    ```
    No data augmentation
    Number of sequences to train: 1
    Training files length: 111
    Window size: 16
    Number of epoches: 100
    Batch size: 3
    Number of iterations in each epoch: 37
    Cyclic learning rate
    epoch 1, iter 1: loss: 8441.85839844 | load time: 0.0571 | backward time: 3.1147
    epoch 1, iter 2: loss: 8551.98437500 | load time: 0.0509 | backward time: 2.9038
    epoch 1, iter 3: loss: 8019.63525391 | load time: 0.0531 | backward time: 2.9171
    epoch 1, iter 4: loss: 8376.16015625 | load time: 0.0518 | backward time: 2.9146
    ...
    ```
3. Run testing:
    ```
    python test.py -m C3D -md C3D-20200904-001923
    ```
    You will get testing outputs as follows:
    ```
    ['2019_05_28_pm2s012']
    2019_05_28_pm2s012
    Length of testing data: 443
    loading time: 0.02
    finished ra normalization
    finished v normalization
    Testing 2019_05_28_pm2s012/000000-000016... (0)
    2019_05_28_pm2s012/0000000000.jpg inference finished in 0.6654 seconds.
    processing time: 0.98
    loading time: 0.02
    finished ra normalization
    finished v normalization
    Testing 2019_05_28_pm2s012/000002-000018... (0)
    2019_05_28_pm2s012/0000000002.jpg inference finished in 0.4723 seconds.
    ...
    ```
4. Run evaluation:
    ```
    python evaluate.py -md C3D-20200904-001923
    ```
    You will get evaluation outputs as follows:
    ```
    true seq
    ./results/C3D-20200904-001923/2019_05_28_pm2s012/rod_res.txt
    Average Precision  (AP) @[ OLS=0.50:0.90 ] = 0.9245
    Average Recall     (AR) @[ OLS=0.50:0.90 ] = 0.9701
    pedestrian: 1930 dets, 1800 gts
    Average Precision  (AP) @[ OLS=0.50:0.90 ] = 0.9245
    Average Precision  (AP) @[ OLS=0.50      ] = 0.9823
    Average Precision  (AP) @[ OLS=0.60      ] = 0.9823
    Average Precision  (AP) @[ OLS=0.70      ] = 0.9520
    Average Precision  (AP) @[ OLS=0.80      ] = 0.9234
    Average Precision  (AP) @[ OLS=0.90      ] = 0.7349
    Average Recall     (AR) @[ OLS=0.50:0.90 ] = 0.9701
    Average Recall     (AR) @[ OLS=0.50      ] = 1.0000
    Average Recall     (AR) @[ OLS=0.75      ] = 0.9850
    ...
    ```

## Radar Data Augmentation
Run below codes to check the results of 3 proposed data augmentation algorithms: flip, range-translation, and angle-translation.

    python data_aug.py

Below figure shows the performance of doing 10-bins range-translation (move upword), 25-degrees angle-translation (move rightword), and angle flip on original RA images. You may use this codes to develop your radar data augmentation and even generate new datas. 
<p align="center"> <img src='docs/aug_viz.png' align="center" height="230px"> </p>

## License

RAMP-CNN is release under MIT license (see [LICENSE](LICENSE)).

## Acknowlegement
This project is not possible without multiple great opensourced codebases. We list some notable examples below.  

* [microDoppler](https://github.com/Xiangyu-Gao/mmWave-radar-signal-processing-and-microDoppler-classification)
* [rodnet](https://github.com/yizhou-wang/RODNet)

### Description
------------
Reimplementation of [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf) and a deterministic variant of SAC from [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf).

Added another branch for [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf) -> [SAC_V](https://github.com/pranz24/pytorch-soft-actor-critic/tree/SAC_V).

### Requirements
------------
*   [mujoco-py](https://github.com/openai/mujoco-py)
*   [PyTorch](http://pytorch.org/)

### Default Arguments and Usage
------------
### Usage

```
usage: main.py [-h] [--env-name ENV_NAME] [--policy POLICY] [--eval EVAL]
               [--gamma G] [--tau G] [--lr G] [--alpha G]
               [--automatic_entropy_tuning G] [--seed N] [--batch_size N]
               [--num_steps N] [--hidden_size N] [--updates_per_step N]
               [--start_steps N] [--target_update_interval N]
               [--replay_size N] [--cuda]
```

(Note: There is no need for setting Temperature(`--alpha`) if `--automatic_entropy_tuning` is True.)

#### For SAC

```
python main.py --env-name Humanoid-v2 --alpha 0.05
```

#### For SAC (Hard Update)

```
python main.py --env-name Humanoid-v2 --alpha 0.05 --tau 1 --target_update_interval 1000
```

#### For SAC (Deterministic, Hard Update)

```
python main.py --env-name Humanoid-v2 --policy Deterministic --tau 1 --target_update_interval 1000
```

### Arguments
------------
```
PyTorch Soft Actor-Critic Args

optional arguments:
  -h, --help            show this help message and exit
  --env-name ENV_NAME   Mujoco Gym environment (default: HalfCheetah-v2)
  --policy POLICY       Policy Type: Gaussian | Deterministic (default:
                        Gaussian)
  --eval EVAL           Evaluates a policy a policy every 10 episode (default:
                        True)
  --gamma G             discount factor for reward (default: 0.99)
  --tau G               target smoothing coefficient(τ) (default: 5e-3)
  --lr G                learning rate (default: 3e-4)
  --alpha G             Temperature parameter α determines the relative
                        importance of the entropy term against the reward
                        (default: 0.2)
  --automatic_entropy_tuning G
                        Automaically adjust α (default: False)
  --seed N              random seed (default: 123456)
  --batch_size N        batch size (default: 256)
  --num_steps N         maximum number of steps (default: 1e6)
  --hidden_size N       hidden size (default: 256)
  --updates_per_step N  model updates per simulator step (default: 1)
  --start_steps N       Steps sampling random actions (default: 1e4)
  --target_update_interval N
                        Value target update per no. of updates per step
                        (default: 1)
  --replay_size N       size of replay buffer (default: 1e6)
  --cuda                run on CUDA (default: False)
```

| Environment **(`--env-name`)**| Temperature **(`--alpha`)**|
| ---------------| -------------|
| HalfCheetah-v2| 0.2|
| Hopper-v2| 0.2|
| Walker2d-v2| 0.2|
| Ant-v2| 0.2|
| Humanoid-v2| 0.05|
