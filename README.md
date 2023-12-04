# Joint Computing, Pushing, and Caching Optimization for Mobile Edge Computing Networks via Soft Actor-Critic Learning
A Deep-Reinforcement Learning Approach for activity optimization in mobile edge computing (MEC) network.

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

## Abstract
Mobile edge computing (MEC) networks bring computing and storage capabilities closer to edge devices, which reduces latency and improves network performance. However, to further reduce transmission and computation costs while satisfying user-perceived quality of experience, a joint optimization in computing, pushing, and caching is needed. In this paper, we formulate the joint-design problem in MEC networks as an infinite-horizon discounted-cost Markov decision process and solve it using a deep reinforcement learning (DRL)-based framework that enables the dynamic orchestration of computing, pushing, and caching. Through the deep networks embedded in the DRL structure, our framework can implicitly predict user future requests and push or cache the appropriate content to effectively enhance system performance. One issue we encountered when considering three functions collectively is the curse of dimensionality for the action space. To address it, we relaxed the discrete action space into a continuous space and then adopted soft actor-critic learning to solve the optimization problem, followed by utilizing a vector quantization method to obtain the desired discrete action. Additionally, an action correction method was proposed to compress the action space further and accelerate the convergence. Our simulations under the setting of a general single-user, single-server MEC network with dynamic transmission link quality demonstrate that the proposed framework effectively decreases transmission bandwidth and computing cost by proactively pushing data on future demand to users and jointly optimizing the three functions. We also conduct extensive parameter tuning analysis, which shows that our approach outperforms the baselines under various parameter settings.

SAC_V](https://github.com/pranz24/pytorch-soft-actor-critic/tree/SAC_V).

## Requirements
*   Python 3.6
*   Preferred system: Linux
*   Pytorch-1.5.1
*   Other packages (refer to [requirement](requirements.txt))


## Default Arguments and Usage
### System Configuration

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

python main.py --automatic_entropy_tuning True --target_update_interval 1000 --lr 1e-4 --exp-case case4 --cuda
#python main.py --alpha 0.2 --target_update_interval 1000 --lr 1e-4 --exp-case case4 --cuda
tensorboard --logdir=runs --host localhost --port 8088

## License

This codebase is released under MIT license (see [LICENSE](LICENSE)).

## Acknowledgement
This project is not possible without multiple great opensourced codebases. We list some notable examples below.  

* [pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic)

## Contact
Contact *Xiangyu Gao* ([xygao@uw.edu](mailto:xygao@uw.edu)). Questions and suggestions are welcome.




