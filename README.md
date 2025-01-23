# Implementation of Deep Hybrid Models for Out-of-Distribution (OOD) detection

Implementation of the Deep Hybrid Models (DHMs) described by Cao and Zhang (2022) [^1]. 
DHMs were supposed to achieve 100% Out-of-Distribution (OOD) detection rates on CIFAR10 and several other datasets; 
however, their code was not publicly released, and we were not able to replicate these results.

OOD detection focuses on detecting classes that were not present in the training set; i.e., on detecting deviations in high-level semantic distributions.
Evaluation is performed by training the model on an in-distribution (ID) dataset, and then evaluating its ability to distinguish ID samples from OOD samples.
In the case of the DHM, it produces a density estimate for any given sample along with a classification; the densities assigned to ID and OOD samples are used to evaluate OOD detection capabilities.
The OOD data is ideally sampled from the same source as the ID data, but represents a mutually exclusive set of classes.
This makes CIFAR-10 and CIFAR-100 excellent datasets for testing OOD capabilities, since they are derived from the same "80 million tiny images" dataset [^2] but contain mutually exclusive classes.
The SVHN dataset is also an important OOD testing set, since although it contains completely different categories conceptually (house numbers), pure normalising flows have often failed to recognise SVHN samples as OOD when trained on CIFAR10 [^3].
This is generally thought to be because the SVHN dataset has very similar distributions of low-level features compared to CIFAR-10, even if the semantic features are completely different.
This makes it an "easy" far-OOD dataset that tests an OOD detector's ability to distinguish high-level features rather than low-level ones. 

## Replicating CIFAR-10 Results

We report the following results compared to the original DHM:

**CIFAR-100 OOD set**

| MODEL            | AuROC           | AuPR-in         | AuTC            |
|------------------|-----------------|-----------------|-----------------|
| DHM (ours)       | 0.897 +/- 0.008 | **0.908 +/- 0.008** | 0.344 +/- 0.006 |
| DHM (original)   | 1.000 +/- 0.00  | 1.000 +/- 0.00  | -               |
| SNGP*            | **0.960 +/- 0.01**  | **0.905 +/- 0.01**  | -               |
| Softmax Baseline | 0.872 +/- 0.009 | 0.80 +/- 0.02   | 0.427 +/- 0.006 |

**SVHN OOD set**

| MODEL            | AuROC          | AuPR-in         | AuTC          |
|------------------|----------------|-----------------|---------------|
| DHM (ours)       | **0.96 +/- 0.01**  | 0.966 +/- 0.007 | 0.27 +/- 0.01 |
| DHM (original)   | 1.000 +/- 0.00 | 1.000 +/- 0.00  | -             |
| SNGP<sup>*</sup> | 0.902 +/- 0.01 | **0.990 +/- 0.01**  | -             |
| Softmax Baseline | **0.95 +/- 0.01**  | 0.93 +/- 0.02   | 0.36 +/- 0.02 |

<sup>*</sup> Note we compare with SNGP [^4] using a WRN-28-10 feature encoder for direct comparison to both DHM implementations.

The results we report for the DHM trained on CIFAR-10 data can be replicated here:

```commandline
python train_dhm_iresflow.py --epochs 200 --batch 256 --N 4 --k 10 --n_blocks 10 --dims 640 --vnorms 222222 --lamb 0.000375 --sn True --n_power_iter 1 --dnn_coeff 3.0 --lr_schedule 60-120-160 --test_every_epoch True --test_model True --seed 0 --lr 1e-4 --flatten True --normalise_features True --distribution_model normflow --actnorm False --save_checkpoints True --dirpath checkpoints/dhm-alt-dists
```


## References

[^1]: Senqi Cao and Zhongfei Zhang. Deep hybrid models for out-of-distribution detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4733–4743, 2022.

[^2]: Antonio Torralba, Rob Fergus, and William T Freeman. 80 million tiny images: A large data set for non-parametric object and scene recognition. IEEE transactions on pattern analysis and machine intelligence, 30(11):1958–1970, 2008.

[^3]: Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur, and Balaji Lakshminarayanan. Do deep generative models know what they don’t know? arXiv preprint arXiv:1810.09136, 2018.

[^4]: Jeremiah Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax Weiss, and Balaji Lakshminarayanan. Simple and principled uncertainty estimation with deterministic deep learning via distance awareness. Advances in neural information processing systems, 33:7498–7512, 2020.

```bibtex
@inproceedings{Cao2022-cc,
  title={Deep hybrid models for out-of-distribution detection},
  author={Cao, Senqi and Zhang, Zhongfei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4733--4743},
  year={2022}
}
```