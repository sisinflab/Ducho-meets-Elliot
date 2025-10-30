# Ducho meets Elliot: Large-scale Benchmarks for Multimodal Recommendation

<img src="https://github.com/sisinflab/Ducho-meets-Elliot/blob/master/framework.png?raw=true"  width="1000">

Official repository for the paper _**Ducho meets Elliot**: Large-scale Benchmarks for Multimodal Recommendation_. The codebase was developed and tested on Ubuntu 22.04 LTS; however, the experiments can be executed on other operating systems with the necessary adjustments to environment variables, activation of Python environments, or configuration of additional utilities.

## Installation 

### Clone the repo
This repository integrates the **Ducho** and **Elliot** frameworks. The [Ducho framework](https://github.com/sisinflab/Ducho.git) is hosted in a separate GitHub repository. To properly clone the **Ducho-meets-Elliot** repository, please use the following command:

```sh
git clone --recursive https://github.com/sisinflab/Ducho-meets-Elliot.git
```

### Build the environment
We recommend using Conda to create virtual environments with all the required libraries. For instructions on installing Conda, please refer to the [official documentation](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

To create the environment, run the following command:

```sh
conda env create -f ducho_env.yml
```

After the installation is complete, activate the environment by running the following command:

```sh
conda activate ducho_env
```

### Download and pre-process the datasets

To download the datasets used in our experiments, please go into the **Ducho** submodule.
```sh
cd Ducho
```

Then, run the following command for the desired dataset:

```sh
python3 ./demos/demo_dataset_name/prepare_dataset.py
```


## Running the experiments

After the preprocessing of the selected dataset is complete, ensure you are in the project's root directory and run the following command to execute the experiments. Be careful to replace _dataset\_name_ and _batch\_size_ with the desired ones.

```sh
bash run_experiments.sh dataset_name batch_size
```

## Full version of Table 2: Overview of multimodal features extractors


| Papers | Year | Domain | Visual | Textual | Audio |
|--------|------|--------|--------|---------|-------|
| Han et al. (2017) | 2017 | Fashion | Custom | Custom |  |
| Oramas et al. (2017) | 2017 | Music |  | Custom | Custom |
| Zhang et al. (2017) | 2017 | Social Media | Custom | Custom |  |
| Ying et al. (2018) | 2018 | E-commerce | VGG16 | Word2Vec |  |
| Wang et al. (2018) | 2018 | E-commerce | AlexNet | Custom |  |
| Liu et al. (2019) | 2019 | E-commerce | AlexNet | PV-DM |  |
| Chen et al. (2019) | 2019 | Fashion | VGG19 | Custom |  |
| Wei et al. (2019) | 2019 | Movie, Social Media | ResNet50 | Sentence2Vec | VGGish |
| Cheng et al. (2019) | 2019 | E-commerce, Restaurant | ResNet152 | LDA |  |
| Dong et al. (2019) | 2019 | Fashion | VGG16 | Bag-of-word scheme |  |
| Chen et al. (2019) | 2019 | Fashion | Inception ResNet V2 | TextCNN |  |
| Yu et al. (2019) | 2019 | Fashion | ResNet50 | CNN, GRU |  |
| Cui et al. (2020) | 2020 | E-commerce, Fashion | GoogLeNet | GloVe |  |
| Wei et al. (2020) | 2020 | Movie, Social Media | N/A | N/A | N/A |
| Sun et al. (2020) | 2020 | Movie, Restaurant | ResNet50 | Word2Vec, SIF |  |
| Chen et al. (2020) | 2020 | E-commerce | AlexNet | KimCNN |  |
| Min et al. (2020) | 2020 | Restaurant | Custom |  |  |
| Shen et al. (2020) | 2020 | E-commerce, Music | ResNet50 | Custom |  |
| Yang et al. (2020) | 2020 | Fashion | VGG | Glove |  |
| Tao et al. (2020) | 2020 | Movie, Social Media | ResNet50 | Sentence2Vec | VGGish |
| Yang et al. (2020) | 2020 | Social Media | Inception V3, ResNet50 | BiLSTM |  |
| Sang et al. (2021) | 2021 | Social Media | VGG, C3D | TextCNN |  |
| Liu et al. (2021) | 2021 | E-commerce, Movie | Inception-v4 | BERT | VGGish |
| Zhang et al. (2021) | 2021 | E-commerce | AlexNet | Sentence-BERT |  |
| Vaswani et al. (2021) | 2021 | Music |  | Sentence-BERT | Custom |
| Lei et al. (2021) | 2021 | Restaurant | Custom | Custom | Custom |
| Wang et al. (2021) | 2021 | Restaurant | VGG19 | TextCNN |  |
| Zhan et al. (2022) | 2022 | Fashion | ResNet50 | Custom |  |
| Wu et al. (2022) | 2022 | News | Mask RCNN, ResNet50, ViLBERT | ViLBERT |  |
| Yi et al. (2022) | 2022 | E-commerce | ResNet50 | Sentence2Vec |  |
| Yi et al. (2022) | 2022 | Movie, Social Media | N/A | N/A | N/A |
| Liu et al. (2022) | 2022 | E-commerce | CLIP | N/A |  |
| Mu et al. (2022) | 2022 | E-commerce | AlexNet | Sentence-BERT |  |
| Chen et al. (2022) | 2022 | Movie, Social Media | ResNet50 | Sentence2Vec | VGGish |
| Zhou & Shen (2022) | 2022 | E-commerce | AlexNet | Sentence-BERT |  |
| Wang et al. (2023) | 2023 | Movie, Social Media | ResNet50 | Sentence2Vec | VGGish |
| Wei et al. (2023) | 2023 | E-commerce, Restaurant, Social Media | Custom | Sentence-BERT | N/A |
| Zhou et al. (2023) | 2023 | E-commerce | AlexNet | Sentence-BERT |  |
| Zhou et al. (2023) | 2023 | E-commerce | AlexNet | Sentence-BERT |  |
| Zhou & Shen (2023) | 2023 | E-commerce | AlexNet | Sentence-BERT |  |
| Yu et al. (2023) | 2023 | E-commerce | AlexNet | Sentence-BERT |  |
| Tao et al. (2023) | 2023 | Movie, Social Media | ResNet50 | Sentence2Vec | VGGish |
| Guo et al. (2024) | 2024 | E-commerce | AlexNet | Sentence-BERT |  |
| Su et al. (2024) | 2024 | E-commerce | AlexNet | Sentence-BERT |  |
| Jiang et al. (2024) | 2024 | Social Media, E-commerce | Custom | Sentence-BERT | N/A |
| Malitesta et al. (2024) | 2024 | E-commerce | ResNet50 | Sentence-BERT |  |
| Xu et al. (2025) | 2025 | E-commerce | AlexNet | Sentence-BERT |  |
| Ong & Khong (2025) | 2025 | E-commerce | VGG16 | Sentence-BERT |  |
| Xu et al. (2025) | 2025 | E-commerce | AlexNet | Sentence-BERT |  |
| Yu et al. (2025) | 2025 | E-commerce | AlexNet | Sentence-BERT |  |


# The Team

Currently, this repository is mantained by:

- Matteo Attimonelli* (matteo.attimonelli@poliba.it)
- Danilo Danese* (danilo.danese@poliba.it)
- Angela Di Fazio* (angela.difazio@poliba.it)
- Daniele Malitesta** (daniele.malitesta@centralesupelec.fr)
- Claudio Pomo* (claudio.pomo@poliba.it)
- Tommaso Di Noia* (tommaso.dinoia@poliba.it)

\* _Politecnico Di Bari, Bari, Italy_

\*\* _Université Paris-Saclay, CentraleSupélec, Inria, France_