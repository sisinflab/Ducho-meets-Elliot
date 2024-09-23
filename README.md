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

# The Team

Currently, this repository is mantained by:

- Matteo Attimonelli* (matteo.attimonelli@poliba.it)
- Danilo Danese* (danilo.danese@poliba.it)
- Angela Di Fazio* (angela.difazio@poliba.it)
- Daniele Malitesta** (daniele.malitesta@poliba.it)
- Claudio Pomo* (claudio.pomo@poliba.it)
- Tommaso Di Noia* (tommaso.dinoia@poliba.it)

\* _Politecnico Di Bari, Bari, Italy_

\*\* _Université Paris-Saclay, CentraleSupélec, Inria, France_