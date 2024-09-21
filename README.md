# MultiModal Benchmarking Ducho


Official repository for the paper _**Ducho meets Elliot**_.


# Clone the repo
In order to correctly clone the repository and the Ducho submodule, run:
```sh
git clone --recursive git@github.com:sisinflab/MultiModalBenchmarking.git
```


# Running the experiments
```sh
cd Ducho
python3 ./demos/demo_dataset_name/prepare_dataset.py
```
Once the preprocessing of the desired dataset finishes, to execute the experiments please run, ensuring to be in the root directory of the project:
```sh
bash run_experiments.sh dataset_name batch_size
```

# The Team

Currently, this repository is mantained by:

- Matteo Attimonelli (matteo.attimonelli@poliba.it)
- Danilo Danese (danilo.danese@poliba.it)
- Angela Di Fazio (angela.difazio@poliba.it)
- Daniele Malitesta (daniele.malitesta@poliba.it)
- Claudio Pomo (claudio.pomo@poliba.it)
- Tommaso Di Noia (tommaso.dinoia@poliba.it)