<div align="center">
  <br />
  <p>
    <a href="https://github.com/Chokyotager/BIND"><img src="/art/BIND.png" alt="banner" /></a>
  </p>
  <br />
  <p>
  </p>
</div>

## About
**Binding INteraction Determination** (BIND) is a graph neural network that uses embeddings from ESM-2 to perform virtual screening witout 3D structural information. By cross-attending graph nodes representing the ligand molecule with the embeddings from ESM-2, BIND is able to achieve high screening power in datasets evaluated in the paper, seemingly surpassing some of the classical structure-based methods.

Summarily, a high screening power describes a model that can strongly separate out true binders from non-binders (decoys). This is especially useful in computer-aided drug design as it enriches lead molecules from molecules that may not have any effect, and when applied in a high-throughput fashion, can filter out vast numbers of unique molecules.

BIND uses graphs to represent molecules as there is no need for SMILES canonicalisation, and graphs model atoms and bonds intuitively. Since ESM-2 is used to represent the protein, BIND is fully sequence-based, and does not require one to determine the pocket or structure.

This repository contains the code we used in training of BIND, and an inference script where you can easily run your own proteins and ligands in standard FASTA and SMILES format. You can also train your own BIND model with instructions provided in this README if you so wish.

## Paper

![Figure abstract](https://github.com/Chokyotager/BIND/blob/main/art/abstract.png?raw=true)

Please read the paper for more details:

Preprint (bioRxiv): https://doi.org/10.1101/2024.04.16.589765

## Installation
```sh
git clone https://github.com/Chokyotager/BIND.git
cd BIND
```

```sh
conda env create -f environment.yml
conda activate bind
```

BIND has been tested on two separate Ubuntu 22.04.3 LTS (GNU/Linux 5.15.0-58-generic x86_64) systems. It theoretically should work for any environment so long as all package requirements are fulfilled.

Installation should take under ten minutes in most cases.

Take note that when you first run inference, BIND will download the ESM-2 weights from the HuggingFace Hub. The transformers library automatically caches ESM-2 afterwards so subsequent runs doesn't need this to be downloaded again.

## Usage

By default, the saved model `BIND_checkpoint_12042024.pth` is used. This is the same model which was used throughout the manuscript and ultimately used generate the figures. You should be able to reproduce the results accordingly.

If you want to train your own model, look at the "training your own model" section below.

### Performing forward/reverse virtual screening with BIND

In order to run inference with BIND, you can run the following script once everything is installed:

```sh
python3 bind.py --proteins <protein>.fasta --ligands <ligands>.smi --output <output>.tsv
```

This will perform inference for all pairs of proteins and ligands specified, and output a single TSV file with five columns: input protein name, input SMILES, predicted pKi, predicted pIC50, predicted Kd, predicted EC50, logits and non-binder probability. Take note that the non-binder probability is just sigmoid(logits) and is the output from the classifier. You can see an example output file in `examples/example_output.tsv`.

In our paper, we found that the strongest screening power comes from the non-binder probability. **The lower the logits / non-binder probability, the better**. Whereas for the drug-target affinity predictions, the higher the better.

You can try running BIND on the example protein and ligands we provide in the examples folder, by running this command:

```sh
python3 bind.py --proteins examples/example_protein.fasta --ligands examples/example_ligands.smi --output examples/example_output.tsv
```

### Advanced inference options

There are some advanced options in running inference. You can also specify the device BIND and ESM-2 are run on, by using `--bind_device` and `--esm_device` respectively. These default to the CPU but if you have a PyTorch-recognisable device (i.e. a GPU), you can specify something like `cuda:0`, etc. Please see the PyTorch documentation [here](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) for what devices are accepted. Usually you'd want to run ESM-2 on the CPU and BIND on your GPU unless you have a super beefy GPU, since ESM-2 is vRAM intensive.

Take note also that BIND defaults to truncating proteins that are longer than 4,096 amino acids. You can change this limit to whatever you want by specifying it in the `--truncate` option.

### API practical demonstration!

There is an IPython Notebook that you can open using Jupyter and/or other notebooks (not tested) named `BIND_demo.ipynb`. It specifies how to use the Python interface to call BIND directly. Usually this is for more advanced users who want to integrate BIND directly into their Python pipeline. For standard screening workflows, calling the inference script will probably suffice in most cases.

### Training your own model

If you want to train your own model, you need to download the BindingDB dataset or make your own. Because this dataset is large and not made by us, we provide a downsampled version of it in `examples/bindingdb_sample.json` as reference. You can run `train_graph.py` to initiate training, and it'll default to running on this example file (change the location in `dataset.py` if you want to use another dataset). Note that the actual dataset we used had 2,749,266 unique protein-ligand entries, of which 2,469,626 were used for training the provided checkpoint. The heavily downsampled example file only has 1,000 and one unique protein, so you probably aren't going to get the best model from it alone.

### Evaluation and miscellaneous scripts from the paper

The scripts used to calculate the success rates, etc., are in paper_eval_scripts. These scripts are just here for reference and are effectively vestigial in the code.

## Limitations

Please read the limitations discussed in the paper to understand the constraints of this work! 

## Contribution

Users are welcome to contribute to the development of the model through pull-requests. You are also welcome to raise feature requests in the "issues" section of the GitHub. Please report any issues / bugs there too.

## Maintenance

The current project is maintained by Hilbert Lam. Correspondence can be found in the manuscript if you have any questions. You can also contact Hilbert via email or (informally) on Discord, using the handle chocoparrot.

## Reproduction of results in the paper

We provide the evaluation dataset FASTAs in our manuscript as supplementary material instead of here as the datasets are rather large. The SMILES are pretty big to be put as supplementary, so you can convert them yourself using OpenBabel from the original datasets.

## Cross-attention graph module

The cross-attention graph module is described in `cross_attention_graph.py`. You have to add your own GATConv2 for the self-attention to make the entire cross-attention graph block as described in the manuscript.

## License

License details can be found in the LICENSE file.

## Citation
```
@article{lam_protein_2024,
	title = {Protein language models are performant in structure-free virtual screening},
	url = {https://www.biorxiv.org/content/10.1101/2024.04.16.589765v1},
	doi = {10.1101/2024.04.16.589765},
	language = {en},
	urldate = {2024-04-21},
	journal = {bioRxiv},
	author = {Lam, Hilbert Yuen In and Jia Sheng, Guan and Xing Er, Ong and Pincket, Robbe and Yuguang, Mu},
	month = apr,
	year = {2024},
}
```
