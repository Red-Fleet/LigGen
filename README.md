# LigGen: Ligand Generation Toolkit

## Introduction
LigGen is a computational tool designed for de novo ligand generation using fragment-based approaches. It can utilize pre-existing molecular fragments or generate new ones using a trained RNN-based fragment generator. Additionally, LigGen allows for fragment generator training, which improves its ability to generate meaningful chemical structures. LigGen also includes a Flask server and a React.js frontend, making it accessible via a web interface.

The server is hosted at: [Neurocare LigGen](https://neurocare-liggen.iiitd.edu.in/) and is free to use.

## Installation
To install LigGen, follow these steps:

1. **Create a Conda environment:**
   ```bash
   conda env create -f environment.yml
   ```
   This will install all the required dependencies.

2. **Manually install PyTorch and TorchText:**
   ```bash
   torch version = 2.3.1
   torchtext version = 0.18.0
   ```

## LigGen Components
LigGen consists of three main functionalities:

### 1. LigGen Ligand Generator
Generates ligands using molecular fragments. These fragments can be:
- Provided manually as an input file.
- Generated dynamically using an RNN-based fragment generator.

### 2. LigGen Train Fragment Generator
Trains an RNN-based model to generate molecular fragments. The trained model parameters are used in LigGen’s fragment generator.

### 3. LigGen Fragment Generator
Uses a pre-trained RNN model to generate new molecular fragments.

## Usage
### Ligand Generator
Run the following command to generate ligands:
```bash
generate_ligands [-h] [-fp FRAGMENT_PATH] -tp TARGET_PATH -o OUTPUT_DIR -c COUNT -gc GRID_CENTER -gs GRID_SIZE [-th THREADS] [-rnn] [-p RNN_PARAMS] [-d RNN_DEVICE] [-ml RNN_MAX_LEN] [-rc RNN_COUNT] [-al ALPHA] [-cp CHAIN_EXTEND_PROBABILITY] [-w WEIGHT] [-mi MAX_ITER] [-t TEMP] [-s SCORE] [-vw VINA_WEIGHT] [-do] [-de]
```
#### Parameters Explained
- **-fp, --fragment_path**: Path to the file containing fragments in SMILES format.
- **-tp, --target_path**: Path to the target protein in PDBQT format.
- **-o, --output_dir**: Directory to store the generated ligands.
- **-c, --count**: Number of ligands to generate.
- **-gc, --grid_center**: `[x,y,z]` coordinates for the center of the docking grid (without space).
- **-gs, --grid_size**: `[x,y,z]` dimensions for the docking grid (without space).
- **-th, --threads**: Number of threads to use for computation.
- **-rnn, --rnn**: Flag to enable RNN-based fragment generation.
- **-p, --rnn_params**: Path to the trained RNN model parameters (default: `model.pt`).
- **-d, --rnn_device**: Compute device (`cpu` or `gpu`, default: `cpu`).
- **-ml, --rnn_max_len**: Maximum length of generated fragments (default: `40`).
- **-rc, --rnn_count**: Number of fragments generated per ligand (default: `256`).
- **-al, --alpha**: Cooling schedule factor (default: `0.3`).
- **-cp, --chain_extend_probability**: Probability of adding a new fragment to an existing ligand chain (default: `0.8`).
- **-w, --weight**: Target molecular weight of generated ligands (default: `500`).
- **-mi, --max_iter**: Maximum number of failed attempts before rejecting a fragment (default: `50`).
- **-t, --temp**: Initial temperature for the Monte Carlo search (default: `50`).
- **-s, --score**: Initial score value (default: `0`).
- **-vw, --vina_weight**: Weight factor for Vina docking score (default: `0.5`), decides amount of weightage given to vina score and systhesiazibility score.
- **-do, --dock**: If set, dock the generated ligands with the protein target.
- **-de, --details**: If set, save detailed ligand generation steps.


### Train Fragment Generator
To train an RNN-based fragment generator:
```bash
train_fragment_generator [-h] -i INPUT_SMILES [-ip IN_MODEL_PARAMS] [-op OUT_MODEL_PARAMS] [-b BATCH_SIZE] [-e EPOCH] [-l MAX_LEN] [-d DEVICE]
```
#### Parameters Explained
- **-i, --input_smiles**: Path to file containing molecular fragments in SMILES format.
- **-ip, --in_model_params**: Path to initialize model parameters (default: `random`).
- **-op, --out_model_params**: Path to save trained model parameters (default: `model.pt`).
- **-b, --batch_size**: Number of training samples per batch (default: `512`).
- **-e, --epoch**: Number of training epochs (default: `1`).
- **-l, --max_len**: Maximum token length for training samples (default: `100`).
- **-d, --device**: Compute device (`cpu` or `gpu`, default: `cpu`).


### Fragment Generator
To generate fragments using a trained RNN model:
```bash
generate_fragments [-h] [-p MODEL_PARAMS] -o OUT_PATH [-b BATCH_SIZE] [-i ITERATION] [-l MAX_LEN] [-d DEVICE]
```
#### Parameters Explained
- **-p, --model_params**: Path to trained RNN model parameters (default: `model.pt`).
- **-o, --out_path**: Path to save generated fragments in SMILES format.
- **-b, --batch_size**: Number of fragments generated per batch (default: `64`).
- **-i, --iteration**: Number of iterations to generate fragments (default: `10`).
- **-l, --max_len**: Maximum token length of generated fragments (default: `100`).
- **-d, --device**: Compute device (`cpu` or `gpu`, default: `cpu`).


## Example Usage
### 1. Generate Ligands with Predefined Fragments
```bash
python generate_ligands.py -fp frags/ligbuilder_frags.smiles -tp in_files/2g94/2g94.pdbqt -o output -c 6 -gc [-4,-4,30] -gs [20,20,20] -th 3 -w 500
```

### 2. Generate Ligands Using RNN Fragments
```bash
python generate_ligands.py -tp in_files/2g94/2g94.pdbqt -o output -c 3 -gc [-4,-4,30] -gs [20,20,20] -th 3 -w 500 -rnn -p ligbuilder_model.pt -d cpu
```

### 3. Train Fragment Generator
```bash
python train_fragment_generator.py -i {input_smiles_path} -ip {old_parameters_used_for_finetuning} -op {learned_output_params}
```

### 4. Generate Fragments Using a Pre-Trained Model
```bash
python generate_fragments.py -p {model_parameters_path} -o {output_path} -i {total_epoches}
```

## Conclusion
LigGen provides a robust framework for ligand generation using fragment-based and deep-learning approaches. By allowing users to train their fragment generators and generate ligands efficiently, LigGen is a powerful tool for virtual screening and drug discovery research.

For any issues, please refer to the documentation.

