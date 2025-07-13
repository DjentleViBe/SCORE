![Banner.jpg](Banner.jpg)
<p align="center" style="font-size: 30px;"><b>S</b>upervised <b>C</b>omp<b>O</b>sition and <b>R</b>iff <b>E</b>ngine</p>

##
Riff generation using Transformers Architecture

## Dependencies
- python > 3.11
- pip
- pytorch
- sklearn
- matplotlib
- PyGuitarPro

### Installation
Use virtual environment if possible. To install dependencies:

On Windows:
```
pip install -r requirements-windows.txt
```
On macOS:
```
pip install -r requirements-macos.txt
```
On Linux:
```
pip install -r requirements-linux.txt
```
## Running
### Training
1. Place the `.gp5` files intended for training inside the folder:
`gprofiles/<MUSIC_STYLE>/`
where `<MUSIC_STYLE>` corresponds to the value of the `TRAINING` variable in `config.py`.
2. Modify the relevant variables in `config.py` as needed for your experiment.
3. Run the training script:
```sh
python main.py --mode train
```

### Evaluation
After training is complete:
1. Run the evaluation script:
```sh
python main.py --mode eval --start_id <id>  
```
where `<id>` is the first token of the musical measure. 

`.gp5` files are exported inside `RESULTS`/`<TEST>` folder where `<TEST>` is the `BACKUP` variable entered in `config.py`.

### (Pre-)trained model
A pre-trained model is available in the latest release section of the repository.
Please download and extract its contents into the `RESULTS` folder before running evaluation or inference. `BACKUP` variable in `config.py` should match the name of the downloaded model.