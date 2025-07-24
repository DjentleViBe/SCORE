<p align="center">
 <img width="400px" src="./Banner.jpg" align="center" alt="SCORE" />
<h1 align="center">SCORE</h1>
<p align="center" style="font-size: 18px;"><b>S</b>upervised <b>C</b>omp<b>O</b>sition and <b>R</b>iff <b>E</b>ngine</p>

<p align="center">Riff generation using Transformers Architecture</p>

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
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirement-windows.txt
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
1. Download [SCORE-SET](https://github.com/DjentleViBe/SCORE-SET/releases). Extract the contents of ```SCORE-SET_v*.*.*.zip``` into ```gprofiles``` folder.
2. Change the `TRAINING` variable in `config.py` to match the extracted folder downloaded in 1.
3. Run:
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
A pre-trained model can be obtained in the latest [release]().
Please download and extract its contents into the `RESULTS` folder before running evaluation. `BACKUP` variable in `config.py` should match the name of the pre-trained model.