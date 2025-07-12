![Banner.jpg](Banner.jpg)
# Supervised Composition and Riff Engine (SC0RE)
Riff generation using various Transformers Architecture

## Dependencies
- python > 3.11
- pip
- pytorch
- sklearn
- matplotlib
- PyGuitarPro

### Installation
Use virtual environment if possible <br>
```pip install -r requirements-{OS_TYPE}.txt```

## Running
### Training
1. Place the `.gp5` files intended for training inside the folder:
`gprofiles/<MUSIC_STYLE>/`
where `<MUSIC_STYLE>` corresponds to the value of the `TRAINING` variable in `config.py`.
2. Modify the relevant variables in `config.py` as needed for your experiment.
3. Set `MODE = 0` in `config.py` to enable training mode. 
4. Run the training script:
```sh
python main.py
```

### Evaluation
After training is complete:
1. Change `MODE = 1` in `config.py` for evaluation.
2. Adjust the `START_ID` variable in `config.py` to the desired starting token ID for evaluation.
3. Run the evaluation script:
```sh
python main.py
```

`.gp5` files are exported inside `RESULTS`/`<TEST>` folder where `<TEST>` is the `BACKUP` variable entered in `config.py`.

### (Pre-)trained model
A pre-trained model is available in the latest release section of the repository.
Please download and extract its contents into the `RESULTS` folder before running evaluation or inference. `BACKUP` variable in `config.py` should match the name of the downloaded model.