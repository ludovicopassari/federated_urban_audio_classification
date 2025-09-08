# Setup ambiente virtuale per eseguire il progetto

## Creare un nuovo environment Conda
```
conda create -n mio_env python=3.11

conda activate mio_env

conda install numpy pandas scikit-learn
pip install requests

```

## Esportare envirnoment conda

```
conda env export --name mio_env > environment.yml
```


## Ricreare environment conda su un altra macchina

```
conda env create -f environment.yml
```

