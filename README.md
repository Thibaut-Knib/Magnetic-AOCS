# SCAO

**Ce repo a été migré sur [github](https://github.com/astronautix/Magnetic-AOCS)**

## Installation
python 3.7
```bash
pip3 install -r requirements.txt
```

## Structure
- `src` : Algorithmes de base
  - `divers` : fichier de calculs rapides de dimensionnement des bobines
  - `environnement` : TODO, rennomer en `env`?
  - `hardware` : TODO, rennomer en `hardw`?
  - `scao` : API et algorithmes de stabilisation du satellite
- `tst` : Tests et validation des algorithmes (fourmille d'exemples!)
  - `sim` : Simultation numerique de l'évolution d'un satellite contrôlé avec les algorithmes dans `scao`
  - `lab` : Algorithmes de tests en laboratoire
    - `bbb` : Algorithmes qui tournent sur la BeagleBone Blue (penser à installer les dépendances dans `tst/lab/bbb/requirements.txt`)
    - `client` : Visualisation et plotting de l'état de la BeagleBone
    - `helmoltz` : Contrôle des bobines de helomotz du LPP


## Notes et commentaires

### Arboressance

```
/ Magnetic-AOCS
      - Doc
      - Magnetic-AOCS
      - Tests
      - Examples /
            - Notebooks
            - scripts /
                 - Bdot.py
                     sys.path.append("../../Magnetic-AOCS")
                  
```
#### Solution 0
```python
import sys
import os
#os.chdir(r"C:\Users\thiba\Documents\Polytechnique\P3A\Magnetic-AOCS\tst\sim")
sys.path.append(os.path.join(*['..'] * 2))
print(os.getcwd())
sys.path.append("../../src/")
sys.path.append("src/")
```

#### Solution 1

`export PYTHONPATH=$PATHPYTHON:<chemain vers Magnetic-AOCS/Magnetic-AOCS>`


#### Solution 2

installer `Magnetic-AOCS/Magnetic-AOCS` avec pip

`pip install -e .`
`pip install -e ./Magnetic-AOCS`








                 

      