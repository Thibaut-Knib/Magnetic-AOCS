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
