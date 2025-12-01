# The Geometry of Neural Networks: a Riemannian Foliation Perspective on Robustness (GoNN:FR).

This package was developed as part of [Eliot Tron](https://halva.ynh.fr/eliot.tron)'s PhD thesis, under the supervision of Nicolas Couëllan, Rita Fioresi and Stéphane Puechmorel.
It implements the basic blocks of the framework described in the manuscript available online [here](https://enac.hal.science/tel-05391126).

## Installation

Install GoNN_FR with pip:

```bash
python -m pip install git+https://codeberg.org/eliot-tron/GoNN_FR
```
or
```bash
python -m pip install git+https://github.com/eliot-tron/GoNN_FR
```
    
## Usage/Examples

More examples can be found in `./examples/`.

```python
from GoNN_FR import geometry, experiment
```


## Documentation

(In construction)
[Documentation](./docs)


## Authors

- [@eliot-tron](https://codeberg.org/eliot-tron) ([website](https://halva.ynh.fr/eliot.tron)).

## Citation
If you use (part of) this work, please cite the manuscript of my PhD thesis with the follow bibtex:

```
@phdthesis{tron:tel-05391126,
  TITLE = {{The geometry of neural networks : a Riemannian foliation perspective on robustness.}},
  AUTHOR = {Tron, Eliot},
  URL = {https://enac.hal.science/tel-05391126},
  NUMBER = {2025ENAC0004},
  SCHOOL = {{ENAC Toulouse}},
  YEAR = {2025},
  MONTH = Oct,
  KEYWORDS = {Neural Network ; Robustness ; Riemannian Foliation ; Fisher Information ; Robustesse ; Information de Fisher ; Feuilletage riemannien ; R{\'e}seaux de Neurones Artificiels},
  TYPE = {Theses},
  PDF = {https://enac.hal.science/tel-05391126v1/file/152652_TRON_2025.pdf},
  HAL_ID = {tel-05391126},
  HAL_VERSION = {v1},
}
```

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)

