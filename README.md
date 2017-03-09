# Multi-task Q transfer

Fork of gym repo, with mountain_car.py modified in `gym/gym/envs/classic_control`


## Installation
For simplicity, gym repo is cloned in the repo. To install project, 

```
git clone https://github.com/GuillaumeGenthial/Q-transfer.git
cd Q-transfer/gym
python setup.py develop
cd ..
```

## Global approximation methods

Files are under the `global_approximation` repository.
To run experience with a specified config file

```
python main.py path_to_config_file
```

## Deep Q networks

Files are under the `deep` repository.
To run experience with a specified config file,


```
python main_deep.py path_to_config_file
```

