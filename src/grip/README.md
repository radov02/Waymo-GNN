# Waymo→GRIP conversion (HDF5 graphs, configurable agent cap, 2 Hz windows)

`git submodule add https://github.com/radov02/Waymo-GNN/ vendor/erik-waymo`
`git submodule add https://github.com/lukaprsina/GRIP vendor/grip`

`pip install -e vendor/grip`

## Data preparation

```sh
python src/grip/train.py prepare \
 --train_hdf5 vendor/erik-waymo/data/graphs/training/training_seqlen90.hdf5 \
 --val_hdf5 vendor/erik-waymo/data/graphs/validation/validation_seqlen90.hdf5 \
 --test_hdf5 vendor/erik-waymo/data/graphs/testing/testing_seqlen90.hdf5 \
 --output_dir data/grip --max_agents 48 --neighbor_distance 10.0
```

## GPU training (2 epochs, batch size 2)

```sh
python src/grip/train.py train \
 --train_data_path data/grip/train_data.pkl \
 --val_data_path data/grip/val_data.pkl \
 --test_data_path data/grip/test_data.pkl \
 --work_dir outputs/grip_demo \
 --epochs 2 --batch_size 2 --val_batch_size 2 \
 --num_workers 0 --device cuda --lr 1e-3
```

```sh
python src/grip/train.py train --train_data_path data/grip/train_data.pkl --val_data_path data/grip/val_data.pkl --work_dir outputs/grip_demo --epochs 1 --batch_size 8 --val_batch_size 8 --num_workers 0 --device cpu --lr 1e-3 --subset_fraction 0.01
```
