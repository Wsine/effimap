# EffiMAP

Project Code: Tepig

For more details, please refer to the following publication.

## Publication

Z. Wei, H. Wang, I. Ashraf and W. K. Chan, "Predictive Mutation Analysis of Test Case Prioritization for Deep Neural Networks," 2022 IEEE 22nd International Conference on Software Quality, Reliability and Security (QRS), Guangzhou, China, 2022, pp. 682-693, doi: 10.1109/QRS57517.2022.00074. keywords: {Deep learning;Analytical models;Computational modeling;Neural networks;Software quality;Predictive models;Computational efficiency;Test case prioritization;mutation analysis;testing},

```
@INPROCEEDINGS{10062402,
  author={Wei, Zhengyuan and Wang, Haipeng and Ashraf, Imran and Chan, W.K.},
  booktitle={2022 IEEE 22nd International Conference on Software Quality, Reliability and Security (QRS)}, 
  title={Predictive Mutation Analysis of Test Case Prioritization for Deep Neural Networks}, 
  year={2022},
  volume={},
  number={},
  pages={682-693},
  keywords={Deep learning;Analytical models;Computational modeling;Neural networks;Software quality;Predictive models;Computational efficiency;Test case prioritization;mutation analysis;testing},
  doi={10.1109/QRS57517.2022.00074}
}
```

## Preparation

The dataset and the model will be fetched automatically. No need to handle by the user.

## Installation

The project is maintained with [Pipenv](https://pipenv.pypa.io/en/latest/). Please refer to the link for installing Pipenv.

The dependencies are very convenient to install by one command. The versions are same as proposed here.

```bash
pipenv sync
```

## How to run

The executions are well organized with the help of arguments. you can run `--help` command to check it.

for example,

```
~/workspace/effimap{main} > pipenv run python src/extract.py -h
usage: extract.py [-h] [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--device {cpu,cuda}] [--gpu {0,1,2,3}] [--seed SEED] [-b BATCH_SIZE] [-m {resnet32,mlp,svhn,stl10,resnet18,resnet20,msgdn}]
                  [-d {cifar10,cifar100,mnist,svhn,stl10,tinyimagenet,nuswide}] [-e EPOCHS] [--lr LR] [--fuzz_energy FUZZ_ENERGY] [--num_model_mutants NUM_MODEL_MUTANTS]
                  [--num_input_mutants NUM_INPUT_MUTANTS] [--task {classify,regress,multilabels}] [--prima_split {val,test}] --strategy {prima,furret}

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR
  --output_dir OUTPUT_DIR
  --device {cpu,cuda}
  --gpu {0,1,2,3}
  --seed SEED
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -m {resnet32,mlp,svhn,stl10,resnet18,resnet20,msgdn}, --model {resnet32,mlp,svhn,stl10,resnet18,resnet20,msgdn}
  -d {cifar10,cifar100,mnist,svhn,stl10,tinyimagenet,nuswide}, --dataset {cifar10,cifar100,mnist,svhn,stl10,tinyimagenet,nuswide}
  -e EPOCHS, --epochs EPOCHS
  --lr LR
  --fuzz_energy FUZZ_ENERGY
  --num_model_mutants NUM_MODEL_MUTANTS
  --num_input_mutants NUM_INPUT_MUTANTS
  --task {classify,regress,multilabels}
  --prima_split {val,test}
  --strategy {prima,furret}
```
