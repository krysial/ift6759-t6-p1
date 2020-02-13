**IFT6759 - Advanced projects in machine learning**
---

IFT6759 is a course about implementing deep learning theory in applied projects.

# Project 1

Setup
---

The typical setup involves installing VSCode with a few extensions:

- SSH remote
- Python
- Codestyle lint

An example of the remote `.vscode/launch.json` file can be seen here:
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [

        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env",
            "args": ["data/admin_cfg.json", "--model=lrcn", "--real", "--batch-size=2"]
        }
    ]
}
```

An example of the remote `.vscode/settings.json` file can be seen here:
```
{
    "python.pythonPath": "~/ift6759-env/bin/python",
    "python.formatting.autopep8Path": "~/ift6759-env/bin/autopep8",
    "python.linting.pycodestyleEnabled": true,
    "python.linting.enabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.nosetestsEnabled": false,
    "python.testing.pytestEnabled": true,
    "python.dataScience.jupyterServerURI": "local"
}
```


One must make also sure that their ssh keys are synced with the Helios login Node.
An example of a ssh configuration can be seen here:

```bash
Host helios
  HostName helios3.calculquebec.ca
  IdentityFile ~/.ssh/umontreal
  User guest133
  ServerAliveInterval 120
```

Make sure you have setup your local `venv` properly, we recommend setting it up on your local disk on helios doing the following.

1.  `ssh helios`
1.  `module load python/3`
1.  `python -m venv ~/ift6759-env`
1.  `source ~/ift6759-env/bin/activate`
1.  `pip install -r requirements.txt`

This setup allows you to develop quickly and run/train the model on the CPU. However, given that the compute nodes don't have access to the internet. You must setup a python venv that is shared between the login node and the compute node. We opted to do that in the following folder `/project/cq-training-1/project1/teams/team06/ift6759-env/bin/activate`. Hence, if your intent is to use the compute node, you must replace step 4 above with the following line of code.

1. `source /project/cq-training-1/project1/teams/team06/ift6759-env/bin/activate`


Training the model
---

**Compute Node**

`sbatch train_on_slurm.sh`

To check the logs you can either tail with:

`train -f /results/slurm-<job-id>_<weird-hash>.out`

or you can run and then tunnel port `6006` to your localhost via:

1. `tensorboard --logdir=results/`

In another terminal run:

2. `ssh -L 6006:localhost:6006 helios`

Now you can visit `localhost:6006` on your local machine.

**Local machine**

If you sourced your local python properly you can train the model on the login node via:

`python train.py data/admin_cfg.json --real --batch-size=1 --model=lrcn`

You can check the help command:

`python train.py -h`

That outputs at time of writing the following:

```
usage: train.py [-h] [--station STATION] [--real] [--crop-size CROP_SIZE]
                [--epoch EPOCH] [--dataset-size DATASET_SIZE]
                [--seq-len SEQ_LEN] [--batch-size BATCH_SIZE] [--model MODEL]
                [--channels [CHANNELS [CHANNELS ...]]] [-u USER_CFG_PATH]
                admin_config_path [input_shape [input_shape ...]]

positional arguments:
  admin_config_path     path to the JSON config file used to store test
                        set/evaluation parameters
  input_shape           input shape of first model layer

optional arguments:
  -h, --help            show this help message and exit
  --station STATION     station to train on
  --real                train on synthetic mnist data
  --crop-size CROP_SIZE
                        size of the crop frame
  --epoch EPOCH         epoch count
  --dataset-size DATASET_SIZE
                        dataset size
  --seq-len SEQ_LEN     sequence length of frames in video
  --batch-size BATCH_SIZE
                        batch size of data
  --model MODEL         model to be train/tested
  --channels [CHANNELS [CHANNELS ...]]
                        channels to keep
  -u USER_CFG_PATH, --user_cfg_path USER_CFG_PATH
                        path to the JSON config file used to store user
                        model/dataloader parameters
```

