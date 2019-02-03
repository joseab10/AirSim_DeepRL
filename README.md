# AirSim + Deep RL

## Environments
To keep the repository as lightweight as possible, the AirSim environment is not included.
For it to work, you can download the precompiled binaries for your OS from:

https://github.com/Microsoft/AirSim/releases

Just download the desired environment and expand it into the `/env` directory at the root of this repository.
The code on this project has been tested with the LandscapeMountains, Blocks, CarMaze and Neighborhood environments.

You can also compile and create your own environments following the instructions from:

https://microsoft.github.io/AirSim/docs/build_linux/

## Run:

`train_drone.py` is the main program to set-up, train and test an agent in one of the AirSim Environments. It is designed to start the environment by itself, bu depending on the system specs, it may timeout. The environment can be also executed manually beforehand, and the API will connect automatically.


The main script supports a lot of arguments to modify it's behaviour, all of them optional. The most important ones are:

`python3.5 train_drone.py [--train] [--test] [--render] [--verbose] [--model <net_name>] [--env <environment>] [--targets <target_set>]`

where:
- `--train` executes the training operation of the agent.

- `--test` executes the test procedure on a trained model.

- `--render` toggles graphical rendering.

- `--verbose` toggles printing each idividual action to stdout.

- `--model <net_name>` defines the name (without file extension) of the network configuration.

- `--env <environment>` defines the environment to train the agent on:
    * **blocks** : Blocks
    * **maze** : SimpleMaze (Windows Only)
    * **neighborhood** : AirSimNH
    * **mountains** : LanscapeMountains (Windows Only)

- `--targets <target_set>`:
    * **mn_lines** : Transmission Lines in the Mountains Environment (by name)
    * **nh_lines** : Distribution Lines in the Neighborhood Environment (by position)
    * **nh_pools** : Backyard pools in the Neighborhood Environment (by position)
    * **default**  : Four fixed targets for any environment (by position)

### Example:

    `python3.5 train_drone.py --test --render --verbose --model knet1 --cpt_fil <path_to_checkpoint> --env mountains --targets mn_lines`