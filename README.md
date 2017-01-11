# Factorization model for recommendation

This project is my first attempt to create a recommendation system using tensorflow. My first idea was to contribute to https://github.com/songgc/TF-recomm . But since my code took its own direction I decided to create this repository instead. Nevertherless the core model implemented here is the same as the one from that repository. In the future I want to implement new models of recommendations and have a more robust test framework; This is a one week project, so if it seems sloppy, it's not an accident.


### Requirements
* Tensorflow 
* Numpy
* Pandas 

## Usage

```
$ 
./download_data.sh
./svd_train_val.py --help
usage: mdp-problog.py [-h] [-g GAMMA] [-e EPS] [-v VERBOSE] domain instance

positional arguments:
  domain                path to MDP domain file
  instance              path to MDP instance file

optional arguments:
  -h, --help            show this help message and exit
  -g GAMMA, --gamma GAMMA
                        discount factor (default=0.9)
  -e EPS, --eps EPS     relative error  (default=0.1)
  -v VERBOSE, --verbose VERBOSE
                        verbose mode (default=0)
```


## Example

```
$ ./mdp-problog.py models/sysadmin/domain.pl models/sysadmin/star2.pl --eps 0.1 --gamma 0.9

  >> Preprocessing program ... Done in 0.025sec.
  >> Relevant grounding ... Done in 0.050sec.
  >> Compilation ... Done in 0.017sec.

>> Policy:
Pi(running(c1,0)=0, running(c2,0)=0, running(c3,0)=0) = reboot(c1)
Pi(running(c1,0)=0, running(c2,0)=0, running(c3,0)=1) = reboot(c1)
Pi(running(c1,0)=0, running(c2,0)=1, running(c3,0)=0) = reboot(c1)
Pi(running(c1,0)=0, running(c2,0)=1, running(c3,0)=1) = reboot(c1)
Pi(running(c1,0)=1, running(c2,0)=0, running(c3,0)=0) = reboot(c2)
Pi(running(c1,0)=1, running(c2,0)=0, running(c3,0)=1) = reboot(c2)
Pi(running(c1,0)=1, running(c2,0)=1, running(c3,0)=0) = reboot(c3)
Pi(running(c1,0)=1, running(c2,0)=1, running(c3,0)=1) = reboot(none)

>> Value:
V(running(c1,0)=0, running(c2,0)=0, running(c3,0)=0) = 19.1341
V(running(c1,0)=0, running(c2,0)=0, running(c3,0)=1) = 20.6162
V(running(c1,0)=0, running(c2,0)=1, running(c3,0)=0) = 20.6162
V(running(c1,0)=0, running(c2,0)=1, running(c3,0)=1) = 21.8910
V(running(c1,0)=1, running(c2,0)=0, running(c3,0)=0) = 20.5864
V(running(c1,0)=1, running(c2,0)=0, running(c3,0)=1) = 23.6645
V(running(c1,0)=1, running(c2,0)=1, running(c3,0)=0) = 23.6645
V(running(c1,0)=1, running(c2,0)=1, running(c3,0)=1) = 25.3019

>> Value iteration converged in 0.295sec after 45 iterations.
@ Average time per iteration = 0.007sec.
@ Max error = 0.02158
```
