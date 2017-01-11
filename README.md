# Factorization model for recommendation

This project is my first attempt to create a recommendation system using tensorflow. My first idea was to contribute to https://github.com/songgc/TF-recomm . But since my code took its own direction I decided to create this repository instead. Nevertherless the core model implemented here is the same as the one from that repository. In the future I want to implement new models of recommendations and have a more robust test framework. This is a one week project, so if it seems sloppy, it's not an accident.


### Requirements
* Tensorflow 
* Numpy
* Pandas 

## Usage

```
$ 
python3 recommender.py --help

optional arguments:
  -h, --help            show this help message and exit
  -d DIMENSION, --dimension DIMENSION
                        embedding vector size (default=15)
  -r REG, --reg REG     regularizer constant for the loss function
                        (default=0.05)
  -l LEARNING, --learning LEARNING
                        learning rate (default=0.001)
  -b BATCH, --batch BATCH
                        batch size (default=1000)
  -s STEPS, --steps STEPS
                        number of training (default=5000)
  -p PATH, --path PATH  ratings path (default=brucutuiv)

```


## Example

```
$
./download_data.sh
python3 recommender.py -s 20000
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
