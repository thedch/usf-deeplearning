# Deep Learning Fellowship
## [The Data Institute | University of San Francisco](http://course.fast.ai/part2.html)

This repository is a catch-all for lecture notes, documentation, code, and other content created during my Spring 2018 Deep Learning Fellowship. 

### Handy Shell Commands

`nvidia-smi # View graphics card usage`

`jupyter notebook password # Set a notebook password to avoid having to copy token link`

`wget setup.thedaniel.me -O ./setup.sh && chmod 700 ./setup.sh && ./setup.sh # Quickly download .vimrc, .bashrc, etc`

### AWS Boot Script

I have created a handy boot script that automatically powers on an AWS instance, connects with convenient port forwarding, records the time spent on the instance, and powers off the instance when you disconnect. It requires minor configuration for your own use. You can find it [here](https://github.com/thedch/bash-scripts/blob/master/ssh-into-aws.sh). 
