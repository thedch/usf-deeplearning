# Deep Learning Fellowship
## [The Data Institute | University of San Francisco](http://course.fast.ai/part2.html)

This repository is a catch-all for lecture notes, documentation, code, and other content created during my Spring 2018 Deep Learning Fellowship. 

### Handy Shell Commands

View graphics card usage:

`nvidia-smi`

Set a notebook password to avoid having to copy token link:

`jupyter notebook password`

Quickly download .vimrc, .bashrc, etc:

`wget setup.thedaniel.me -O ./setup.sh && chmod 700 ./setup.sh && ./setup.sh`

Create a link from the fastai library to your current directory. Useful for Python imports:

`sudo ln -s ~/fastai/fastai/ fastai`

### AWS Boot Script

I have created a handy boot script that automatically powers on an AWS instance, connects with convenient port forwarding, records the time spent on the instance, and powers off the instance when you disconnect. It requires minor configuration for your own use. You can find it [here](https://github.com/thedch/bash-scripts/blob/master/ssh-into-aws.sh). 
