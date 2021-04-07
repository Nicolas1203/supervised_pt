# Docker commands

## Build

    docker build -t training_pt .
    docker build -t tensorboard .

## Run

    docker run -it --rm -u $(id -u):$(id -g) --name training_nicolas -v $PWD:/home/micheln -v /data/dataset/yt:/data --gpus all --shm-size=8G training_pt

    docker run -it --rm -u $(id -u):$(id -g) --name tensorboard -p 6006:6006 -v $PWD:/home/micheln tensorboard