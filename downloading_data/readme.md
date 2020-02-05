# Data for lartpc_game

I am not the owner of the original data, nor do I have permission to redistribute it.
So head on here http://deeplearnphysics.org/DataChallenge/ and download the train_15k.root or test_10k.root from 
"MULTIPLE PARTICLE SAMPLE" section.

## Downloading the original data

Put the downloaded data here, in `data_lartpc2d` folder, and follow instructions below.
Or use `run.sh`, (not tested) to do all of the things.

## Processing data

To process the data, i am using [larcv repository](https://github.com/DeepLearnPhysics/larcv2)
which has a .Dockerfile that I have modified. 

The modified .Dockerfile is in this repo at `lartpc_game/downloading_data/.Dockerfile`

First we need to build it:

```bash
sudo docker build -t lartpc -f .Dockerfile .
```

Then create dump folder for the processed data.

```bash
mkdir dump
```

Then run the docker and processing script:

```bash
sudo docker run --rm -v $(pwd):/home/app/ --cpus=1 -it lartpc 
cd /home/app
python3 process_lartpc_data.py 
```

You can use scripts arguments to personalise the process in process_lartpc_data.py, look inside the file for docs.

**warning, option -h is not working with this script**

If you dont need all of the data, and just want to run this library, use less cases e.g. `python3 process_lartpc_data.py --entries 10`

