echo "Creating data_lartpc2d file"
mkdir data_lartpc2d
echo "Downloading data"
cd data_lartpc2d
wget -P data_lartpc2d http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/segmentation/multipvtx/test_10k.root
wget -P data_lartpc2d http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/segmentation/multipvtx/train_15k.root
echo "Building docker image"
sudo docker build -t lartpc -f .Dockerfile .
echo "Creating dump file"
mkdir dump
echo "Running docker and processing script"
sudo docker run --rm -v $(pwd):/home/app/ -v $(pwd)/data_lartpc2d:/home/data --cpus=1 lartpc /bin/bash "cd /home/app; python3 process_lartpc_data.py --entries 10"
