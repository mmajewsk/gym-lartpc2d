sudo docker build -t lartpc -f .Dockerfile .
mkdir dump
sudo docker run --rm -v $(pwd):/home/app/ -v $(pwd)/data_lartpc2d:/home/data --cpus=1 lartpc /bin/bash "cd /home/app; python3 process_lartpc_data.py --entries 10"
