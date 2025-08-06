#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4 # 8, 16
#SBATCH --mem=32G               # memory per node (ex: 16G) you can get more 
#SBATCH --time=01:00:00 		   # time period you need for your code (it is 12 hours for example)
#SBATCH --mail-user=<jamal73sm@gmail.com> 	# replace with your email address to get emails to know when it is started or failed. 
#SBATCH --mail-type=ALL

#cd /home/shahab33/projects/def-arashmoh/shahab33/FeDK2P
cd /project/def-arashmoh/shahab33/Rohollah/projects/FeDK2P/FeDK2P


module purge
module load python
module load cuda

#source ~/FeDK2P/bin/activate  	# activate your environment
source /project/def-arashmoh/shahab33/Rohollah/projects/FeDK2P/FeDK2P/fedk2p/bin/activate 

python fedk2p.py   	# this is the direction and the name of your code
