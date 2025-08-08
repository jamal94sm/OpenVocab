#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --nodes=1
#SBATCH --gpus=a100:1  # Graham: t4 or v100 or a100 or dgx or a5000 or h100; Narval: a100, a100_4g.20g; Cedar: p100, p100l, v100l, a40
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4 # 8, 16
#SBATCH --mem=40G               # memory per node (ex: 16G) you can get more 
#SBATCH --time=20:00:00 		   # time period you need for your code (it is 12 hours for example)
#SBATCH --mail-user=<jamal73sm@gmail.com> 	# replace with your email address to get emails to know when it is started or failed. 
#SBATCH --mail-type=ALL


#cd /home/shahab33/projects/def-arashmoh/shahab33/FeD2P #Cedar

#cd /project/def-arashmoh/shahab33/Rohollah/projects/FeD2P #Graham

#cd /project/def-arashmoh/shahab33/FeD2P #Narval 

cd /project/def-arashmoh/shahab33/OpenVocab #Narval 


module purge
module load python
module load cuda

#source /home/shahab33/FeDK2P/bin/activate  	# Cedar

#source /home/shahab33/fed2p/bin/activate #Graham

source /home/shahab33/fed2p/bin/activate #Narval

#python main1.py --alpha_dirichlet 0.1 --output_name "alpha0.1_"  	# this is the direction and the name of your code
python main1.py
