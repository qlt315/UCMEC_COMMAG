# When User-centric Network Meets Nobile Edge Computing: Challenges and Optimization
Revsied simulation code based on our paper in IEEE Communications Magazine -- When User-centric Network Meets Nobile Edge Computing: Challenges and Optimization

This version of the code has made changes to the original paper code, which are mainly summarized as follows:


(1) The optimization objective has been changed from the energy consumption to the average total delay


(2) The computing resource allocation stategy is obtained by solving the convex problem via the cvxpy library, rather than explored by the PPO-based agent


(3) Fixed some bugs in the previous code and optimized the code structure

This paper is available on https://ieeexplore.ieee.org/document/9952196


**Running Environments:**


python==3.7.9   numpy==1.19.4


pytorch==1.12.0   tensorboard==0.6.0


gym==0.21.0


