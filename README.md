# PBLC-sysu 
One classification is a common situation in the realistic context.  However, most methods focus on  binary classification to solve practical problems，which is not appropriate and efficient. Therefore, we propose Positive and unlabeled Background learning with Constraint as a BackBone for overcoming data shortcomings that misses reliable negative samples. Extensive experimental evaluations are conducted in the field of point cloud 、disaster monitoring、remote sensing and species distribution to verify the proposed method. The reaults show PBLC outperforms high accuracy and alleviate data defect。

The paper is published in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing: 
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9201373

The theoretical basis of PBLC algorithm is PBL，publised in 
https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1600-0587.2011.06888.x

$Install on Windons 10 / Linux

1.Ensure you have installed tensorflow1.0+ in your enviroment.

$Data：./TF/data
This folder contains simple train data and test data based on Species Distribution Model.

$Train:
In the model folder, there are two networks：ANN and GLM. And you can choose the model you need in the train.py. In addition, the network needs the train data、the numble of features、the path saved your model and the test data. Your network is working when running the train.py.
training data: train_data.csv
test data: test_data.csv
predicted probability: ./result/
class prior: Pr(y = 1) = 0.5
number of positive data: 1000
number of background data: 5000
true value of c: 1000 / (1000 + 5000 * 0.5) = 0.2857

$Authors

Wenkai Li -owner of patent of the PBLC algorithm

May Hu -editor
