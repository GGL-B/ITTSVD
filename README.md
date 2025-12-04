
ITTSVD


Title: Irregular Tensor Truncation Singular Value Decomposition for single-cell multi-omics data analysis

We introduce a novel framework, ITTSVD, which integrates biological data into irregular tensor data. 
Based on the properties of generalized singular values, a new truncation function is proposed; 
after decomposing the original data, subsequent biological downstream analyses are performed using 
the irregular low-rank tensor and shared matrix. Extensive experiments are conducted on multiple datasets, 
and the results demonstrate that ITTSVD outperforms other advanced comparative methods.


Description: ITTSVD is written in the Python programming language. To use, please download the ITTSVD folder 
and follow the instructions provided in the README.


To operate:
	* for the example of ITTSVD_Demo:
		The main function, we have provided some pre-data for ITTSVD, if you want to run ITTSVD framwork, Please load the data into 
        the path you need first, please run the script "ITTSVD_Demo" directly.   
   
    * for the example of SS:
        Under this code, we will obtain the information required for subsequent experiments, such as the final value, pre-label,
        and evaluation metric Normalized Mutual Information (NMI), Adjusted Mutual Information (AMI), Adjusted Rand Index (ARI), accuracy.
        
    * bestMap.py:
        permute labels of L2 to match L1 as good as possible, the generated vector is used to calculate the subsequent NMI(compute_NMI.py)
        AMI(AMI.py), ARI(ARI.py), accuracy, pre_label.
    

    * ITTSVD.PY - The iterative process of ITTSVD algorithm.

    * Some of the data in the paper is included in the "Data" file.
