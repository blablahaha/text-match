## Overview
Implementation of deep text matching algorithms, such as DSSM...(to be continued).

## Data Structure
1. Training set: In the training file, each line contains four parts,  and word id of each part are concatenated by comma:
	* Id of query
	* Query
	* Positive doc
	* Negative docs: number of negative doc is defined in config file. And each negative sample is concatenated by semicolon