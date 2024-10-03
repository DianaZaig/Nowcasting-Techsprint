**Benchmarking**


Building on the research conducted by the Bank of Tanzania (Barongo and Mbelwa), we utilized a common dataset to evaluate the performance of the Temporal Fusion Transformer model against their 3 viable models --Random Forest model, Multi-Layer Perceptron model, and hybrid Random Forest Multi-Layer Perceptron model-- as well as a simple ARIMA.


Comparative evaluation of alternative models:
-----------------------------------------------------------------------------
Metric				RF	MLP	Hybrid RF-MLP	ARIMA	TFT
-----------------------------------------------------------------------------
Accuracy			98%	94%	96%		
Precision - Macro average	98%	94%	96%		
Precision - Weighted average	98%	94%	96%		
Recall - Macro average		98%	94%	96%		
Recall - Weighted average	98%	94%	96%		
F1 score - Macro average	98%	94%	96%		
F1 score - Weighted average	98%	94%	96%		
G-mean				98%	94%	96%		
Cohen’s Kappa			98%	92%	95%		
BA				98%	94%	96%		
Youden’s Index			95%	98%	90%		
AUC - Class 1			98%	95%	95%		
AUC - Class 2			97%	94%	95%		
AUC - Class 3			99%	95%	98%		
AUC - Class 4			100%	98%	99%		
AUC - Class 5			100%	99%	100%		
DP				2.6	2.3	2.6		
Type I error			2%	3.7%	0.8%		
Type II error			3.2%	8.3%	9.1%		
NLiR				2%	4.1%	0.9%		




References:
Rweyemamu Ignatius Barongo, Jimmy Tibangayuka Mbelwa,
Using machine learning for detecting liquidity risk in banks,
Machine Learning with Applications,
Volume 15,
2024,
100511,
ISSN 2666-8270,
https://doi.org/10.1016/j.mlwa.2023.100511.
