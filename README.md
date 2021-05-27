# UPC Postgrad in Quantum Computing final thesis - Sergio Gago
## May 2021

This repository contains all the research, process and engineering on how to apply quantum computing algorithms for simulating credit risk scenarios, clasically done with Montecarlo analysis.

Index:
[Credit Risk Analysis with Quantum Computers](final_project_credit_risk.pdf)
_Note this is not your typical academic paper, please do not cite. Use at your own peril_

[Final Project Notebook](final_project_credit_risk.ipynb)
[Credit Risk class](credit_risk.py) - You can import this module as is and use it for your own calculations
[Class usage Example](mortgage_risk_calculator.py) - Usage example
[Fannie Mae clustering hybrid system](dataset_clustering_analysis.ipynb) - Example of clustering with KMeans + Quantum algorithm simulation

In the rest of the repository you can find small tutorials and manuals on how to map probability distributions in a quantum circuit, how to build the QAE algorithm by hand or how to use the same method to calculate the Blach-Scholes equation for option pricing.

## Raw resources and materials

Quick problem explanation
[Quantum Finance: Credit Risk Analysis with QAE | by Alice Liu](https://medium.com/@aliceliu2004/quantum-finance-credit-risk-analysis-with-qae-b339b585aaed)

Brassard’s original paper summary and explanation
[Quantum Amplitude Amplification and Estimation](https://blog.alexandrecarlier.com/projects/quantum_ampl_estimation/)

Qiskit Tutorial
https://qiskit.org/documentation/tutorials/finance/09_credit_risk_analysis.html

QPE Qiskit Tutorial
https://qiskit.org/textbook/ch-algorithms/quantum-phase-estimation.html

Qiskit built-in function
https://qiskit.org/documentation/stubs/qiskit.aqua.algorithms.AmplitudeEstimation.html#qiskit.aqua.algorithms.AmplitudeEstimation

Build Q operator in Qiskit
https://quantumcomputing.stackexchange.com/questions/13524/how-to-define-q-operator-in-quantum-amplitude-estimation

Process
Quantum Counting: https://qiskit.org/textbook/ch-algorithms/quantum-counting.html#6.-References-


### Papers:
Comparison of Amplitude Estimation Algorithms by Implementations
https://www.researchgate.net/publication/341342061_Comparison_of_Amplitude_Estimation_Algorithms_by_Implementation

Quantum Amplitude Amplification and Estimation (Brassard, Mosca, Hoyer, Tapp 2000)
https://arxiv.org/pdf/quant-ph/0005055.pdf

Iterative Quantum Amplitude Estimation (D. Grinko, J. Gacon, C. Zoufal, S. Waaazqoerner 2019)
https://arxiv.org/abs/1912.05559

Comparing several QAE techniques in IBM (2020)
https://arxiv.org/pdf/2008.02102.pdf

Quantum approximate counting, simplified (Aaronson 2019) - QAE without QFT
https://arxiv.org/abs/1908.10846

Quantum Speedup for Montecarlo Methods (Montanaro 2015)
https://arxiv.org/abs/1504.06987

Creating superpositions that correspond to efficiently integrable probability distributions
https://arxiv.org/abs/quant-ph/0208112

Quantum Computing for Finance: state of the art and future prospects (Egger, Gambella, Marcek, etc..  2020):
https://arxiv.org/pdf/2006.14510.pdf
			
Marek Rutkowski and Silvio Tarca, “Regulatory capital modelling for credit risk,”
https://arxiv.org/abs/1412.1183

SVMs and Quantum classifiers to establish credit risk: https://www.researchgate.net/publication/341700842_Credit_risk_scoring_with_a_supervised_quantum_classifier

Basel II Requirements: https://www.bis.org/publ/bcbs107.pdf
