from credit_risk import CreditRisk

total_mortgages = [[0.15, 0.1, 100000],
                    [0.25, 0.05, 200000],
                    [0.2, 0.07, 300000],
                    [0.02, 0.01, 400000],
                    [0.05, 0.05, 300000],
                    [0.2, 0.03, 390000],
                    [0.01, 0.01, 100000],
                    [0.03, 0.09, 120000],
                    [0.2, 0.07, 300000],
                    [0.02, 0.01, 400000],
                    [0.05, 0.05, 300000],
                    [0.25, 0.05, 310000],
                    [0.01, 0.01, 600000],
                    [0.05, 0.01, 800000],
                    [0.04, 0.01, 300000],
                    [0.2, 0.4, 560000],
                    [0.7, 0.10, 100000],
                    [0.04, 0.01, 100000],
                    [0.2, 0.07, 300000],
                    [0.02, 0.01, 400000],
                    [0.05, 0.05, 300000],
                    [0.02, 0.03, 390000],
                    [0.1, 0.01, 200000],
                    [0.04, 0.01, 600000],
                    [0.03, 0.01, 700000]]

# Confidence level for VaR and CVaR. On BaselII around 99,9%
alpha = 0.03
problem_size = 5
mortgages = total_mortgages[:problem_size]

cr = CreditRisk()
cr.run(alpha, mortgages, 2, "simulator", False)


#problem_size = 12
#mortgages = total_mortgages[:problem_size]
#print(cr.classical_run(alpha, mortgages))