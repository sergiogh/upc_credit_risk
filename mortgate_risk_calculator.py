import numpy as np

from qiskit import QuantumRegister, QuantumCircuit, Aer, execute
from qiskit.circuit.library import IntegerComparator
from qiskit.aqua.algorithms import IterativeAmplitudeEstimation

from qiskit.finance.applications import GaussianConditionalIndependenceModel as GCI

######################
# Problem parameters #
######################

# Each asset mapped as [default probability, sensitivity o the PDF, loss given default (expressed in '0.000)]

problem_size = 3

mortgages = [[0.15, 0.1, 100000],
             [0.25, 0.05, 200000],
             [0.2, 0.07, 300000],
             [0.02, 0.01, 400000],
             [0.05, 0.05, 300000],
             [0.2, 0.03, 390000],
             [0.01, 0.01, 100000],
             [0.03, 0.09, 120000]]

# Get only a subset when making the problem smaller
mortgages = mortgages[:problem_size]

# Confidence level for VaR and CVaR. On BaselII around 99,9%
alpha = 0.05


# Mapping parameters
# Loss Given Default multiplier (we can't map very big numbers, so we eliminate zeroes, from X00,000 -> X0)
lgd_factor = 100000

# Z represents our distribution, discretized with n qubits. The more qubits, the merrier. (I.e. the more values we will be able to approximate)
n_z = 5
z_max = 2
z_values = np.linspace(-z_max, z_max, 2**n_z)

K = len(mortgages)

probability_default = []
sensitivity_z = []
loss_given_default = []
for m in mortgages:
    probability_default.append(m[0])
    sensitivity_z.append(m[1])
    loss_given_default.append(int(m[2] / lgd_factor))   # LGD is simplified, reduced proportionately and taken only the integer part


def get_classical_expectation_loss(uncertainty_model, K):
    job = execute(uncertainty_model, backend=Aer.get_backend('statevector_simulator'))
    # analyze uncertainty circuit and determine exact solutions using Montecarlo over the circuit.
    # We could compare with a fully classical MC modeling the GCI, but it is too difficult and I'm lazy

    p_z = np.zeros(2**n_z)
    p_default = np.zeros(K)
    values = []
    probabilities = []
    num_qubits = uncertainty_model.num_qubits
    for i, a in enumerate(job.result().get_statevector()):
        # get binary representation
        b = ('{0:0%sb}' % num_qubits).format(i)
        prob = np.abs(a)**2

        # extract value of Z and corresponding probability    
        # Note Z i mapped in the least significant n_z qubits. We add probabilities for each element in the distribution
        i_normal = int(b[-n_z:], 2)
        p_z[i_normal] += prob

        # determine overall default probability for k 
        # Most significant qubits represent 1 for default of that asset.
        loss = 0
        for k in range(K):
            if b[K - k - 1] == '1':
                p_default[k] += prob
                loss += loss_given_default[k]
        values += [loss]
        probabilities += [prob]   

    values = np.array(values)
    probabilities = np.array(probabilities)
        
    # L = λ1*X1(Z) + λ2*X2(Z) + ... + λn*Xn(Z)
    expected_loss = np.dot(values, probabilities)

    losses = np.sort(np.unique(values))
    pdf = np.zeros(len(losses))
    for i, v in enumerate(losses):
        pdf[i] += sum(probabilities[values == v])
    cdf = np.cumsum(pdf)

    i_var = np.argmax(cdf >= 1-alpha)
    exact_var = losses[i_var]
    exact_cvar = np.dot(pdf[(i_var+1):], losses[(i_var+1):])/sum(pdf[(i_var+1):])

    # Calculate P[L <= VaR[L]]
    alpha_point = np.where(values == exact_var)[0].min()
    p_l_less_than_var = np.sum(probabilities[:alpha_point])

    return expected_loss, exact_var, exact_cvar, p_l_less_than_var


def get_uncertainty_model(n_z, z_max, probability_default, sensitivity_z):
    return GCI(n_z, z_max, probability_default, sensitivity_z) 


uncertainty_model = get_uncertainty_model(n_z, z_max, probability_default, sensitivity_z)
expected_loss, exact_var, exact_cvar, p_l_less_than_var = get_classical_expectation_loss(uncertainty_model, K)


print('LGD: ', loss_given_default, ' Total Assets value: $ {0:12,.0f}'.format(sum(loss_given_default)*lgd_factor))
print('Assets: ', K)
print('Assets default Probabilities: ', probability_default)
print('Expected Loss E[L]:                $ {0:12,.0f}'.format(expected_loss*lgd_factor))
print('Value at Risk VaR[L](%.2f):        $ {0:12,.0f}'.format((exact_var*lgd_factor)) % (alpha))
print('P[L <= VaR[L]](%.2f):              %.4f' % (alpha, p_l_less_than_var))
print('Conditional Value at Risk CVaR[L]: $ {0:12,.0f}'.format(exact_cvar*lgd_factor))