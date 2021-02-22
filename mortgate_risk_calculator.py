import numpy as np

from qiskit import QuantumRegister, QuantumCircuit, Aer, execute
from qiskit.circuit.library import IntegerComparator, LinearAmplitudeFunction, WeightedAdder
from qiskit.aqua.algorithms import IterativeAmplitudeEstimation

from qiskit.finance.applications import GaussianConditionalIndependenceModel as GCI

######################
# Problem parameters #
######################

# Each asset mapped as [default probability, sensitivity o the PDF, loss given default (expressed in '0.000)]

problem_size = 8

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
alpha = 0.02


# Mapping parameters
# Loss Given Default multiplier (we can't map very big numbers, so we eliminate zeroes, from X00,000 -> X0)
lgd_factor = 100000

# Z represents our distribution, discretized with n qubits. The more qubits, the merrier. (I.e. the more values we will be able to approximate)
n_z = 2
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

    return expected_loss, exact_var, exact_cvar, p_l_less_than_var, losses


def get_uncertainty_model(n_z, z_max, probability_default, sensitivity_z):
    return GCI(n_z, z_max, probability_default, sensitivity_z) 

def get_weighted_adder(n_z, K, loss_given_default):
    return WeightedAdder(n_z + K, [0]*n_z + loss_given_default)


def get_quantum_expected_loss(uncertainty_model, weighted_adder, loss_given_default):

    # define linear objective function
    breakpoints = [0]
    slopes = [1]
    offsets = [0]
    f_min = 0
    f_max = sum(loss_given_default)
    c_approx = 0.25

    objective = LinearAmplitudeFunction(
        weighted_adder.num_sum_qubits,
        slope=slopes, 
        offset=offsets, 
        # max value that can be reached by the qubit register (will not always be reached)
        domain=(0, 2**weighted_adder.num_sum_qubits-1),  
        image=(f_min, f_max),
        rescaling_factor=c_approx,
        breakpoints=breakpoints,
        name="F"
    )

    # define the registers for convenience and readability
    qr_state = QuantumRegister(uncertainty_model.num_qubits, 'state')
    qr_sum = QuantumRegister(weighted_adder.num_sum_qubits, 'sum')
    qr_carry = QuantumRegister(weighted_adder.num_carry_qubits, 'carry')
    qr_obj = QuantumRegister(1, 'objective')
    qr_control = QuantumRegister(1, 'control')


    # define the circuit
    if weighted_adder.num_control_qubits > 0:
        state_preparation = QuantumCircuit(qr_state, qr_obj, qr_sum, qr_carry, qr_control, name='A')
        state_preparation.append(uncertainty_model.to_gate(), qr_state)
        state_preparation.append(weighted_adder.to_gate(), qr_state[:] + qr_sum[:] + qr_carry[:] + qr_control[:])
        state_preparation.append(objective.to_gate(), qr_sum[:] + qr_obj[:])
        state_preparation.append(weighted_adder.to_gate().inverse(), qr_state[:] + qr_sum[:] + qr_carry[:] + qr_control[:])
    else: 
        state_preparation = QuantumCircuit(qr_state, qr_obj, qr_sum, qr_carry, name='A')
        state_preparation.append(uncertainty_model.to_gate(), qr_state)
        state_preparation.append(weighted_adder.to_gate(), qr_state[:] + qr_sum[:] + qr_carry[:])
        state_preparation.append(objective.to_gate(), qr_sum[:] + qr_obj[:])
        state_preparation.append(weighted_adder.to_gate().inverse(), qr_state[:] + qr_sum[:] + qr_carry[:])

    backend = Aer.get_backend('qasm_simulator')

    epsilon_iae = 0.01
    alpha_iae = 0.05
    iae = IterativeAmplitudeEstimation(state_preparation=state_preparation,
                                    epsilon=epsilon_iae, alpha=alpha_iae,
                                    objective_qubits=[len(qr_state)],
                                    post_processing=objective.post_processing)
    result = iae.run(quantum_instance=backend, shots=100)

    conf_int = np.array(result['confidence_interval'])
    return conf_int, result['estimation']

def build_cdf_state_preparation(comparator_value, uncertainty_model, weighted_adder):
    cdf_qr_state = QuantumRegister(uncertainty_model.num_qubits, 'state')
    cdf_qr_sum = QuantumRegister(weighted_adder.num_sum_qubits, 'sum')
    cdf_qr_carry = QuantumRegister(weighted_adder.num_carry_qubits, 'carry')
    cdf_qr_obj = QuantumRegister(1, 'objective')
    
    comparator = IntegerComparator(weighted_adder.num_sum_qubits, comparator_value + 1, geq=False)

    if weighted_adder.num_control_qubits > 0:
        cdf_qr_control = QuantumRegister(weighted_adder.num_control_qubits, 'control')
        cdf_state_preparation = QuantumCircuit(cdf_qr_state, cdf_qr_obj, cdf_qr_sum, cdf_qr_carry, cdf_qr_control, name='A')
        cdf_state_preparation.append(uncertainty_model, cdf_qr_state)
        cdf_state_preparation.append(weighted_adder, cdf_qr_state[:] + cdf_qr_sum[:] + cdf_qr_carry[:] + cdf_qr_control[:])
        cdf_state_preparation.append(comparator, cdf_qr_sum[:] + cdf_qr_obj[:] + cdf_qr_carry[:])
        cdf_state_preparation.append(weighted_adder.inverse(), cdf_qr_state[:] + cdf_qr_sum[:] + cdf_qr_carry[:] + cdf_qr_control[:])
    else:
        cdf_state_preparation = QuantumCircuit(cdf_qr_state, cdf_qr_obj, cdf_qr_sum, cdf_qr_carry, name='A')
        cdf_state_preparation.append(uncertainty_model, cdf_qr_state)
        cdf_state_preparation.append(weighted_adder, cdf_qr_state[:] + cdf_qr_sum[:] + cdf_qr_carry[:])
        cdf_state_preparation.append(comparator, cdf_qr_sum[:] + cdf_qr_obj[:] + cdf_qr_carry[:])
        cdf_state_preparation.append(weighted_adder.inverse(), cdf_qr_state[:] + cdf_qr_sum[:] + cdf_qr_carry[:])
    
    return cdf_state_preparation

def get_quantum_var(alpha, losses, uncertainty_model, weighted_adder):

    backend=Aer.get_backend('statevector_simulator')
    target_value = 1 - alpha
    low_level = min(losses) - 1
    high_level = max(losses)
    low_value = 0
    high_value = 1

    num_eval = 0

    # check if low_value already satisfies the condition
    if low_value > target_value:
        level = low_level
        value = low_value
    elif low_value == target_value:
        level = low_level
        value = low_value

    # check if high_value is above target
    if high_value < target_value:
        level = high_level
        value = high_value
    elif high_value == target_value:
        level = high_level
        value = high_value


    while high_level - low_level > 1:

        level = int(np.round((high_level + low_level) / 2.0))
        num_eval += 1

        cdf_state_pareparation = build_cdf_state_preparation(level, uncertainty_model, weighted_adder)
        iae_var = IterativeAmplitudeEstimation(state_preparation=cdf_state_pareparation, epsilon=0.01, alpha=0.05, objective_qubits = [uncertainty_model.num_qubits])
        value = iae_var.run(quantum_instance=backend, shots=100)['estimation']

        if value >= target_value:
            high_level = level
            high_value = value
        else:
            low_level = level
            low_value = value

    return level, value


def get_quantum_cvar(var, estimated_probability, uncertainty_model, weighted_adder):
    breakpoints = [0, var]
    slopes = [0, 1]
    offsets = [0, 0]  # subtract VaR and add it later to the estimate
    f_min = 0
    f_max = sum(loss_given_default) - var
    c_approx = 0.25

    cvar_objective = LinearAmplitudeFunction(
        weighted_adder.num_sum_qubits,
        slopes,
        offsets,
        domain=(0, 2**weighted_adder.num_sum_qubits - 1),
        image=(f_min, f_max),
        rescaling_factor=c_approx,
        breakpoints=breakpoints
    )

    # define the registers for convenience and readability
    cvar_qr_state = QuantumRegister(uncertainty_model.num_qubits, 'state')
    cvar_qr_sum = QuantumRegister(weighted_adder.num_sum_qubits, 'sum')
    cvar_qr_carry = QuantumRegister(weighted_adder.num_carry_qubits, 'carry')
    cvar_qr_obj = QuantumRegister(1, 'objective')
    cvar_qr_work = QuantumRegister(cvar_objective.num_ancillas - len(cvar_qr_carry), 'work')

    if weighted_adder.num_control_qubits > 0:
        cvar_qr_control = QuantumRegister(weighted_adder.num_control_qubits, 'control')
        cvar_state_preparation = QuantumCircuit(cvar_qr_state, cvar_qr_obj, cvar_qr_sum, cvar_qr_carry, cvar_qr_control, cvar_qr_work, name='A')
        cvar_state_preparation.append(uncertainty_model, cvar_qr_state)
        cvar_state_preparation.append(weighted_adder, cvar_qr_state[:] + cvar_qr_sum[:] + cvar_qr_carry[:] + cvar_qr_control[:])
        cvar_state_preparation.append(cvar_objective, cvar_qr_sum[:] + cvar_qr_obj[:] + cvar_qr_carry[:] + cvar_qr_work[:])
        cvar_state_preparation.append(weighted_adder.inverse(), cvar_qr_state[:] + cvar_qr_sum[:] + cvar_qr_carry[:] + cvar_qr_control[:])
    else:
        cvar_state_preparation = QuantumCircuit(cvar_qr_state, cvar_qr_obj, cvar_qr_sum, cvar_qr_carry, cvar_qr_work, name='A')
        cvar_state_preparation.append(uncertainty_model, cvar_qr_state)
        cvar_state_preparation.append(weighted_adder, cvar_qr_state[:] + cvar_qr_sum[:] + cvar_qr_carry[:])
        cvar_state_preparation.append(cvar_objective, cvar_qr_sum[:] + cvar_qr_obj[:] + cvar_qr_carry[:] + cvar_qr_work[:])
        cvar_state_preparation.append(weighted_adder.inverse(), cvar_qr_state[:] + cvar_qr_sum[:] + cvar_qr_carry[:])

    # set target precision and confidence level
    epsilon_iae = 0.01
    alpha_iae = 0.05

    # construct amplitude estimation
    ae_cvar = IterativeAmplitudeEstimation(state_preparation=cvar_state_preparation,
                                        epsilon=epsilon_iae, alpha=alpha_iae,
                                        objective_qubits=[len(cvar_qr_state)],
                                        post_processing=cvar_objective.post_processing)
    result_cvar = ae_cvar.run(quantum_instance=Aer.get_backend('qasm_simulator'), shots=100)

    d = (1.0 - estimated_probability)
    v = result_cvar['estimation'] / d if d != 0 else 0

    return v + var


estimated_vars = []
var_errors = []
estimated_cvars = []
cvar_errors = []
z_range = range(2,10)

for n_z in z_range:

    uncertainty_model = get_uncertainty_model(n_z, z_max, probability_default, sensitivity_z)
    expected_loss, exact_var, exact_cvar, p_l_less_than_var, losses = get_classical_expectation_loss(uncertainty_model, K)

    weighted_adder = get_weighted_adder(n_z, K, loss_given_default)
    confidence_expected_loss, expected_loss_estimation = get_quantum_expected_loss(uncertainty_model, weighted_adder, loss_given_default)    

    estimated_var, estimated_var_probability = get_quantum_var(alpha, losses, uncertainty_model, weighted_adder)
    estimated_cvar = get_quantum_cvar(estimated_var, estimated_var_probability, uncertainty_model, weighted_adder)

    
    print('-------------------------')
    print('-------------------------')
    print('Test for discretization: ', n_z)
    print('-------------------------')
    print('LGD: ', loss_given_default, ' Total Assets value: $ {0:12,.0f}'.format(sum(loss_given_default)*lgd_factor))
    print('Assets: ', K)
    print('Assets default Probabilities: ', probability_default)
    print('-------------------------')
    print('Expected Loss E[L]:                $ {0:12,.0f}'.format(expected_loss * lgd_factor))
    print('Estimated Loss E[L]: $ {0:9,.0f}'.format(expected_loss_estimation * lgd_factor))
    print('Confidence interval: \t[%.0f, %.0f]' % (tuple(confidence_expected_loss * lgd_factor)))
    print('-------------------------')
    print('Value at Risk VaR[L](%.2f):        $ {0:12,.0f}'.format((exact_var*lgd_factor)) % (alpha))
    print('Estimated Value at Risk: $ {0:9,.0f}'.format(estimated_var * lgd_factor))
    error_var_estimation = 1-(exact_var) / estimated_var
    print('Error VaR Estiamtion: ', error_var_estimation)
    print('P[L <= VaR[L]](%.2f):              %.4f' % (alpha, p_l_less_than_var))
    print('Estimated P[L <= VaR[L]](%.2f):              %.3f' % (alpha, estimated_var_probability))
    print('-------------------------')
    print('Conditional Value at Risk CVaR[L]: $ {0:12,.0f}'.format(exact_cvar * lgd_factor))
    print('Estimated Conditional Value at Risk CVaR[L]: $ {0:12,.0f}'.format(estimated_cvar * lgd_factor))
    error_cvar_estimation = 1-(exact_cvar) / estimated_cvar
    print('Error CVaR Estiamtion: ', error_cvar_estimation)

    estimated_cvars.append(estimated_cvar)
    estimated_vars.append(estimated_var)
    var_errors.append(error_var_estimation)
    cvar_errors.append(error_cvar_estimation)


print(estimated_vars)
print(var_errors)
print(estimated_cvars)
print(cvar_errors)