
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit, Aer, execute, IBMQ
from qiskit.providers.aer.noise import NoiseModel
from qiskit.circuit.library import IntegerComparator, LinearAmplitudeFunction, WeightedAdder
from qiskit.aqua.algorithms import IterativeAmplitudeEstimation
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.finance.applications import GaussianConditionalIndependenceModel as GCI
import random

class CreditRisk():

    device_backend = False
    backend = False
    probability_default = []
    sensitivity_z = []
    loss_given_default = []
    loans = []
    n_z = 4
    z_max = 2
    z_values = False
    alpha = 0.03

    def __init__(self):
        

        random.seed(42)
        np.random.seed(42)
        aqua_globals.random_seed = 123456

        z_values = np.linspace(-self.z_max, self.z_max, 2**self.n_z)

        self.probability_default = []
        self.sensitivity_z = []
        self.loss_given_default = []
        self.loans = []

        IBMQ.load_account()

    def get_classical_expectation_loss(self):

        K = len(self.loans)
        uncertainty_model = self.get_uncertainty_model()
        classical_expectations_qc = QuantumCircuit(uncertainty_model.num_qubits)
        classical_expectations_qc.append(uncertainty_model, range(uncertainty_model.num_qubits))
        classical_expectations_qc.measure_all()
        shots = 4000
        job = execute(classical_expectations_qc, backend=self.device_backend, shots=shots)
        counts = job.result().get_counts()
        p_z = np.zeros(2**self.n_z)
        p_default = np.zeros(K)
        values = []
        probabilities = []

        for i in counts:
            prob = counts[i]/shots

            # extract value of Z and corresponding probability    
            # Note Z i mapped in the least significant n_z qubits. We add probabilities for each element in the distribution
            i_normal = int(i[-self.n_z:], 2)
            p_z[i_normal] += prob

            # determine overall default probability for k 
            # Most significant qubits represent 1 for default of that asset.
            loss = 0
            for k in range(K):
                if i[K - k - 1] == '1':
                    p_default[k] += prob
                    loss += self.loss_given_default[k]
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

        i_var = np.argmax(cdf >= 1-self.alpha)
        exact_var = losses[i_var]
        exact_cvar = np.dot(pdf[(i_var+1):], losses[(i_var+1):])/sum(pdf[(i_var+1):])
        print(exact_var)
        print(exact_cvar)
        # Calculate P[L <= VaR[L]]
        p_l_less_than_var = cdf[exact_var]

        return expected_loss, exact_var, exact_cvar, p_l_less_than_var, losses

    def get_uncertainty_model(self):
        return GCI(self.n_z, self.z_max, self.probability_default, self.sensitivity_z) 

    def get_weighted_adder(self):
        K = len(self.loans)
        return WeightedAdder(self.n_z + K, [0]*self.n_z + self.loss_given_default)

    def get_quantum_expected_loss(self):

        uncertainty_model = self.get_uncertainty_model()
        weighted_adder = self.get_weighted_adder()

        # define linear objective function
        breakpoints = [0]
        slopes = [1]
        offsets = [0]
        f_min = 0
        f_max = sum(self.loss_given_default)
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

        epsilon_iae = 0.01
        alpha_iae = 0.05
        iae = IterativeAmplitudeEstimation(state_preparation=state_preparation,
                                            epsilon=epsilon_iae, alpha=alpha_iae,
                                            objective_qubits=[len(qr_state)],
                                            post_processing=objective.post_processing)
        
        result = iae.run(quantum_instance=self.backend, shots=4000)

        conf_int = np.array(result['confidence_interval'])
        return conf_int, result['estimation']

    def build_cdf_state_preparation(self, comparator_value, uncertainty_model, weighted_adder):
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

    def get_quantum_var(self, losses):

        uncertainty_model = self.get_uncertainty_model()
        weighted_adder = self.get_weighted_adder()

        target_value = 1 - self.alpha
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

            cdf_state_pareparation = self.build_cdf_state_preparation(level, uncertainty_model, weighted_adder)
            iae_var = IterativeAmplitudeEstimation(state_preparation=cdf_state_pareparation, 
                                                    epsilon=0.01, 
                                                    alpha=0.05, 
                                                    objective_qubits = [uncertainty_model.num_qubits])
            value = iae_var.run(quantum_instance=self.backend, shots=1000)['estimation']

            if value >= target_value:
                high_level = level
                high_value = value
            else:
                low_level = level
                low_value = value

        return level, value

    def get_quantum_cvar(self, var, estimated_probability):

        uncertainty_model = self.get_uncertainty_model()
        weighted_adder = self.get_weighted_adder()

        breakpoints = [0, var]
        slopes = [0, 1]
        offsets = [0, 0]
        f_min = 0
        f_max = sum(self.loss_given_default) - var
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
        result_cvar = ae_cvar.run(quantum_instance=self.backend, shots=4000)

        d = (1.0 - estimated_probability)
        v = result_cvar['estimation'] / d if d != 0 else 0

        return v + var

    def montecarlo_sample(self, loans):
        total_loss_given_default = 0
        for i in range(len(loans)):
            total_loss_given_default = total_loss_given_default + random.choices([0, loans[i][2]], [1-loans[i][0], loans[i][0]])[0]
        return total_loss_given_default

    def classical_run(self, alpha, loans):

        total_value = 0
        expected_loss = 0
        for i in range(len(loans)):
            total_value += loans[i][2]
            expected_loss += loans[i][0] * loans[i][2]

        runs = 100000
        simulations = np.zeros(runs)
        for run in range(runs):
            simulations[run] = self.montecarlo_sample(loans)

        var_level = 100 - alpha * 100
        ordered_simulations = np.sort(simulations)
        var = np.percentile(ordered_simulations, var_level)
        cvar = simulations[ordered_simulations > var].mean()
    
        # Return VaR and CVaR
        return var, cvar+var



    def run(self, alpha, loans, n_z = 3, device="simulator", noise=True):
        
        provider = IBMQ.get_provider()
        if(device == "simulator"):
            self.device_backend = Aer.get_backend('qasm_simulator')
        else:
            self.device_backend = provider.get_backend('ibmq_qasm_simulator')
            #device_backend = provider.get_backend('ibmq_16_melbourne')

        if(noise):
            ibmq_16_melbourne = provider.get_backend('ibmq_16_melbourne')
            noise_model = NoiseModel.from_backend(ibmq_16_melbourne)
            self.backend = QuantumInstance(self.device_backend, noise_model=noise_model)
        else:
            self.backend = QuantumInstance(self.device_backend)

        # Mapping parameters
        # Loss Given Default multiplier (we can't map very big numbers, so we eliminate zeroes, from X00,000 -> X0)
        zeroes = len(str(int(min(np.array(loans)[:,2:3].tolist())[0])))-1
        lgd_factor = 10**zeroes
        
        self.n_z = n_z
        self.alpha = alpha
        self.loans = loans
        K = len(loans)

        for m in loans:
            self.probability_default.append(m[0])
            self.sensitivity_z.append(m[1])
            self.loss_given_default.append(int(m[2] / lgd_factor))   # LGD is simplified, reduced proportionately and taken only the integer part
        #print(self.loss_given_default)
        # sensitivity_z = np.zeros(K) # Remove Sensitivities for testing
        expected_loss, exact_var, exact_cvar, p_l_less_than_var, losses     = self.get_classical_expectation_loss()
        confidence_expected_loss, expected_loss_estimation                  = self.get_quantum_expected_loss()    
        estimated_var, estimated_var_probability                            = self.get_quantum_var(losses)
        estimated_cvar                                                      = self.get_quantum_cvar(estimated_var, estimated_var_probability)
        classical_var, classical_cvar                                       = self.classical_run(self.alpha, self.loans)


        print('-------------------------')
        print('-------------------------')
        print('-------------------------')
        print('LGD: ', self.loss_given_default, ' Total Assets value: $ {0:12,.0f}'.format(sum(self.loss_given_default)*lgd_factor))
        print('Assets: ', K)
        print('Assets default Probabilities: ', self.probability_default)
        print('Asset Sensitivities: ', self.sensitivity_z)
        print('-------------------------')
        print('Expected Loss E[L]:                $ {0:12,.0f}'.format(expected_loss * lgd_factor))
        print('Estimated Loss E[L]: $ {0:9,.0f}'.format(expected_loss_estimation * lgd_factor))
        print('Confidence interval: \t[%.0f, %.0f]' % (tuple(confidence_expected_loss * lgd_factor)))
        print('-------------------------')
        print('Value at Risk VaR[L](%.2f):        $ {0:12,.0f}'.format((exact_var*lgd_factor)) % (self.alpha))
        print('Estimated Value at Risk: $ {0:9,.0f}'.format(estimated_var * lgd_factor))
        error_var_estimation = 1-(exact_var) / estimated_var
        print('Error VaR Estimation: ', error_var_estimation)
        print('P[L <= VaR[L]](%.2f):              %.4f' % (self.alpha, p_l_less_than_var))
        print('Estimated P[L <= VaR[L]](%.2f):              %.3f' % (self.alpha, estimated_var_probability))
        print('-------------------------')
        print('Conditional Value at Risk CVaR[L]: $ {0:12,.0f}'.format(exact_cvar * lgd_factor))
        print('Estimated Conditional Value at Risk CVaR[L]: $ {0:12,.0f}'.format(estimated_cvar * lgd_factor))
        error_cvar_estimation = 1-(exact_cvar) / estimated_cvar
        print('Error CVaR Estimation: ', error_cvar_estimation)
        print('-------------------------')
        print('Montecarlo Value at Risk VaR[L](%.2f):        $ {0:12,.0f}'.format(classical_var) % (self.alpha))
        print('Montecarlo Expected Shortfall CVaR[L]: $ {0:12,.0f}'.format(classical_cvar))