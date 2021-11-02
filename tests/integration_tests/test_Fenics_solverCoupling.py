# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports (problem definition)
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.noise_model import NormalNoiseModel
from tests.integration_tests.Uniaxial_tension_solver import FenicsYoungsModulusTestSimulation
from probeye.inference.torch_.solver import PyroSolver
from probeye.postprocessing.sampling import create_pair_plot,create_trace_plot
from probeye.inference.torch_.misc import dataloader

path_exp1 = '../../../BayesianInference/usecases/Concrete/Data/E-modul/Wolf 8.2 Probe 1/specimen.dat'
dia_exp1 = 98.6
height_exp1 = 300.3
# idx_s_1 = 330
# idx_e_1 = 145
idx_s_1 = 300
idx_e_1 = 220

data_exp1 = dataloader(path_exp1,dia_exp1,height_exp1,idx_s_1,idx_e_1)

# Setting up numerical values
low_E = 90
high_E = 110


# sigma prior parameters
low_sigma = 0
high_sigma = 0.005

forward_solve = FenicsYoungsModulusTestSimulation()

print("testing the forward solver")
input = {'E' : 98,
         'nu' : 0.2,
         'height' : 300.3,
         'diameter' : 98.6,
         'strain':data_exp1['strain']
         }

# run the model
model_answer = forward_solve(input)
plt.plot(data_exp1['strain'],model_answer['stress'], 'b', label = 'ForwardSolver')
plt.plot(data_exp1['strain'], data_exp1['stress'], 'r', label='Exp #1')
plt.xlabel('strain')
plt.ylabel('stress')
plt.legend()
plt.title('Experimental Values for last load cycle')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.show()


class LinearModel(ForwardModelBase):
    def response(self, inp):
        input = {'E': inp['E'],
                 'nu': 0.2,
                 'height': 300.3,
                 'diameter': 98.6,
                 'strain': inp['strain']
                 }
        solver_response = {}

        for os in self.output_sensors:
            solver_response[os.name] = np.array(forward_solve(input)['stress'])
        return solver_response


problem = InferenceProblem("Youngs Modulus Calibration")

problem.add_parameter('E', 'model',
                      tex="$E$",
                      info="Slope of the graph",
                      prior=('uniform', {'low': low_E,
                                        'high': high_E}))

problem.add_parameter('sigma', 'noise',
                      tex=r"$\sigma$",
                      info="Std. dev, of 0-mean noise model",
                      prior=('uniform', {'low': low_sigma,
                                         'high': high_sigma}))


isensor = Sensor("strain")
osensor = Sensor("stress")
linear_model = LinearModel(['E'], [isensor], [osensor])
problem.add_forward_model("LinearModel1", linear_model)
# add the noise model to the problem

problem.add_noise_model(NormalNoiseModel(prms_def={'sigma': 'std'}, sensors=osensor))

problem.add_experiment(f'TestSeries_1', fwd_model_name="LinearModel1",
                       sensor_values={isensor.name: data_exp1['strain'],
                                      osensor.name: data_exp1['stress']})
problem.info()

# -- Solve inference problem
inference_solver = PyroSolver(problem)

pos = inference_solver.run_mcmc(n_steps=30,n_initial_steps=10)

mcmc = inference_solver.raw_results
mcmc.summary()

# -- Visualisation
create_trace_plot(pos, problem)
create_pair_plot(pos,problem,focus_on_posterior=True)
plt.show()