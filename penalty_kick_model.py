import numpy as np

def force_length_muscle(lm):
    """
    Compute the force-length scale factor for a muscle based on its length.

    Args:
    - lm: muscle (contractile element) length

    Returns:
    - force_length_scale_factor: the force-length scale factor
    """
    global force_length_regression
    if len(np.array(lm).shape) <= 0:
      lm = np.array([lm])
    force_length_scale_factor = force_length_regression(lm)
    return force_length_scale_factor

def force_length_parallel(lm):
    """
    Compute the normalized force produced by the parallel elastic element of a muscle
    based on its normalized length.

    Args:
    - lm: normalized length of muscle (contractile element)

    Returns:
    - normalize_PE_force: normalized force produced by parallel elastic element
    """
    if len(np.array(lm).shape) <= 0:
      lm = np.array([lm])
    normalize_PE_force = np.zeros(lm.shape)

    for i in range(len(lm)):
        if lm[i] < 1:
            normalize_PE_force[i] = 0
        else:
            normalize_PE_force[i] = 3*((lm[i] - 1)**2)/(0.6 + lm[i] - 1)

    return normalize_PE_force


def force_length_tendon(lt):
    """
    Compute the normalized tension produced by the tendon (series elastic element)
    based on its normalized length.

    Args:
    - lt: normalized length of tendon (series elastic element)

    Returns:
    - normalize_tendon_tension: normalized tension produced by tendon
    """
    if len(np.array(lt).shape) <= 0:
      lt = np.array([lt])

    normalize_tendon_tension = np.zeros(lt.shape)

    for i in range(len(lt)):
        if lt[i] < 1:
            normalize_tendon_tension[i] = 0
        else:
            normalize_tendon_tension[i] = 10*(lt[i]-1) + 240*((lt[i]-1)**2)

    return normalize_tendon_tension


def force_velocity_muscle(vm):
    """
    Compute the force-velocity scale factor for a muscle based on its velocity.

    Args:
    - vm: muscle (contractile element) velocity

    Returns:
    - force_velocity_scale_factor: the force-velocity scale factor
    """
    vm = np.array([vm])
    if vm.shape[1] > vm.shape[0]:
        vm = vm.T
    global force_velocity_regression
    force_velocity_scale_factor = model_eval('Sigmoid', vm, force_velocity_regression)
    return force_velocity_scale_factor


def model_eval(function_type, input, ridge_coeff):

    if function_type == 'Sigmoid':
        fun = lambda x, mu, sigma: 1 / (1 + np.exp(-(x-mu) / sigma))
        X = np.array([fun(input, i, 0.15) for i in np.arange(-1, -0.09, 0.2)])

    elif function_type == 'Gaussian':
        # Add code for Gaussian function here if needed
        pass

    X = np.reshape(X, ridge_coeff[1:].shape)

    Output = ridge_coeff[0] + np.dot(X, ridge_coeff[1:])

    return Output

import numpy as np
from scipy.interpolate import interp1d, make_smoothing_spline
from scipy.optimize import curve_fit

def get_muscle_force_length_regression():
    # Define data points
    datax = np.array([37.415144, 38.511749, 39.373368, 39.451697, 40.391645, 40.469974, 40.626632, 41.409922, 41.488251, 41.488252, 41.488253, 41.879896, 42.036554, 42.584856, 42.819843, 42.898172, 42.898173, 43.211488, 43.289817, 43.524804, 43.603133, 43.603134, 43.759791, 43.759792, 44.308094, 44.464752, 45.169713, 45.483029, 45.561358, 45.639687, 45.639688, 45.718016, 45.874674, 46.344648, 46.344649, 46.501305, 46.657963, 46.736292, 47.049608, 47.362924, 47.441253, 47.597911, 47.597912, 47.67624, 48.224543, 48.929504, 49.007833, 49.007834, 49.007835, 49.477807, 49.712794, 49.712795, 49.791123, 50.182768, 50.652742, 50.652743, 50.652744, 50.73107, 50.887728, 50.966057, 51.122715, 51.436031, 51.749347, 52.610966, 53.394256, 53.472585, 53.550914, 53.629243, 53.629244, 53.707572, 53.86423, 53.86424, 54.255875, 54.334204, 54.725849, 55.822454, 56.214099, 56.449086, 56.840731, 57.075718, 57.075719, 57.310705, 57.545692, 57.780679, 57.859008, 58.407311, 58.563969, 58.955614, 59.347258, 59.425587, 59.660574, 59.817232, 60.052219, 60.522193, 60.600522, 61.227154, 61.305483, 61.383812, 61.383813, 61.383814, 61.462141, 62.167102, 62.32376, 62.637076, 63.10705, 63.342037, 63.420366, 63.420367, 63.498695, 63.655352, 63.733681, 63.81201, 63.890339, 63.890340, 64.438642, 64.438643, 64.751958, 64.830287, 65.065274, 65.37859, 65.613577, 65.691906, 65.770235, 65.770236, 66.083551, 66.396867, 66.94517, 67.023499, 67.180157, 67.415144, 67.571802, 67.806789, 68.355091, 68.511749, 69.451697, 70.313316, 70.313317, 70.469974, 71.409922, 72.428198, 72.976501, 73.368146, 73.368147, 73.368148, 73.446475, 74.386423, 75.169713, 75.4047, 76.344648 ])
    datay = np.array([9.8, 14.6, 23.9, 3.5, 21.8, 17.6, 36.8, 26.6, 14.6, 15.8, 31.4, 2, 32, 42.5, 23.6, 46.4, 48.5, 50.3, 23.9, 54.2, 35, 44.9, 22.4, 57.2, 45.8, 60.5, 54.2, 46.1, 53.6, 46.4, 43.7, 67.7, 70.4, 62.9, 73.7, 71.6, 44.6, 75.5, 80.9, 62.9, 71.6, 81.5, 81.5, 66.8, 83.3, 82.1, 62.9, 81.2, 85.7, 76.1, 82.1, 86.9, 85.1, 87.8, 74.3, 84.8, 90.5, 87.2, 78.2, 80, 78.5, 89.9, 91.1, 89, 88.7, 78.5, 95, 96.8, 83.3, 92.3, 92.3, 96.8, 93.8, 94.7, 99.5, 96.2, 96.2, 100.1, 99.8, 99.2, 99.8, 98, 99.5, 99.2, 91.7, 90.8, 96.5, 99.2, 97.7, 91.4, 95.9, 96.8, 95.6, 99.8, 93.8, 84.5, 92, 95, 87.8, 77.3, 79.7, 89.6, 96.8, 80, 86.3, 59.9, 81.2, 79.7, 53, 85.7, 89.6, 86.9, 80.6, 76.1, 87.2, 81.8, 53.6, 52.4, 76.1, 64.1, 72.8, 68, 66.2, 47.6, 72.2, 76.1, 66.2, 42.8, 63.8, 35.6, 51.8, 62.9, 59.9, 27.2, 41.6, 29.6, 29.6, 48.8, 34.7, 24.5, 25.7, 34.7, 18.8, 17.6, 12.5, 12.5, 17.6, 12.5, 8.6])

    # Normalize
    Maxy = max(datay)
    index = np.argmax(datay)
    Maxx = datax[index]

    normx = datax / Maxx
    normy = datay / Maxy

    def gauss_function(x, a, x0, sigma):
      return a*np.exp(-(x-x0)**2/(2*sigma**2))

    popt, pcov = curve_fit(gauss_function, normx, normy)

    def f(x):
      return gauss_function(x, popt[0], popt[1], popt[2])

    return f


import numpy as np
from sklearn.linear_model import Ridge

def get_muscle_force_velocity_regression():
    # Input Parameters
    # data[:,0]: samples of an independent variable
    # data[:,1]: corresponding samples of a dependent variable
    data = np.array([
        [-1.0028395556708567, 0.0024834319945283845],
        [-0.8858611825192801, 0.03218792009622429],
        [-0.5176245843258415, 0.15771090304473967],
        [-0.5232565269687035, 0.16930496922242444],
        [-0.29749770052593094, 0.2899790099290114],
        [-0.2828848376217543, 0.3545364496120378],
        [-0.1801231103040022, 0.3892195938775034],
        [-0.08494610976156225, 0.5927831890757294],
        [-0.10185137142991896, 0.6259097662790973],
        [-0.0326643239546236, 0.7682365981934388],
        [-0.020787245583830716, 0.8526638522676352],
        [0.0028442725407418212, 0.9999952831301149],
        [0.014617579774061973, 1.0662107025777694],
        [0.04058866536166583, 1.124136223202283],
        [0.026390887007381902, 1.132426122025424],
        [0.021070257776939272, 1.1986556920827338],
        [0.05844673474682183, 1.2582274002971627],
        [0.09900238201929201, 1.3757434966156459],
        [0.1020023112662436, 1.4022310794556732],
        [0.10055894908138963, 1.1489210160137733],
        [0.1946227683309354, 1.1571212943090965],
        [0.3313459588217258, 1.152041225442796],
        [0.5510200231126625, 1.204839508502158]
    ])

    velocity = data[:, 0]
    force = data[:, 1]

    # Ridge Regression
    # NOTE: this is different than TASK 2 of Question 1, follow the instruction for TASK 2
    def fun(x, mu, sigma):
        return 1 / (1 + np.exp(-(x-mu) / sigma))
    X = np.zeros((len(velocity), 5))
    for i, j in enumerate(np.arange(-1, -0.01, 0.2)):
        X[:, i] = fun(velocity, j, 0.15)
    ridge_model = Ridge(alpha=1, fit_intercept=False).fit(X, force)
    coefs = ridge_model.coef_
    intercept = np.array([ridge_model.intercept_])
    
    return np.concatenate([intercept, coefs])

import scipy.optimize

def get_velocity(a, lm, lt):
    # damping coefficient (see damped model in Millard et al.)
    beta = 0.1

    # define the function to find the root of
    def fun(vm):
        return 1*(a*force_length_muscle(lm)*force_velocity_muscle(vm) + force_length_parallel(lm)+ beta*vm)-force_length_tendon(lt)

    vm = 0
    #print(lm)
    #print(lt)
    root = scipy.optimize.fsolve(fun, vm)

    return root

def gravity_moment(theta):
    # Inputs
    # theta: angle of body segment (up from prone)

    # Output
    # moment about ankle due to force of gravity on body

    mass = 75  # body mass (kg; excluding feet)
    centre_of_mass_distance = 1  # distance from ankle to body segment centre of mass (m)
    g = 9.81  # acceleration of gravity
    moment = mass * g * centre_of_mass_distance * np.cos(theta)
    return moment

class HillTypeMuscle:
    """
    Damped Hill-type muscle model adapted from Millard et al. (2013). The
    dynamic model is defined in terms of normalized length and velocity.
    To model a particular muscle, scale factors are needed for force, CE
    length, and SE length.
    """
    
    def __init__(self, f0M, resting_length_muscle, resting_length_tendon):
        self.resting_length_muscle = resting_length_muscle
        self.resting_length_tendon = resting_length_tendon
        self.f0M = f0M
        
    def norm_tendon_length(self, muscle_tendon_length, normalized_muscle_length):
        """
        Calculate the normalized length of the tendon.
        """
        return (muscle_tendon_length - self.resting_length_muscle * normalized_muscle_length) / self.resting_length_tendon
    
    def get_force(self, total_length, norm_muscle_length):
        """
        Calculate muscle tension (N).
        """
        normalized_tendon_length = self.norm_tendon_length(total_length, norm_muscle_length)
        return self.f0M * force_length_tendon(normalized_tendon_length)



import numpy as np

def soleus_length(theta):
    """
    Inputs:
    - theta: body angle (up from prone horizontal)
    
    Output:
    - soleus length
    """
    # define rotation matrix
    rotation = np.array([[np.cos(theta), -np.sin(theta)], 
                         [np.sin(theta), np.cos(theta)]])
    
    # coordinates in global reference frame
    origin = np.dot(rotation, np.array([0.3, 0.03]).T)
    insertion = np.array([-0.05, -0.02])
    
    difference = origin - insertion
    soleus_length = np.sqrt(difference[0]**2 + difference[1]**2)
    
    return soleus_length


import numpy as np

def tibialis_length(theta):
    # Inputs
    # theta: body angle (up from prone horizontal)

    # Output
    # tibialis anterior length

    # define rotation matrix
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # coordinates in global reference frame
    origin = rotation @ np.array([0.3, -0.03]).reshape((2,1))
    insertion = np.array([0.06, -0.03]).reshape((2,1))

    difference = origin - insertion
    tibialis_anterior_length = np.sqrt(difference[0]**2 + difference[1]**2)

    return tibialis_anterior_length[0]

def dynamics(x, soleus, tibialis):
    # Inputs
    #  x: state vector (ankle angle, angular velocity, soleus normalized CE length, TA normalized CE length)
    # soleus: soleus muscle (HillTypeModel)
    # tibialis: tibialis anterior muscle (HillTypeModel)
    # Output
    # x_dot: derivative of state vector

    soleus_activation = 0.05
    tibialis_activation = 0.4
    ankle_inertia = 90

    # we first obtain the normalized tendon lengths for both muscles
    # they depend on normalized muscle length and musculotendon length

    lt_sol = soleus.norm_tendon_length(soleus_length(x[0]), x[2])
    lt_ta = tibialis.norm_tendon_length(tibialis_length(x[0]), x[3])

    # we calculate torque caused by each muscle as well as gravity
    # note that f_ext is 0 so that term is not present (0'd out)

    torque_sol = 16000*force_length_tendon(lt_sol)*0.05
    torque_ta = 2000*force_length_tendon(lt_ta)*0.03
    torque_mg = 75*9.81*1*np.cos(x[0])

    # once we have torques, x_dot[1] is easy to calculate and the rest are
    # implemented in accordance to what is present in the lectures

    x_dot = [0, 0, 0, 0]
    x_dot[0] = x[1]
    x_dot[1] = (torque_sol-torque_ta-torque_mg)/ankle_inertia
    x_dot[2] = get_velocity(soleus_activation, x[2], lt_sol)
    x_dot[3] = get_velocity(tibialis_activation, x[3], lt_ta)

    return x_dot

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def simulate(T):
    # Runs a simulation of the model and plots results.

    # Inputs
    # T: total time to simulate, in seconds

    rest_length_soleus = soleus_length(math.pi/2)
    rest_length_tibialis = tibialis_length(math.pi/2)

    soleus = HillTypeMuscle(16000, 0.6*rest_length_soleus, 0.4*rest_length_soleus)
    tibialis = HillTypeMuscle(2000, 0.6*rest_length_tibialis, 0.4*rest_length_tibialis)

    def f(x, t):
        return dynamics(x, soleus, tibialis)

    tspan = [0, T]
    time = np.linspace(tspan[0], tspan[-1], 10000)
    initialCondition = [math.pi/2, 0, 1, 1]
    y = odeint(f, initialCondition, time, full_output=False)

    theta = y[:,0]
    soleus_norm_length_muscle = y[:,2]
    tibialis_norm_length_muscle = y[:,3]

    soleus_moment_arm = 0.05
    tibialis_moment_arm = 0.03
    soleus_moment = np.zeros(y.shape[0])
    tibialis_moment = np.zeros(y.shape[0])
    for i in range(y.shape[0]):
        soleus_moment[i] = soleus_moment_arm * soleus.get_force(soleus_length(theta[i]), soleus_norm_length_muscle[i])
        tibialis_moment[i] = -tibialis_moment_arm * tibialis.get_force(tibialis_length(theta[i]), tibialis_norm_length_muscle[i])

    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    axs[0].plot(time, theta, linewidth=1.5)
    axs[0].set_ylabel('Body Angle (rad)')

    axs[1].plot(time, soleus_moment, 'r', linewidth=1.5, label='Soleus')
    axs[1].plot(time, tibialis_moment, 'g', linewidth=1.5, label='Tibialis')
    axs[1].plot(time, gravity_moment(theta), 'k', linewidth=1.5, label='Gravity')
    axs[1].legend(loc='upper left')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Torques (Nm)')

    plt.show()

def plot_curves():
    # define the ranges for lm, vm and lt
    lm = np.arange(0, 1.81, 0.01)
    vm = np.arange(-1.2, 1.21, 0.01)
    lt = np.arange(0, 1.08, 0.01)

    # set the line width and font size
    LineWidth = 1.5
    FontSize = 12

    # create the subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # plot the force-length curves
    ax1.plot(lm, force_length_muscle(lm), 'r', linewidth=LineWidth)
    ax1.plot(lm, force_length_parallel(lm), 'g', linewidth=LineWidth)
    ax1.plot(lt, force_length_tendon(lt), 'b', linewidth=LineWidth)
    ax1.set_xlim([-0.1, 1.85])
    ax1.set_ylim([0, 1.6])
    ax1.legend(['CE', 'PE', 'SE'], loc='upper left', fontsize=FontSize)
    ax1.set_xlabel('Normalized length', fontsize=FontSize)
    ax1.set_ylabel('Force scale factor', fontsize=FontSize)
    ax1.tick_params(axis='both', which='major', labelsize=FontSize)

    # plot the force-velocity curve
    ax2.plot(vm, force_velocity_muscle(vm), 'k', linewidth=LineWidth)
    ax2.set_xlim([-1.3, 1.3])
    ax2.set_ylim([0, 1.35])
    ax2.set_xlabel('Normalized muscle velocity', fontsize=FontSize)
    ax2.set_ylabel('Force scale factor', fontsize=FontSize)
    ax2.tick_params(axis='both', which='major', labelsize=FontSize)

    # adjust the layout of the subplots
    plt.subplots_adjust(hspace=0.3)

    # display the plot
    plt.show()

