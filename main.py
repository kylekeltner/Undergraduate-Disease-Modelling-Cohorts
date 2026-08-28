#IMPORTS
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import callables
import scipy as sp
from scipy.integrate import odeint
import numpy as np
from scipy.stats import qmc
import sympy as sp

'''
    DATA POINTS FOR FITTING
'''
x_coords = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120])
y_coords = np.array([40000, 44000, 48000, 52500, 50000, 54500, 56750, 58000, 60000])

'''
    ALL EQUATION PARAMETERS
    lambda_p: colon cell population
    lambda_c: cancer cell population
    lambda_i: immune cell population
    Beta: rate of carcinogenesis scaled by immune cells
    alpha_c: rate of immune cell activation by cancer cells
    alpha_i: immune cell impact on death of cancer cells
    S_pc: death rate of cancer cells
    S_pn: death rate of normal cells
    S_i: death rate of immune cells
    r: rate that normal cells exit
'''

lambda_p = 5
lambda_c = 2 # gathered from averages in Data Driven Mathematical Model of Colon Cancer Progression, C bar_0
lambda_i = 0.5
Beta = 2e-8
alpha_c = .5
alpha_i = .2
S_pc = 1.03e-3 #gathered from averages in Data Driven Mathematical Model of Colon Cancer Progression, avg of delta_c
S_pn = 6.52e-3 #gathered from averages in Data Driven Mathematical Model of Colon Cancer Progression, avg of delta_n
S_i = 5e-2 
r = 1.8

'''
    ALL R_0 PARAMETERS
    u_c: Truthfully, I do not know what it is
'''

u_c = 1.8

'''
    DYNHAMIC COUNTERPARTS FOR EQUATION PARAMETERS
    lambda_p: colon cell population
    lambda_c: cancer cell population
    lambda_i: immune cell population
    Beta: rate of carcinogenesis scaled by immune cells
    alpha_c: rate of immune cell activation by cancer cells
    alpha_i: immune cell impact on death of cancer cells
    S_pc: death rate of cancer cells
    S_pn: death rate of normal cells
    S_i: death rate of immune cells
    r: rate that normal cells exit
'''

dyn_lambda_p = 5
dyn_lambda_c = 2
dyn_lambda_i = 0.5
dyn_Beta = 2e-8
dyn_alpha_c = .5
dyn_alpha_i = .2
dyn_S_pc = 1.03e-3
dyn_S_pn = 6.52e-3
dyn_S_i = 5e-2 
dyn_r = 1.8

'''
    MODEL FUNCTIONS
    Passes state vector x of type list 
    Passes time t of type float
'''
def odes(x: list, t: float) -> list: 
    for i in range( len( x ) ):
        print(x[i])
    P_n = x[0]
    P_c = x[1]
    I   = x[2]
    D_c = x[3]
    
    '''
        ODE's for proliferative, dead, and immune cells
    '''

    dP_ndt = lambda_p - ( Beta * I * P_n ) - ( S_pn * P_n )
    dP_cdt = ( lambda_c * P_c ) + ( Beta * I * P_n ) - ( alpha_i * I * P_c ) - ( S_pc * P_c )
    dIdt   = lambda_i + ( alpha_c * P_c ) - ( S_i * I )
    dD_cdt = ( S_pc * P_c ) + ( alpha_i * I * P_c ) - ( r * I * D_c )
    

    return [dP_ndt, dP_cdt, dIdt, dD_cdt]

'''
    POSSIBLE NEW PARAMETERS/ODE CONFIGURATIONS
'''

'''
    DEFINE INITIAL CONDITIONS FOR POPULATIONS
    x_0 = [P_n, P_c, I, D_c]
'''

x_0 = [1.75e5, 5.3e-3, 10, 0] 

t = np.linspace(0,120,1000)
x = odeint(odes, x_0, t) #how does this pass x_0 into odes?? <------------------!!!! WHAAAAAAAAAAAAAAAAAAAAAAAA

P_n = x[:,0]
P_c = x[:,1]
I   = x[:,2]
D_c = x[:,3]

#Define Cancer
C = P_c + D_c

fig_plot, axs_plot = plt.subplots(2 , 2 , figsize = (10 , 7) )
fig_sliders = plt.figure( figsize = ( 10 , 5 ) )
fig_overlay = plt.subplots( figsize = ( 7 , 7 ) )


'''
    PLOTS FOR 4 PARAMETERS, TOTAL CANCER CELLS NOT INCLUDED YET
'''

plots = [
    (axs_plot[0, 0], P_n, 'cyan', '$P_n(t)$', 'Proliferative Normal Cells'),
    (axs_plot[0, 1], P_c, 'olive', '$P_c(t)$', 'Proliferative Cancer Cells'),
    (axs_plot[1, 0], D_c, 'blue', '$D_c(t)$', 'Dead Cancer Cells'),
    (axs_plot[1, 1], I,   'purple', '$I(t)$', 'Immune Cells'),
]

for ax, data, color, title, ylabel in plots:
    ax.plot(t, data, color = color)
    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xlabel('$t$')
    ax.set_ylabel(ylabel)
    ax.grid()
    fig_overlay[1].plot(t, data, label = title, color = color)

'''
    SLIDERS FOR COEFFICIENT ADJUSTMENT
'''

beta_slider = Slider(fig_sliders.add_axes([0.08, 0.9, 0.35, 0.03]), label = 'Beta', valmin = 0, valmax = 1e-7, valstep = 1e-9, valinit = Beta )
lambda_p_slider = Slider(fig_sliders.add_axes([0.08, 0.8, 0.35, 0.03]), label = 'lambda_p', valmin = 1, valmax = 15, valstep = 0.1, valinit = lambda_p )
lambda_c_slider = Slider(fig_sliders.add_axes([0.08, 0.7, 0.35, 0.03]), label = 'lambda_c', valmin = 0, valmax = 15, valstep = 0.1, valinit = lambda_c )
lambda_i_slider = Slider(fig_sliders.add_axes([0.08, 0.6, 0.35, 0.03]), label = 'lambda_i', valmin = 0, valmax = 10, valstep = 0.01, valinit = lambda_i )
alpha_c_slider = Slider(fig_sliders.add_axes([0.08, 0.5, 0.35, 0.03]), label = 'alpha_c', valmin = 0, valmax = 3, valstep = 0.01, valinit = alpha_c )
alpha_i_slider = Slider(fig_sliders.add_axes([0.56, 0.9, 0.35, 0.03]), label = 'alpha_i', valmin = 0, valmax = 3, valstep = 0.01, valinit = alpha_i )
S_pc_slider = Slider(fig_sliders.add_axes([0.56, 0.8, 0.35, 0.03]), label = 'S_pc', valmin = 0, valmax = 0.1, valstep = 1e-5, valinit = S_pc )
S_pn_slider = Slider(fig_sliders.add_axes([0.56, 0.7, 0.35, 0.03]), label = 'S_pn', valmin = 0, valmax = 1e-2, valstep = 1e-4, valinit = S_pn )
S_i_slider = Slider(fig_sliders.add_axes([0.56, 0.6, 0.35, 0.03]), label = 'S_i', valmin = 0, valmax = 1, valstep = 1e-3, valinit = S_i )
r_slider = Slider(fig_sliders.add_axes([0.56, 0.5, 0.35, 0.03]), label = 'r', valmin = 0, valmax = 5, valstep = 0.1, valinit = r )

sliders = [ beta_slider, lambda_p_slider, lambda_c_slider, lambda_i_slider, alpha_c_slider, alpha_i_slider, S_pc_slider, S_pn_slider, S_i_slider, r_slider ]

button = Button( fig_sliders.add_axes([0.45, 0.05, 0.1, 0.04]), 'Reset', color = 'lightgoldenrodyellow', hovercolor = '0.975' )

'''
    UPDATE FUNCTION FOR SLIDERS
    Passes value of type float from slider
    Updates ODEs, redraws plots
'''

def update_from_slider_value(val: float) -> None:

    dP_ndt = lambda_p_slider.val - ( beta_slider.val * I * P_n ) - ( S_pn_slider.val * P_n )
    dP_cdt = ( lambda_c_slider.val * P_c ) + ( beta_slider.val * I * P_n ) - ( alpha_i_slider.val * I * P_c ) - ( S_pc_slider.val * P_c )
    dIdt   = lambda_i_slider.val + ( alpha_c_slider.val * P_c ) - ( S_i_slider.val * I )
    dD_cdt = ( S_pc_slider.val * P_c ) + ( alpha_i_slider.val * I * P_c ) - ( r_slider.val * I * D_c )

    axs_plot[0, 0].lines[0].set_ydata(dP_ndt)
    axs_plot[0, 1].lines[0].set_ydata(dP_cdt)
    axs_plot[1, 0].lines[0].set_ydata(dD_cdt)
    axs_plot[1, 1].lines[0].set_ydata(dIdt)
    fig_overlay[1].lines[0].set_ydata(dP_ndt)
    fig_overlay[1].lines[1].set_ydata(dP_cdt) 
    fig_overlay[1].lines[2].set_ydata(dD_cdt)
    fig_overlay[1].lines[3].set_ydata(dIdt)

    fig_plot.canvas.draw_idle( )
    fig_overlay[0].canvas.draw_idle() #Not sure why this fig_overlay draw uses 0 idx instead of 1, doesn't work otherwise

'''
    UPDATE FUNCTION FOR REPRODUCTIVE NUMBER
    Passes value of type float from sliders
    Updates r_0
'''

def update_reproductive_number(val: float) -> None:
    r_0 = callables.reproductive_number( u_c, alpha_c_slider.val, beta_slider.val, lambda_p_slider.val, S_pn_slider.val, S_pc_slider.val, lambda_i_slider.val, S_i_slider.val )
    print(f"Updated reproductive number (R_0): {r_0}")
    return r_0

for i in sliders:
    i.on_changed( update_from_slider_value )
    i.on_changed( update_reproductive_number )

'''
    RESET FUNCTION FOR SLIDERS
    resets sliders to intial values
'''

def reset_sliders( val: float ) -> None:
    for i in sliders:
        i.reset( )

sensitivities = callables.r_0_sensitivity_analysis( [ lambda_p, lambda_c, lambda_i, Beta, alpha_c, alpha_i, S_pc, S_pn, S_i, r ], 50 ) 
print( sensitivities )

button.on_clicked( reset_sliders )
plt.tight_layout( )
plt.show( )



dict1 = { buh: 12, swag: 9 }
list1=[]
for kw, arg in dict1.items():
    list1.append(kw)
print(list1)