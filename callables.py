import sympy as sp
import math

'''
    FUNCTION FOR CALCULATING R_0
    Takes intput of type float from a set of dynamic parameters
    Returns value of type float for R_0
'''

def reproductive_number( d_lambda_c: float, d_alpha_i: float, d_alpha_c: float, d_Beta: float, d_lambda_p: float, d_S_pn: float, d_S_pc: float, d_lambda_i: float, d_S_i: float ) -> float:
    r_0 = ( d_lambda_c + ( d_Beta * ( d_lambda_i / d_S_i ) * ( d_lambda_p / ( d_Beta * ( d_lambda_i / d_S_i ) ) ) ) ) / ( ( d_alpha_i * ( d_lambda_i / d_S_i ) ) + d_S_pc )
    return r_0

'''
    FUNCTION FOR CALCULATING PARAMETER SENSITIVITIES TO R_0
    Takes intput of type list of dynamic parameters and a float for R_0
    Returns value of type dict for sensitivities of each parameter to R_0
'''

def r_0_sensitivity_analysis( dyn_params_names: list, dyn_params: list, r_0: float ) -> dict:
    symbol_list = [ ]
    for i in dyn_params_names:
        symbol_list.append( sp.Symbol( str( i ) ) )
    r_0_symbol = ( sp.Symbol( 'd_lambda_c' ) + ( sp.Symbol( 'd_Beta' ) * ( sp.Symbol( 'd_lambda_i' ) / sp.Symbol( 'd_S_i' ) ) * ( sp.Symbol( 'd_lambda_p' ) / ( sp.Symbol( 'd_Beta' ) * ( sp.Symbol( 'd_lambda_i' ) / sp.Symbol( 'd_S_i' ) ) ) ) ) ) / ( ( sp.Symbol( 'd_alpha_i' ) * ( sp.Symbol( 'd_lambda_i' ) / sp.Symbol( 'd_S_i' ) ) ) + sp.Symbol( 'd_S_pc' ) )

    sensitivities = { }
    for i in range( len( symbol_list ) ):
        sensitivities[ dyn_params_names[i] ] = ( ( 1 / r_0 ) * ( sp.diff( r_0_symbol, symbol_list[i] ) ).subs( { symbol_list[j]: dyn_params[j] for j in range( len( symbol_list ) ) } ) )
    return sensitivities


#( lambda_c + ( Beta * ( lambda_i / S_i ) * ( lambda_p / ( Beta * ( lambda_i / S_i ) ) ) ) ) / ( ( alpha_i * ( lambda_i / S_i ) ) + S_pc )


#r_0 = ( u_c * ( 1 - alpha_c ) + Beta * ( lambda_p / S_pn ) ) / ( ( S_pc * ( 1 + alpha_c ) )  * ( lambda_i / S_i ) )

"""
NON-SLIDER VALUES FOR R_0

alpha_c = 0    #for future...keep curcumin at a range of 0.1 to 0.2

Beta = 0.000124   #mins: 0.00012 - 0.000124

# Deltas
S_c = .9 #0.9      #min: 0.9    #hrs: 0.015 
S_p = 9e-6 #9e-6     #min: 9e-6   #hrs: 0.00399995 
S_i = 2 #2   #min: 3.162   #hrs: 0.0527 #can't be anything other than 2 

lambda_p = 18 #18       #min: 18       #hrs: 0.3
lambda_i = .3 #0.3      #min: 0.3      #hrs: 0.005 #can't be more than .4

# mu 
u_c = 1.8 #1.8      #min: 1.8    #hrs: 0.03 

r = 1.8 #1.8             #min: 1.8     #hrs: 0.03 #0 is important 

#reproductive number
r_0 = (u_c * (1 - alpha_c) + Beta * (lambda_p / S_p)) / ((S_c * (1 + alpha_c)) * (lambda_i / S_i)) 

"""


#lambda_c, lambda_p, lambda_i, alpha_i, S_i, S_pc = sp.symbols('lambda_c, lambda_p, lambda_i, alpha_i, S_i, S_pc')

#R_0 = (lambda_p + lambda_c)/(alpha_i(lambda_i/S_i)+ S_pc)

#R_e = R_0(P_n/(P_n + P_c))

#S_lambda_c = (lambda_c/R_0)(sp.diff(R_0,lambda_c))

#S_lambda_p = (lambda_p/R_0)(sp.diff(R_0,lambda_p))
#lambda_p/(lambda_c + lambda_p)

#S_alpha_i = (alpha_i/R_0)(diff(R_0,alpha_i))
#-(alpha_i(lambda_i/S_i))/(alpha_i(lambda_i/S_i)+ S_pc)

#S_lambda_i = (lambda_i/R_0)(sp.diff(R_0,lambda_i))
#-(alpha_i(lambda_i/S_i))/(alpha_i(lambda_i/S_i)+ S_pc)

#S_S_i = (S_i/R_0)(sp.diff(R_0,S_i))
#(S_i(alpha_i(lambda_i/(S_i)^2)))/(alpha_i(lambda_i/S_i)+ S_pc)

#S_S_pc = (S_pc/R_0)(sp.diff(R_0,S_pc))
#-S_pc/(alpha_i(lambda_i/S_i)+ S_pc)


#Access individual sensitivities

#for var, S in sensitivities.items():
    #print(f"S{var} = {sp.simplify(S)}")

