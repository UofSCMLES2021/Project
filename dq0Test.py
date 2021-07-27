# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 21:34:38 2021

@author: jackh
"""
import  ClarkePark 
import  numpy  as  np 
import  matplotlib.pyplot  as  plt

end_time = 3 / float(60)
step_size = end_time / (1000)
delta = 0
t = np . arange(0, end_time, step_size)
wt = 2 * np . pi * float(60) * t

rad_angA = float(0) * np . pi / 180
rad_angB = float(240) * np . pi / 180
rad_angC = float(120) * np . pi / 180

A = (np.sqrt(2) * float(127)) * np.sin(wt + rad_angA)
B = (np.sqrt(2) * float(127)) * np.sin(wt + rad_angB)
C = (np.sqrt(2) * float(127)) * np.sin(wt + rad_angC)

d,  q,  z = ClarkePark . abc_to_dq0(A,  B,  C,  wt,  delta)
a,  b,  c = ClarkePark . dq0_to_abc(d,  q,  z,  wt,  delta)

plt.figure(figsize = (8 , 3)) 
plt.plot(t , a , label = "A" , color = "royalblue" ) 
plt.plot(t , b , label = "B" , color = "orangered" ) 
plt.plot(t , c , label = "C" , color = "forestgreen" ) 
plt.legend([ 'A' , 'B' , 'C' ]) 
plt.legend( ncol = 3 , loc = 4 ) 
plt.ylabel( "Voltage [Volts]" ) 
plt.xlabel( "Time [Seconds]" ) 
plt.title( "Three-phase ABC system" ) 
plt.grid( 'on')
plt.show()

plt.figure(figsize = (8 , 3)) 
plt.plot(t , d , label = "d" , color = "royalblue") 
plt.plot(t , q , label = "q" , color = "orangered") 
plt.plot(t , z , label = "0" , color = "forestgreen")
plt.legend() 
plt.legend(ncol = 3 , loc = 4) 
plt.ylabel("Voltage [Volts]") 
plt.xlabel("Time [Seconds]") 
plt.title("Three-phase ABC system") 
plt.grid('on')
plt.show()
