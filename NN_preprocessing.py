import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sn
from sklearn.metrics import mean_squared_error, hinge_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neural_network import MLPRegressor
# implement dq0 transformation
import ClarkePark

def plot_corr_matrix(dataframe: pd.DataFrame) -> None:
    # Plot correlation matrix for dataframe.
    # :parameter dataframe: pandas dataframe containing data.
    correlations = dataframe.corr()
    print(correlations)
    plt.figure(figsize=(12, 12))
    plt.title('Correlation Matrix')
    plt.matshow(correlations)
    sn.heatmap(correlations, annot=True, center=0)
    plt.savefig('figures/correlations.png', bbox_inches='tight')
    
# Returns the sequence of moving RMS Values of val    
def RMS(val, N):
     return (pd.DataFrame(abs(val)**2).rolling(N).mean()) **0.5

if __name__ == '__main__':
    plt.close('all')
    #%% Import Inverter Dataset
    FILENAME = 'Inverter_Data_Set.csv'
    COLS = ['motor speed (min^-1)', 'DC-link voltage (V)', 'DC-link voltage 1 sampling step before in (V)', 'DC-link '
                                                                                                            'voltage '
                                                                                                            '2 '
                                                                                                            'sampling '
                                                                                                            'steps '
                                                                                                            'before '
                                                                                                            'in V',
            'DC-link voltage 3 sampling steps before in V', 'Phase current of phase a in A', 'Phase current of phase '
                                                                                             'b in A', 'Phase current '
                                                                                                       'of phase c in '
                                                                                                       'A',
            'Phase current of phase a 1 sampling step before in A', 'Phase current of phase b 1 sampling step before '
                                                                    'in A', 'Phase current of phase c 1 sampling step '
                                                                            'before in A', 'Phase current of phase a '
                                                                                           '2 sampling steps before in '
                                                                                           'A', 'Phase current of '
                                                                                                'phase b 2 sampling '
                                                                                                'steps before in A',
            'Phase current of phase c 2 sampling steps before in A', 'Phase current of phase a 3 sampling steps '
                                                                     'before in A', 'Phase current of phase b 3 '
                                                                                    'sampling steps before in A',
            'Phase current of phase c 3 sampling steps before in A', 'Duty cycle of phase a 2 sampling steps before',
            'Duty cycle of phase b 2 sampling steps before', 'Duty cycle of phase c 2 sampling steps before',
            'Duty cycle of phase a 3 sampling steps before', 'Duty cycle of phase b 3 sampling steps before',
            'Duty cycle of phase c 3 sampling steps before', 'Measured voltage of phase a 1 sampling step before in '
                                                             'V', 'Measured voltage of phase b 1 sampling step before '
                                                                  'in V', 'Measured voltage of phase c 1 sampling '
                                                                          'step before in V']

    # Read CSV
    df = pd.read_csv(FILENAME, header=0, names=COLS)
    # print(df.head())
    # print(df.columns)
    
    # Define parameters of dataset
    fs = 12.5e6     # Sampling Rate
    Ts = 1/fs       # Sampling Period
    N = len(df[:])  # Total Number of Samples
    T = N*Ts        # Total Sampling Duration
    p = 4           # Induction Machine Poles
    

    # Get desired input and outputs
    I_a = df.loc[:, ['Phase current of phase a in A']]
    RPM = df.loc[:, ['motor speed (min^-1)']]
    V_ab = df.loc[:, ['Measured voltage of phase a 1 sampling step before in V']]        
    V_ab_RMS = RMS(V_ab,len(V_ab))
    
    # Break into separate runs by RPM
    r3000 = range(0,59951)
    # fe = 3000* p / 120
    fe = 60
    r500 = range(59952,144883)
    r1500 = range(144885,199554)
    r3000_2 = range(199555,224534)
    r2500 = range(224535,234526)
    df3k = df.iloc[r3000,:]
    df500 = df.iloc[r500,:]
    df1500 = df.iloc[r1500,:]
    df3k2 = df.iloc[r3000_2,:]
    df2500 = df.iloc[r2500,:]
    wt3k = np.array(range(1,len(r3000)),dtype=(float))
    for i in range(0,len(r3000)-1):
        wt3k[i] = 2*np.pi*fe*float(i*Ts)
    
    
    # Plot correlation matrix
    plot_corr_matrix(df)
    plt.show()

    # x_values, y_values = np.asarray(df.loc[:, ['motor speed (min^-1)', 'DC-link voltage (V)', 'DC-link voltage 1 sampling step before in (V)', 'DC-link '
    #                                                                                                         'voltage '
    #                                                                                                         '2 '
    #                                                                                                         'sampling '
    #                                                                                                         'steps '
    #                                                                                                         'before '
    #                                                                                                         'in V',
    #         'DC-link voltage 3 sampling steps before in V', 'Phase current of phase a in A', 'Phase current of phase '
    #                                                                                           'b in A', 'Phase current '
    #                                                                                                     'of phase c in '
    #                                                                                                     'A',
    #         'Phase current of phase a 1 sampling step before in A', 'Phase current of phase b 1 sampling step before '
    #                                                                 'in A', 'Phase current of phase c 1 sampling step '
    #                                                                         'before in A', 'Phase current of phase a '
    #                                                                                         '2 sampling steps before in '
    #                                                                                         'A', 'Phase current of '
    #                                                                                             'phase b 2 sampling '
    #                                                                                             'steps before in A',
    #         'Phase current of phase c 2 sampling steps before in A', 'Phase current of phase a 3 sampling steps '
    #                                                                   'before in A', 'Phase current of phase b 3 '
    #                                                                                 'sampling steps before in A',
    #         'Phase current of phase c 3 sampling steps before in A', 'Duty cycle of phase a 2 sampling steps before',
    #         'Duty cycle of phase b 2 sampling steps before', 'Duty cycle of phase c 2 sampling steps before',
    #         'Duty cycle of phase a 3 sampling steps before', 'Duty cycle of phase b 3 sampling steps before',
    #         'Duty cycle of phase c 3 sampling steps before',  'Measured voltage of phase b 1 sampling step before '
    #                                                               'in V', 'Measured voltage of phase c 1 sampling '
    #                                                                       'step before in V']]), np.asarray(V_ab)
    
    
    #%% Plot Features
    # for i in range(1,len(COLS)):
    #     plt.figure()
    #     plt.title(COLS[i])
    #     plt.xlabel('Sample Number')
    #     plt.ylabel(COLS[i])
    #     plt.grid(True)
    #     plt.plot(range(0,len(df.loc[:,COLS[i]])),df.loc[:,COLS[i]])        
    #     plt.savefig('figures/'+COLS[i]+'.png',dpi=500)
    #     plt.close()
    #%% Plot Target
    plt.figure()
    plt.plot(RPM)
    plt.title('Motor RPM')
    plt.xlabel('Sample Number')
    plt.ylabel(r'V_{L-L RMS}')
    plt.savefig('figures/MotorRPM.png',dpi=500)
    #%% Preprocessing
    # Take the RMS of the Voltages and Currents
    
    
    V_a = np.array(range(1,len(df3k[:])))
    V_b = np.array(range(1,len(df3k[:])))
    V_c = np.array(range(1,len(df3k[:])))
    I_a_arr = np.array(range(1,len(df3k[:])))
    I_b_arr = np.array(range(1,len(df3k[:])))
    I_c_arr = np.array(range(1,len(df3k[:])))
    delta = 0

    for i in range(1,len(df3k[:])-1):
        # Create an array of each quantity and its values 1, 2 and 3 steps before
        V_a[i] = np.array(df3k.loc[i,'Measured voltage of phase a 1 sampling step before in V'])
        V_b[i] = np.array(df3k.loc[i,'Measured voltage of phase b 1 sampling step before in V'])
        V_c[i] = np.array(df3k.loc[i,'Measured voltage of phase c 1 sampling step before in V'])
        I_a_arr[i] = np.array(df3k.loc[i,'Phase current of phase a in A'])
        I_b_arr[i] = np.array(df3k.loc[i,'Phase current of phase b in A'])
        I_c_arr[i] = np.array(df3k.loc[i,'Phase current of phase c in A'])
    
    # Take the RMS of each array
    V_a_rms = np.array(RMS(V_a,len(df3k)))[len(RMS(V_a,len(df3k)))-1]
    V_b_rms = np.array(RMS(V_b,len(df3k)))[len(RMS(V_b,len(df3k)))-1]
    V_c_rms = np.array(RMS(V_c,len(df3k)))[len(RMS(V_c,len(df3k)))-1]
    I_a_rms = np.array(RMS(I_a_arr,len(df3k)))[len(RMS(I_a_arr,len(df3k)))-1]
    I_b_rms = np.array(RMS(I_b_arr,len(df3k)))[len(RMS(I_a_arr,len(df3k)))-1]
    I_c_rms = np.array(RMS(I_c_arr,len(df3k)))[len(RMS(I_a_arr,len(df3k)))-1]
    
    # Take the dq0 transform of the voltages and currents
    V_d,  V_q,  V_z = ClarkePark.abc_to_dq0(V_a, V_b, V_c, wt3k, delta)
    I_d,  I_q,  I_z = ClarkePark.abc_to_dq0(I_a_arr, I_b_arr, I_c_arr, wt3k, delta)
    
    # Plot the 3-phase Voltages and Currents
    plt.figure()
    plt.plot(V_a,label=r'$V_ab$')
    plt.plot(V_b,label=r'$V_bc$')
    plt.plot(V_c,label=r'$V_ca$')
    plt.title('3-Phase Voltage')
    plt.xlabel('Sample Number')
    plt.ylabel(r'$V_{3-\Phi}$ [V]')
    plt.xlim([0,700])
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/V3p.png')
    
    plt.figure()
    plt.plot(V_a,label=r'$I_a$')
    plt.plot(V_b,label=r'$I_b$')
    plt.plot(V_c,label=r'$I_c$')
    plt.title('3-Phase Current')
    plt.xlabel('Sample Number')
    plt.ylabel(r'$I_{3-\Phi}$ [V]')
    plt.xlim([0,700])
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/I3p.png')  
    

    # Plot the RMS Values
    plt.figure()
    plt.plot(V_a_rms)
    plt.title('RMS Phase A Voltage')
    plt.xlabel('Sample Number')
    plt.ylabel(r'$V_{L-L RMS}$ [V]')
    plt.savefig('figures/RMSV_a.png',dpi=500)
    
    # Plot the dq0 Values
    plt.figure()
    plt.plot(V_d,label=r'$V_d$')
    plt.plot(V_q,label=r'$V_q$')
    plt.plot(V_z,label=r'$V_0$')
    plt.title('dq0 Voltage')
    plt.xlabel('Sample Number')
    plt.ylabel('[V]')
    plt.xlim([0,700])
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/Vdq0.png',dpi=500)
    
    plt.figure()
    plt.plot(I_d,label=r'$I_d$')
    plt.plot(I_q,label=r'$I_q$')
    plt.plot(I_z,label=r'$I_0$')
    plt.title('dq0 Current')
    plt.xlabel('Sample Number')
    plt.ylabel('[A]')
    plt.xlim([0,700])
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/Idq0.png',dpi=500)
    
    x_values, y_values = np.asarray(df3k.loc[:, ['Phase current of phase a in A', 
                                                 'Phase current of phase a 1 sampling step before in A',
                                                 'Phase current of phase a 2 sampling steps before in A',
                                                 'Phase current of phase a 3 sampling steps before in A',
                                                 'motor speed (min^-1)',
                                                 ]])[1:len(df3k)], np.asarray(V_a)
    
    # plt.figure()
    # plt.plot(y_values)
    # plt.show()
    # plt.figure()
    # plt.subplot(211)
    # plt.scatter(x_values[:, 0], y_values)
    # plt.xlabel('Phase current of phase a in A')
    # plt.ylabel('voltage')
    # plt.subplot(212)
    # plt.scatter(x_values[:, 1], y_values)
    # plt.xlabel('motor speed (min^-1)')
    # plt.ylabel('voltage')
    # plt.title('Phase Voltage as a Function of Motor Speed')
    # plt.show()
    
    # Create model and train on data
    model = MLPRegressor(hidden_layer_sizes=(32, 32, 32, 32), max_iter=50)
    # x_values, y_values = np.asarray(df3k)[1:len(df3k)], np.asarray(V_a)
    
    print('Input shape: {}'.format(x_values.shape))
    print('Output shape: {}'.format(y_values.shape))
    train_x, test_x, train_y, test_y = train_test_split(x_values, y_values)
    learning_curve(model, X=x_values, y=y_values, train_sizes=([0.1, 0.2, 0.5, 0.8, 0.9]))
    model.fit(train_x, train_y)
    print('Model Score: {}'.format(model.score(test_x, test_y)))
    y_pred = model.predict(test_x)
    print('Training MSE: {}'.format(mean_squared_error(train_y, model.predict(train_x))))
    print('MSE: {}'.format(mean_squared_error(test_y, y_pred)))
    
    # Plot prediction vs target
    # plt.figure()
    # plt.scatter(test_x,test_y,label='Measured Data')
    # plt.plot(test_x,y_pred,'--'label='Prediction')
    # plt.xlabel('')
    # plt.ylabel('')
    # plt.title('Measured Data vs Predicted Values')
    

