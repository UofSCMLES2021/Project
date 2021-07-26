import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sn
from sklearn.metrics import mean_squared_error, hinge_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neural_network import MLPRegressor


def plot_corr_matrix(dataframe: pd.DataFrame) -> None:
    """ Plot correlation matrix for dataframe.

        :parameter dataframe: pandas dataframe containing data."""
    correlations = dataframe.corr()
    print(correlations)
    plt.figure(figsize=(12, 12))
    plt.title('Correlation Matrix')
    # plt.matshow(correlations)
    sn.heatmap(correlations, annot=True, center=0)
    # plt.savefig('figures/correlations.png', bbox_inches='tight')


if __name__ == '__main__':
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
                                                                            'before in A', 'Phase current of phase a  '
                                                                                           'sampling steps before in '
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

    # Get desired input and outputs
    y = df.loc[:, ['Phase current of phase a in A']]
    x = df.loc[:, ['motor speed (min^-1)']]
    z = df.loc[:, ['Measured voltage of phase a 1 sampling step before in V']]

    # Plot correlation matrix
    plot_corr_matrix(df)
    plt.tight_layout()
    plt.show()

    # Create model and train on data
    model = MLPRegressor(hidden_layer_sizes=(32, 32, 32, 32), max_iter=1000)
    # x_values, y_values = np.asarray(df), np.asarray(z)
    # x_values, y_values = np.asarray(df.loc[:, ['motor speed (min^-1)', 'DC-link voltage (V)', 'DC-link voltage 1 sampling step before in (V)', 'DC-link '
    #                                                                                                         'voltage '
    #                                                                                                         '2 '
    #                                                                                                         'sampling '
    #                                                                                                         'steps '
    #                                                                                                         'before '
    #                                                                                                         'in V',
    #         'DC-link voltage 3 sampling steps before in V', 'Phase current of phase a in A', 'Phase current of phase '
    #                                                                                          'b in A', 'Phase current '
    #                                                                                                    'of phase c in '
    #                                                                                                    'A',
    #         'Phase current of phase a 1 sampling step before in A', 'Phase current of phase b 1 sampling step before '
    #                                                                 'in A', 'Phase current of phase c 1 sampling step '
    #                                                                         'before in A', 'Phase current of phase a  '
    #                                                                                        'sampling steps before in '
    #                                                                                        'A', 'Phase current of '
    #                                                                                             'phase b 2 sampling '
    #                                                                                             'steps before in A',
    #         'Phase current of phase c 2 sampling steps before in A', 'Phase current of phase a 3 sampling steps '
    #                                                                  'before in A', 'Phase current of phase b 3 '
    #                                                                                 'sampling steps before in A',
    #         'Phase current of phase c 3 sampling steps before in A', 'Duty cycle of phase a 2 sampling steps before',
    #         'Duty cycle of phase b 2 sampling steps before', 'Duty cycle of phase c 2 sampling steps before',
    #         'Duty cycle of phase a 3 sampling steps before', 'Duty cycle of phase b 3 sampling steps before',
    #         'Duty cycle of phase c 3 sampling steps before',  'Measured voltage of phase b 1 sampling step before '
    #                                                               'in V', 'Measured voltage of phase c 1 sampling '
    #                                                                       'step before in V']]), np.asarray(z)
    x_values, y_values = np.asarray(df.loc[:, ['Phase current of phase a in A',
                                               'Phase current of phase a  sampling steps before in A',
                                               'Phase current of phase a 1 sampling step before in A', 'motor speed ('
                                                                                                    'min^-1)']]), \
                         np.asarray(z)
    y_values = y_values.squeeze(axis=1)
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
    # plt.show()
    print('Input shape: {}'.format(x_values.shape))
    print('Output shape: {}'.format(y_values.shape))
    train_x, test_x, train_y, test_y = train_test_split(x_values, y_values)
    # learning_curve(model, X=x_values, y=y_values, train_sizes=([0.1, 0.2, 0.5, 0.8, 0.9]))
    model.fit(train_x, train_y)
    print('Model Score: {}'.format(model.score(test_x, test_y)))
    y_pred = model.predict(test_x)
    print('Training MSE: {}'.format(mean_squared_error(train_y, model.predict(train_x))))
    print('MSE: {}'.format(mean_squared_error(test_y, y_pred)))

