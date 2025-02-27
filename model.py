import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.acquisition import gaussian_lcb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

df = pd.read_csv("water_turbine_data.csv") 

space = [
    Integer(df['num_paddles'].min() - 5, df['num_paddles'].max() + 5, name='num_paddles'),
    Real(df['paddle_angle'].min() - 10, df['paddle_angle'].max() + 10, name='paddle_angle')
]

def objective(params):
    num_paddles, paddle_angle = params
    closest_row = df.iloc[((df['num_paddles'] - num_paddles).abs() +
                           (df['paddle_angle'] - paddle_angle).abs()).idxmin()]
    
    return -closest_row['millivolts']  # negative because minimize in gp_minimize


res = gp_minimize(objective, space, n_calls=70, acq_func='LCB', random_state=42)

# train Gaussian Process Regression
X_train = df[['num_paddles', 'paddle_angle']].values
y_train = df['millivolts'].values
kernel = RBF(length_scale=10) + WhiteKernel(noise_level=0.5)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.5, normalize_y=True)
gpr.fit(X_train, y_train)

# wider range of values
num_paddles_range = np.arange(df['num_paddles'].min() - 5, df['num_paddles'].max() + 6, 1)
paddle_angle_range = np.linspace(df['paddle_angle'].min() - 10, df['paddle_angle'].max() + 10, 50)
X_mesh, Y_mesh = np.meshgrid(num_paddles_range, paddle_angle_range)
X_pred = np.c_[X_mesh.ravel(), Y_mesh.ravel()]

Z_pred = gpr.predict(X_pred).reshape(X_mesh.shape)

# find the best predicted values from GPR
max_index = np.unravel_index(np.argmax(Z_pred), Z_pred.shape)
best_num_paddles = X_mesh[max_index]
best_paddle_angle = Y_mesh[max_index]
best_millivolts = Z_pred[max_index]
print(f"Best Number of Paddles (GPR): {best_num_paddles}")
print(f"Best Paddle Angle (GPR): {best_paddle_angle}")
print(f"Predicted Maximum Millivolt Output (GPR): {best_millivolts}")

# plot 3D surface plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_mesh, Y_mesh, Z_pred, cmap=cm.viridis, edgecolor='k', linewidth=0)
ax.set_xlabel("Number of Paddles")
ax.set_ylabel("Paddle Angle")
ax.set_zlabel("Predicted Millivolts")
ax.set_title("3D Surface Plot of Optimized Turbine Output")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# plot optimization progression
generations = np.arange(len(res.func_vals))
plt.figure(figsize=(8, 5))
plt.plot(generations, -np.array(res.func_vals), marker='o', linestyle='-', color='b')
plt.xlabel("Iterations")
plt.ylabel("Millivolt Output")
plt.title("Optimization Progression")
plt.show()

# plot 2D slices
plt.figure(figsize=(8, 5))
plt.scatter(df['num_paddles'], df['millivolts'], label="Real Data", color='blue')
plt.plot(num_paddles_range, gpr.predict(np.c_[num_paddles_range, [df['paddle_angle'].mean()]*len(num_paddles_range)]), label="Prediction", color='red')
plt.xlabel("Number of Paddles")
plt.ylabel("Millivolts Output")
plt.legend()
plt.title("Effect of Number of Paddles on Output")
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df['paddle_angle'], df['millivolts'], label="Real Data", color='blue')
plt.plot(paddle_angle_range, gpr.predict(np.c_[[df['num_paddles'].mean()]*len(paddle_angle_range), paddle_angle_range]), label="Prediction", color='red')
plt.xlabel("Paddle Angle")
plt.ylabel("Millivolts Output")
plt.legend()
plt.title("Effect of Paddle Angle on Output")
plt.show()
