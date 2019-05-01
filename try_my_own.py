# this is just to demonstrate gradient descent

# https://www.youtube.com/watch?v=sDv4f4s2SB8&t=576s
# video by StatQuest with Josh Starmer on youtube called "Gradient Descent, Step-by-Step"

# also https://www.youtube.com/watch?v=XdM6ER7zTLk

from numpy import *

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def get_slope_of_sum_of_squared_residuals_for_b(points, b, m):
	slope_of_sum_of_squared_residuals = 0
	N = float(len(points))
	for point in points:
		x = point[0]
		y = point[1]
		slope_of_sum_of_squared_residuals += -(2/N) * (y - ((m * x) + b))
	return slope_of_sum_of_squared_residuals

def get_step_size_for_b(points, learning_rate, b, m):
	slope_of_sum_of_squared_residuals = get_slope_of_sum_of_squared_residuals_for_b(points, b, m)
	step_size_for_b = slope_of_sum_of_squared_residuals * learning_rate
	return step_size_for_b

def get_slope_of_sum_of_squared_residuals_for_m(points, b, m):
	slope_of_sum_of_squared_residuals = 0
	N = float(len(points))
	for point in points:
		x = point[0]
		y = point[1]
		slope_of_sum_of_squared_residuals += -(2/N) * x * (y - ((m * x) + b))
	return slope_of_sum_of_squared_residuals

def get_step_size_for_m(points, learning_rate, b, m):
	slope_of_sum_of_squared_residuals = get_slope_of_sum_of_squared_residuals_for_m(points, b, m)
	step_size_for_m = slope_of_sum_of_squared_residuals * learning_rate
	return step_size_for_m

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        #b, m = update_b_and_m(b, m, array(points), learning_rate)
        b = b - get_step_size_for_b(points, learning_rate, b, m)
        m = m - get_step_size_for_m(points, learning_rate, b, m)
    return [b, m]

def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 5000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

    index = 0
    print(points[index][1], " = ", m*points[index][0] + b)
if __name__ == '__main__':
    run()