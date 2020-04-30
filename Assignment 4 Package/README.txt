The user only needs to modify variable values within main.
The following boolean values can be changed to obtain the required values:
sanity_check - outputs a graph of log likelihoods to check if gradient descent is working
log_marginal_likelihood - prints the log marginal likelihood
im_sampling - prints the importance sampling accuracy on the test dataset. Note that the number of samples of w(i) can be changed by changing num_samples
All three boolean values can be set to True to check all results.