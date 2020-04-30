from data_utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import multivariate_normal


def GD(x_train, y_train, x_test, y_test, w0, learning_rate, num_iter=500, SGD=False):
    k = 0
    wk = w0 # weights
    # losses
    losses = np.array([])
    # iter
    iters = np.array([])
    # "test" (validation) data losses
    losses_test = np.array([])
    # testing accuracies
    accuracies = np.array([])
    # seeding
    np.random.seed(100)
    # gradient change
    prev_grad = np.zeros((wk.shape))
    grad = np.ones((wk.shape))
    while num_iter>0 and np.linalg.norm(np.subtract(grad,prev_grad))>1e-9:
        # find gradient gk
        # update the theta
        # different gradient for different methods
        if SGD:
            t = np.random.randint(0,x_train.shape[0]-1)
            x_t = x_train[t,:]
            f_pred_t = _sigmoid(wk, x_t)
            y_t = y_train[t]
            grad = _grad(x_t, y_t, f_pred_t)
        else:
            f_pred = _sigmoid(wk, x_train)
            grad = _grad(x_train, y_train, f_pred)
        wk -= learning_rate*grad
        # evaluate f wk+1 = log Pr(y|w,X)
        f_wkp1 = _f(x_train, y_train, wk)
        # record loss
        losses = np.append(losses,f_wkp1)
        losses_test = np.append(losses_test, _f(x_test, y_test, wk))
        y_pred_test = _sigmoid(wk, x_test)
        accuracies = np.append(accuracies, _accuracy(y_test, y_pred_test))
        iters = np.append(iters, k)
        # calculate stopping condition
        k += 1
        num_iter -= 1

    return wk, losses, losses_test, accuracies, iters


def run_GD(x_train, y_train, x_test, y_test, lr_desired, num_iter, save=True):
    for lr in lr_desired:
        curr = time.time()
        w0 = np.zeros((x_train.shape[1],1))
        wk, losses, losses_test, accuracies, iters = GD(x_train, y_train, x_test, y_test, w0, learning_rate=lr, num_iter = num_iter)
        end = time.time()
        plt.plot(losses, label="Learning rate {}".format(lr))
        print("Time elapsed: {}".format(end - curr))
        print("Full Batch GD Training loss: {}".format(losses[losses.shape[0] - 1]))
        print("Full Batch GD Testing accuracy: {}, Loss: {}; learning rate = {}".format(accuracies[accuracies.shape[0]-1], losses_test[losses_test.shape[0]-1], lr))
    plt.legend(loc="upper right")
    plt.xlabel("Iteration #")
    plt.ylabel("Negative Log Likelihood (Loss)")
    plt.title("Full Batch GD Loss vs Iteration #")
    if save:
        plt.savefig("Full Batch GD.png")
    plt.show()


def run_SGD(x_train, y_train, x_test, y_test, lr_desired, num_iter, save=True):
    for lr in lr_desired:
        curr = time.time()
        w0 = np.zeros((x_train.shape[1],1))
        wk, losses, losses_test, accuracies, iters = GD(x_train, y_train, x_test, y_test, w0, learning_rate=lr, num_iter = num_iter, SGD=True)
        end = time.time()
        plt.plot(losses, label="Learning rate {}".format(lr))
        print("Time elapsed: {}".format(end-curr))
        print("SGD Training loss: {}".format(losses[losses.shape[0]-1]))
        print("SGD Testing accuracy: {}, Loss: {}; learning rate = {}".format(accuracies[accuracies.shape[0]-1], losses_test[losses_test.shape[0]-1],lr))
    plt.legend(loc="upper right")
    plt.xlabel("Iteration #")
    plt.ylabel("Negative Log Likelihood (Loss)")
    plt.title("SGD Loss vs Iteration #")
    if save:
        plt.savefig("SGD.png")
    plt.show()


def lml(x_train, y_train, x_test, y_test, var_list):
    """

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test: required for GD function
    :param var_list: list of variances
    :return:
    """
    lr = 0.01
    num_iter = 1000
    w0 = np.zeros((x_train.shape[1], 1))
    wk, losses, losses_test, accuracies, iters = GD(x_train, y_train, x_test, y_test, w0, learning_rate=lr,
                                                          num_iter=num_iter)
    # wk is best estimate w*
    # evaluate f wk+1 = log Pr(y|w,X)
    f_wkp1 = _f(x_train, y_train, wk)
    for var in var_list:
        # evaluate log Pr(w*)
        log_pr_w = _log_pr_w(wk, var)
        # evaluate log g(w*)
        # find Hessian
        hessian = _H(x_train, wk, var)
        # find g(w*)
        log_g = -1.0 * wk.shape[0] * 0.5 * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(-1.0 * hessian))
        # print("g is {}".format(g))
        # record log marginal likelihood
        lml = -1 * f_wkp1 + log_pr_w - log_g
        print("Log marginal likelihood for variance = {} is: {}".format(var, lml))

def importance_sampling(x_train, y_train, x_test, y_test, w_map, var=1, num_samples = 3):
    w_samples = _sample_q(w_map, var, num_samples) # proposal distribution to sample w from
    r_wi_list = np.array([])
    for i in range(num_samples):
        w_i = w_samples[i,:]
        w_i = w_i.reshape((w_i.shape[0],1))
        r_wi_list = np.append(r_wi_list, _r(w_i, x_train, y_train, var, w_map))
    # print(r_wi_list.shape)
    den = np.sum(r_wi_list)
    const = np.divide(r_wi_list, den)
    pred_vals = np.array([])
    for k in range(x_test.shape[0]):
        x_star = x_test[k,:]
        #y_star = y_test[k,:].astype(int)
        y_star = np.array([1])
        # print(y_star.shape)
        pr_list = np.array([])
        for i in range(num_samples):
            # find posterior
            w_i = w_samples[i, :]
            w_i = w_i.reshape((w_i.shape[0], 1))
            pr_list = np.append(pr_list, _p_y_wx(x_star, y_star, w_i)*r_wi_list[i]/den)
        # print("probability list shape {}".format(pr_list.shape))
        # print("r list shape {}".format(const.shape))
        # pred_val= np.vdot(pr_list, const)
        #print(pred_val.shape)
        pred_val = sum(pr_list)
        pred_vals = np.append(pred_vals, pred_val)
    pred_vals = pred_vals.reshape((pred_vals.shape[0],1))
    # print(pred_vals)
    # print(y_test)
    print("Importance sampling testing accuracy with sample size s={} is : {}".format(num_samples, _accuracy(y_test, pred_vals)))


def _p_y_wx(x, y, w):
    x = x.reshape((x.shape[0], 1))
    f_hat = _sigmoid(w,x.T)
    result = np.multiply(np.power(f_hat, y[0]), np.power(1-f_hat[0],1-y[0]))
    # print("probability shape {}".format(result))
    return result

def _r(w, X, y, var, w_map):
    # find r of w*, w is a sample from proposal distribution q
    pr_y_wx = np.exp(-1*_f(X,y,w))
    pr_w = np.exp(_log_pr_w(w, var))
    q_w = _q(w, w_map, var)
    return np.divide(np.multiply(pr_y_wx, pr_w),q_w)

def _q(w, w_map, var):
    cov = var*np.eye(w_map.shape[0])
    mean = w_map.flatten()
    q = multivariate_normal.pdf(w, mean=mean, cov=cov)
    assert(q.shape[0] == w.shape[0])
    return np.product(q)

def _sample_q(w_map, var, num_samples):
    # proposal distribution
    cov = var * np.eye(w_map.shape[0])
    mean = w_map.flatten()
    q = multivariate_normal(mean=mean, cov=cov)
    s = np.asarray(q.rvs(size=num_samples, random_state=123))
    return s

# def _pr_y_wx(w, X, y_bool):
#     result = np.array([1])
#     y_all = y_bool.astype(int)
#     for i in range(X.shape[0]):
#         # compute sigma wx
#         y = y_all[0, :].reshape((1, 1))
#         f_hat = _sigmoid(w, X[0, :])
#         temp1 = np.power(f_hat, y)
#         temp2 = np.power(np.subtract(np.ones(f_hat.shape),f_hat), np.subtract(np.ones(y.shape),y))
#         assert(temp1.shape==(1,1))
#         assert(temp2.shape==(1,1))
#         result = np.multiply(result,np.multiply(temp1,temp2))
#         assert (result.shape == (1, 1))
#     return result
#
# def _pr_w()

def _H (X, w, var):
    # compute and return the Hessian (size D+1 x D+1), D+1 = length of x
    # don't forget f_hat is _sigmoid

    # sum of two Hessians, one of log Pr y|w, X and the other of log Pr w
    # H of log Pr y|w, X
    result1 = np.zeros((X.shape[1],X.shape[1]))  # D+1 x D+1
    for i in range(X.shape[0]):  # 1 to N
        x_i = X[i,:]  # 1 x D+1
        x_i_reshaped = x_i.reshape((5,1))
        f = _sigmoid(w, x_i)
        result_i = f*(f-1)*np.dot(x_i_reshaped, x_i_reshaped.T)
        result1=np.add(result1,result_i)  # check dimension here

    # H of log Pr w
    result2 = -1.0*np.eye(X.shape[1])/var

    return np.add(result1, result2)

def _log_pr_w(w, var):
    first = -1.0*w.shape[0]*0.5*np.log(2*np.pi)
    second = -1.0*w.shape[0]*0.5*np.log(var)
    third = -1.0*np.vdot(w,w)/(2*var)
    return first + second + third

def _sigmoid (th, x):
    # 1/1+e^z
    exponent = np.dot(x,th)
    den = np.ones((exponent.shape[0],1)) + np.exp(exponent)
    result = 1./den
    return result

def _grad(x, y, f_pred):
    return (np.sum(np.subtract(y,f_pred)*x,axis=0)).reshape((5,1))

def _f(x,y,th):
    fHat = _sigmoid(th, x)
    # NLL of Bernoulli
    first = np.vdot(y, (np.log(fHat)))
    temp1 = np.subtract(np.ones((y.shape[0], y.shape[1])),y)
    temp2 = np.log(np.subtract(np.ones((fHat.shape[0], fHat.shape[1])),fHat))
    second = np.vdot(temp1, temp2)
    nll = first + second
    return -1.*nll

def _rmse(y_pred, y_actual):
    """
    calculate the root mean squared error between the estimated y values and
    the actual y values
    :param y_estimates: list of ints
    :param y_valid: list of ints, actual y values
    :return: float, rmse value between two lists
    """
    return np.sqrt(np.average(np.abs(y_pred-y_actual)**2))

def _accuracy(y,y_pred):
    y_pred=np.rint(y_pred)
    result = 0
    for i in range(y_pred.shape[0]):
        if y[i]==y_pred[i]:
            result+=1
    return result*1.0/y_pred.shape[0]

def _plot2(x,y1,y2,legend1,legend2,x_label,y_label,title):
    line1,line2 = plt.plot(x,y1,x,y2)
    plt.legend((line1,line2),(legend1,legend2))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    y_train = y_train[:,(1,)]
    y_valid = y_valid[:,(1,)]
    y_test = y_test[:,(1,)]

    # merge training and testing into one
    x_train = np.vstack((x_train,x_valid))
    y_train = np.vstack((y_train,y_valid))

    # augment x
    x_train = np.hstack((np.ones((x_train.shape[0],1)),x_train))
    x_test = np.hstack((np.ones((x_test.shape[0],1)),x_test))

    sanity_check = False
    log_marginal_likelihood = False
    im_sampling = True


    if sanity_check:
        # program #
        lr_list = [0.01,0.001,0.0001] # learning rates to try
        num_iter = 3000 # number of iterations
        save = True # save the images to disk
        run_GD(x_train, y_train, x_test, y_test, lr_desired=lr_list, num_iter=num_iter, save=save)
        run_SGD(x_train, y_train, x_test, y_test, lr_desired=lr_list, num_iter=num_iter, save=save)
    if log_marginal_likelihood:
        var_list = [0.5,1,2]
        lml(x_train, y_train, x_test, y_test, var_list)
    if im_sampling:
        w0 = np.zeros((x_train.shape[1], 1))
        num_iter = 1000
        num_samples = 200
        w_map, losses, losses_test, accuracies, iters = GD(x_train, y_train, x_test, y_test, w0, learning_rate=0.01,
                                                        num_iter=num_iter, SGD=False)

        importance_sampling(x_train, y_train, x_test, y_test, w_map, num_samples=num_samples)