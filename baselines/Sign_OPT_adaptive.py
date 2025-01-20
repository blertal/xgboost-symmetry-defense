############################################################
### Forked from https://github.com/cmhcbb/attackbox
############################################################

import time
import numpy as np
from numpy import linalg as LA
import scipy.spatial
from scipy.linalg import qr
#from qpsolvers import solve_qp
import random
import torch
from torchvision import transforms

start_learning_rate = 1.0
MAX_ITER = 1000

hflip = transforms.RandomHorizontalFlip(p=1.0)


def quad_solver(Q, b):
    """
    Solve min_a  0.5*aQa + b^T a s.t. a>=0
    """
    K = Q.shape[0]
    alpha = np.zeros((K,))
    g = b
    Qdiag = np.diag(Q)
    for i in range(20000):
        delta = np.maximum(alpha - g/Qdiag,0) - alpha
        idx = np.argmax(abs(delta))
        val = delta[idx]
        if abs(val) < 1e-7:
            break
        g = g + val*Q[:,idx]
        alpha[idx] += val
    return alpha

def sign(y):
    """
    y -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = np.sign(y)
    y_sign[y_sign==0] = 1
    return y_sign

class OPT_attack_sign_SGD_cpu(object):
    def __init__(self, model, norm_order, inverted=False, k=200, train_dataset=None):
        self.model = model
        self.k = k
        self.norm_order = norm_order
        self.train_dataset = train_dataset
        self.log = np.ones((MAX_ITER,2))

    def norm(self, p):
        return LA.norm(p, self.norm_order)

    def get_log(self):
        return self.log
        
        
    def symm_predict(self, x0):
  
        #print('mmm', x0.shape[0])
        
        if x0.shape[0] == 784:
            x0 = x0.reshape((1,784))
   
            #print('mmm', self.model.predict_label(x0))
            #print('mmm', preds.shape, x0.shape)
        #print('mmm', x0.shape[0])
        
        flipped_x0 = np.reshape(x0, (x0.shape[0], 1, 28, 28))
        flipped_x0 = torch.from_numpy(flipped_x0)
        flipped_x0 = hflip(flipped_x0)
        flipped_x0 = flipped_x0.cpu().detach().numpy()
        flipped_x0 = np.reshape(flipped_x0, (x0.shape[0],28*28))
        
        if x0.shape[0] != 1:
            x0 = x0.reshape((x0.shape[0],784))
            flipped_x0 = x0.reshape((x0.shape[0],784))
        
        preds             = self.model.predict_label(x0)
        preds_flipped     = self.model.predict_label(flipped_x0)
        preds_inv         = self.model.predict_label(1.0-x0)
        preds_inv_flipped = self.model.predict_label(1.0-flipped_x0)
        
        the_range = x0.shape[0]
        if x0.shape[0] == 784:
            preds = np.array([preds])
            preds_flipped = np.array([preds_flipped])
            preds_inv = np.array([preds_inv])
            preds_inv_flipped = np.array([preds_inv_flipped])
            the_range = 1
        
        symm_preds = []
        for ii in range(the_range):
            if (preds[ii] == preds_flipped[ii]) or (preds[ii] == preds_inv[ii]) or (preds[ii] == preds_inv_flipped[ii]):
                symm_preds.append(preds[ii])
            elif (preds_flipped[ii] == preds_inv[ii]) or (preds_flipped[ii] == preds_inv_flipped[ii]):
                symm_preds.append(preds_flipped[ii])
            elif (preds_inv[ii] == preds_inv_flipped[ii]):
                symm_preds.append(preds_inv[ii])
            else:
                rand_no = np.random.randint(0, 4)
                if rand_no == 0:
                    symm_preds.append(preds[ii])
                elif rand_no == 1:
                    symm_preds.append(preds_flipped[ii])
                elif rand_no == 2:
                    symm_preds.append(preds_inv[ii])
                elif rand_no == 3:
                    symm_preds.append(preds_inv_flipped[ii])

        symm_preds = np.asarray(symm_preds)
        return symm_preds
        

    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1000, query_limit=20000,
                          distortion=None, seed=None, svm=False, momentum=0.0, stopping=0.0001):
        """ Attack the original image and return adversarial example
            model: (cpu model)
            train_dataset: set of training data
            (x0, y0): original image
        """

        model = self.model
        # y0 = y0[0]
        query_count = 0
        ls_total = 0

        if (self.symm_predict(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return (False, x0)

        if seed is not None:
            np.random.seed(seed)

        # Calculate a good starting point.
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        timestart = time.time()
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape)
            if self.symm_predict(x0+np.array(theta, dtype=float))!=y0:
                initial_lbd = self.norm(theta)
                theta /= initial_lbd
                lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)

        # It's very likely that np.random.randn cannot find an adv point.
        if g_theta == float('inf'):
            for i in range(num_directions * 10):
                query_count += 1
                theta = np.random.uniform(-1, 1, *x0.shape)
                # theta = np.random.randn(*x0.shape)
                if self.symm_predict(x0+np.array(theta, dtype=float))!=y0:
                    initial_lbd = self.norm(theta)
                    theta /= initial_lbd
                    lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                    query_count += count
                    if lbd < g_theta:
                        best_theta, g_theta = theta, lbd
                        print("--------> Found distortion %.4f in uniform generator %d" % (g_theta, i))
                        break

        timeend = time.time()
        time1 = timeend-timestart
        if g_theta == float('inf'):
            print("Failed to find initial point!!!!")
            return (False, x0)
        print("==========> Found best distortion %.4f in %.4f seconds "
              "using %d queries" % (g_theta, timeend-timestart, query_count))

        if x0.shape[0] == 784:
            x0 = x0.reshape((1,784))
            
        flipped_x0 = np.reshape(x0, (x0.shape[0], 1, 28, 28))
        flipped_x0 = torch.from_numpy(flipped_x0)
        flipped_x0 = hflip(flipped_x0)
        flipped_x0 = flipped_x0.cpu().detach().numpy()
        flipped_x0 = np.reshape(flipped_x0, (x0.shape[0],28*28))

        self.log[0][0], self.log[0][1] = g_theta, query_count
        # Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        vg = np.zeros_like(xg)
        learning_rate = start_learning_rate
        prev_obj = 100000
        distortions = [gg]
        for i in range(iterations):
            if svm == True:
                #sign_gradient, grad_queries = self.sign_grad_svm(x0, y0, xg, initial_lbd=gg, h=beta)
                sg1, gq1 = self.sign_grad_svm(x0, y0, xg, initial_lbd=gg, h=beta)
                sg2, gq2 = self.sign_grad_svm(1.0-x0, y0, xg, initial_lbd=gg, h=beta)
                sg3, gq3 = self.sign_grad_svm(flipped_x0, y0, xg, initial_lbd=gg, h=beta)
                sg4, gq4 = self.sign_grad_svm(1.0-flipped_x0, y0, xg, initial_lbd=gg, h=beta)
                sign_gradient = sg1 + sg2 + sg3 + sg4
                grad_queries  = gq1 + gq2 + gq3 + gq4
            else:
                #sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)
                sg1, gq1 = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)
                sg2, gq2 = self.sign_grad_v1(1.0-x0, y0, xg, initial_lbd=gg, h=beta)
                sg3, gq3 = self.sign_grad_v1(flipped_x0, y0, xg, initial_lbd=gg, h=beta)
                sg4, gq4 = self.sign_grad_v1(1.0-flipped_x0, y0, xg, initial_lbd=gg, h=beta)
                sign_gradient = sg1 + sg2 + sg3 + sg4
                grad_queries  = gq1 + gq2 + gq3 + gq4

            if False:
                # Compare cosine distance with numerical gradient.
                gradient, _ = self.eval_grad(model, x0, y0, xg, initial_lbd=gg, tol=beta/500, h=0.01)
                print("    Numerical - Sign gradient cosine distance: ",
                      scipy.spatial.distance.cosine(gradient.flatten(), sign_gradient.flatten()))

            # Line search
            ls_count = 0
            min_theta = xg
            min_g2 = gg
            min_vg = vg
            for _ in range(15):
                if momentum > 0:
#                     # Nesterov
#                     vg_prev = vg
#                     new_vg = momentum*vg - alpha*sign_gradient
#                     new_theta = xg + vg*(1 + momentum) - vg_prev*momentum
                    new_vg = momentum*vg - alpha*sign_gradient
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * sign_gradient
                new_theta /= self.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local(
                    model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                ls_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    if momentum > 0:
                        min_vg = new_vg
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    if momentum > 0:
#                         # Nesterov
#                         vg_prev = vg
#                         new_vg = momentum*vg - alpha*sign_gradient
#                         new_theta = xg + vg*(1 + momentum) - vg_prev*momentum
                        new_vg = momentum*vg - alpha*sign_gradient
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * sign_gradient
                    new_theta /= self.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local(
                        model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        if momentum > 0:
                            min_vg = new_vg
                        break
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-8):
                    break

            xg, gg = min_theta, min_g2
            vg = min_vg

            query_count += (grad_queries + ls_count)
            ls_total += ls_count
            distortions.append(gg)

            if query_count > query_limit:
               break

            if (i+1)%10==0:
                print("Iteration %3d distortion %.4f num_queries %d" % (i+1, gg, query_count))
            self.log[i+1][0], self.log[i+1][1] = gg, query_count
            #if distortion is not None and gg < distortion:
            #    print("Success: required distortion reached")
            #    break

#             if gg > prev_obj-stopping:
#                 print("Success: stopping threshold reached")
#                 break
#             prev_obj = gg
        target = self.symm_predict(x0 + np.array(gg*xg, dtype=float))
        timeend = time.time()
        time2 = timeend-timestart
        print("\nAdversarial Example Found Successfully: distortion %.4f target"
              " %d queries %d \nTime: %.4f seconds" % (gg, target, query_count, timeend-timestart))

        self.log[i+1:,0] = gg
        self.log[i+1:,1] = query_count
        #print(self.log)
        #print("Distortions: ", distortions)
        return (True, x0 + np.array(gg*xg, dtype=float))

    def sign_grad_v1(self, x0, y0, theta, initial_lbd, h=0.001, D=4, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k
        sign_grad = np.zeros(theta.shape)
        queries = 0
        ### USe orthogonal transform
        #dim = np.prod(sign_grad.shape)
        #H = np.random.randn(dim, K)
        #Q, R = qr(H, mode='economic')
        xs = []
        preds = []
        us = []
        for iii in range(K):
#             # Code for reduced dimension gradient
#             u = np.random.randn(N_d,N_d)
#             u = u.repeat(D, axis=0).repeat(D, axis=1)
#             u /= self.norm(u)
#             u = u.reshape([1,1,N,N])

            u = np.random.randn(*theta.shape)
            #u = Q[:,iii].reshape(sign_grad.shape)
            u /= self.norm(u)
            us.append(u)

            new_theta = theta + h*u
            new_theta /= self.norm(new_theta)

            xs.append(x0+np.array(initial_lbd*new_theta, dtype=float))

        preds = self.symm_predict(np.array(xs))

        for iii in range(K):
            sign = 1
            pred = preds[iii]
            u = us[iii]
            # Targeted case.
            if (target is not None and pred == target):
                sign = -1

            # Untargeted case
            # preds.append(self.symm_predict(x0+np.array(initial_lbd*new_theta, dtype=float)).item())
            if (target is None and pred != y0):
                sign = -1
            queries += 1
            sign_grad += u*sign

        sign_grad /= K

#         sign_grad_u = sign_grad/self.norm(sign_grad)
#         new_theta = theta + h*sign_grad_u
#         new_theta /= self.norm(new_theta)
#         fxph, q1 = self.fine_grained_binary_search_local(self.model, x0, y0, new_theta, initial_lbd=initial_lbd, tol=h/500)
#         delta = (fxph - initial_lbd)/h
#         queries += q1
#         sign_grad *= 0.5*delta

        return sign_grad, queries

    def sign_grad_v2(self, x0, y0, theta, initial_lbd, h=0.001, K=200):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = np.zeros(theta.shape)
        queries = 0
        for _ in range(K):
            u = np.random.randn(*theta.shape)
            u /= self.norm(u)

            ss = -1
            new_theta = theta + h*u
            new_theta /= self.norm(new_theta)
            if self.symm_predict(x0+np.array(initial_lbd*new_theta, dtype=float)) == y0:
                ss = 1
            queries += 1
            sign_grad += sign(u)*ss
        sign_grad /= K
        return sign_grad, queries


    def sign_grad_svm(self, x0, y0, theta, initial_lbd, h=0.001, K=100, lr=5.0, target=None):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = np.zeros(theta.shape)
        queries = 0
        dim = np.prod(theta.shape)
        X = np.zeros((dim, K))
        for iii in range(K):
            u = np.random.randn(*theta.shape)
            u /= self.norm(u)

            sign = 1
            new_theta = theta + h*u
            new_theta /= self.norm(new_theta)

            # Targeted case.
            if (target is not None and
                self.symm_predict(x0+np.array(initial_lbd*new_theta, dtype=float)) == target):
                sign = -1

            # Untargeted case
            if (target is None and
                self.symm_predict(x0+np.array(initial_lbd*new_theta, dtype=float)) != y0):
                sign = -1
                #if self.symm_predict(x0+np.array((initial_lbd*1.00001)*new_theta, dtype=float)) == y0:
                #    print "Yes"
                #else:
                #    print "No"

            queries += 1
            X[:,iii] = sign*u.reshape((dim,))

        Q = X.transpose().dot(X)
        q = -1*np.ones((K,))
        G = np.diag(-1*np.ones((K,)))
        h = np.zeros((K,))
        ### Use quad_qp solver
        #alpha = solve_qp(Q, q, G, h)
        ### Use coordinate descent solver written by myself, avoid non-positive definite cases
        alpha = quad_solver(Q, q)
        sign_grad = (X.dot(alpha)).reshape(theta.shape)

        return sign_grad, queries



    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        if self.symm_predict(x0+np.array(lbd*theta, dtype=float)) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while self.symm_predict(x0+np.array(lbd_hi*theta, dtype=float)) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while self.symm_predict(x0+np.array(lbd_lo*theta, dtype=float)) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.symm_predict(x0 + np.array(lbd_mid*theta, dtype=float)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best:
            if self.symm_predict(x0+np.array(current_best*theta, dtype=float)) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.symm_predict(x0 + np.array(lbd_mid*theta, dtype=float)) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def eval_grad(self, model, x0, y0, theta, initial_lbd, tol=1e-5,  h=0.001, sign=False):
        # print("Finding gradient")
        fx = initial_lbd # evaluate function value at original point
        grad = np.zeros_like(theta)
        x = theta
        # iterate over all indexes in x
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

        queries = 0
        while not it.finished:

            # evaluate function at x+h
            ix = it.multi_index
            oldval = x[ix]
            x[ix] = oldval + h # increment by h
            unit_x = x / self.norm(x)
            if sign:
                if self.symm_predict(x0+np.array(initial_lbd*unit_x, dtype=float)) == y0:
                    g = 1
                else:
                    g = -1
                q1 = 1
            else:
                fxph, q1 = self.fine_grained_binary_search_local(model, x0, y0, unit_x, initial_lbd = initial_lbd, tol=h/500)
                g = (fxph - fx) / (h)

            queries += q1
            # x[ix] = oldval - h
            # fxmh, q2 = self.fine_grained_binary_search_local(model, x0, y0, x, initial_lbd = initial_lbd, tol=h/500)
            x[ix] = oldval # restore

            # compute the partial derivative with centered formula
            grad[ix] = g
            it.iternext() # step to next dimension

        # print("Found gradient")
        return grad, queries

    def attack_targeted(self, x0, y0, target, alpha = 0.2, beta = 0.001, iterations = 5000, query_limit=40000,
                        distortion=None, seed=None, svm=False, stopping=0.0001):
        """ Attack the original image and return adversarial example
            model: (cpu model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model
        y0 = y0[0]
        print("Targeted attack - Source: {0} and Target: {1}".format(y0, target.item()))

        if (self.symm_predict(x0) == target):
            print("Image already target. No need to attack.")
            return x0, 0.0

        if self.train_dataset is None:
            print("Need training dataset for initial theta.")
            return x0, 0.0

        if seed is not None:
            np.random.seed(seed)

        num_samples = 100
        best_theta, g_theta = None, float('inf')
        query_count = 0
        ls_total = 0
        sample_count = 0
        print("Searching for the initial direction on %d samples: " % (num_samples))
        timestart = time.time()

        # Iterate through training dataset. Find best initial point for gradient descent.
        for i, (xi, yi) in enumerate(self.train_dataset):
            yi_pred = self.symm_predict(xi)
            query_count += 1
            if yi_pred != target:
                continue

            theta = xi.cpu().numpy() - x0.cpu().numpy()
            initial_lbd = self.norm(theta)
            theta /= initial_lbd
            lbd, count = self.fine_grained_binary_search_targeted(model, x0, y0, target, theta, initial_lbd, g_theta)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                print("--------> Found distortion %.4f" % g_theta)

            sample_count += 1
            if sample_count >= num_samples:
                break

            if i > 500:
                break

        timeend = time.time()
        if g_theta == np.inf:
            return x0, float('inf')
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" %
              (g_theta, timeend-timestart, query_count))

        # Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        learning_rate = start_learning_rate
        prev_obj = 100000
        distortions = [gg]
        for i in range(iterations):
            if svm == True:
                #sign_gradient, grad_queries = self.sign_grad_svm(x0, y0, xg, initial_lbd=gg, h=beta)
                sg1, gq1 = self.sign_grad_svm(x0, y0, xg, initial_lbd=gg, h=beta, target=target)
                sg2, gq2 = self.sign_grad_svm(1.0-x0, y0, xg, initial_lbd=gg, h=beta, target=target)
                sg3, gq3 = self.sign_grad_svm(flipped_x0, y0, xg, initial_lbd=gg, h=beta, target=target)
                sg4, gq4 = self.sign_grad_svm(1.0-flipped_x0, y0, xg, initial_lbd=gg, h=beta, target=target)
                sign_gradient = sg1 + sg2 + sg3 + sg4
                grad_queries  = gq1 + gq2 + gq3 + gq4
            else:
                #sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)
                sg1, gq1 = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta, target=target)
                sg2, gq2 = self.sign_grad_v1(1.0-x0, y0, xg, initial_lbd=gg, h=beta, target=target)
                sg3, gq3 = self.sign_grad_v1(flipped_x0, y0, xg, initial_lbd=gg, h=beta, target=target)
                sg4, gq4 = self.sign_grad_v1(1.0-flipped_x0, y0, xg, initial_lbd=gg, h=beta, target=target)
                sign_gradient = sg1 + sg2 + sg3 + sg4
                grad_queries  = gq1 + gq2 + gq3 + gq4

            if False:
                # Compare cosine distance with numerical gradient.
                gradient, _ = self.eval_grad(model, x0, y0, xg, initial_lbd=gg, tol=beta/500, h=0.01)
                print("    Numerical - Sign gradient cosine distance: ",
                      scipy.spatial.distance.cosine(gradient.flatten(), sign_gradient.flatten()))

            # Line search
            ls_count = 0
            min_theta = xg
            min_g2 = gg
            for _ in range(15):
                new_theta = xg - alpha * sign_gradient
                new_theta /= self.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local_targeted(
                    model, x0, y0, target, new_theta, initial_lbd = min_g2, tol=beta/500)
                ls_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = xg - alpha * sign_gradient
                    new_theta /= self.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local_targeted(
                        model, x0, y0, target, new_theta, initial_lbd = min_g2, tol=beta/500)
                    ls_count += count
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        break

            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-8):
                    break

            xg, gg = min_theta, min_g2

            query_count += (grad_queries + ls_count)
            ls_total += ls_count
            distortions.append(gg)

            if query_count > query_limit:
                break

            if i%5==0:
                print("Iteration %3d distortion %.4f num_queries %d" % (i+1, gg, query_count))
#                 print("Iteration: ", i, " Distortion: ", gg, " Queries: ", query_count,
#                       " LR: ", alpha, "grad_queries", grad_queries, "ls_queries", ls_count)

            #if distortion is not None and gg < distortion:
            #    print("Success: required distortion reached")
            #    break

#             if gg > prev_obj-stopping:
#                 print("Success: stopping threshold reached")
#                 break
#             prev_obj = gg

        adv_target = self.symm_predict(x0 + np.array(gg*xg, dtype=float))
        if (adv_target == target):
            timeend = time.time()
            print("\nAdversarial Example Found Successfully: distortion %.4f target"
                  " %d queries %d LS queries %d \nTime: %.4f seconds" % (gg, target, query_count, ls_total, timeend-timestart))

            return x0 + np.array(gg*xg, dtype=float)
        else:
            print("Failed to find targeted adversarial example.")
            return x0

    def fine_grained_binary_search_local_targeted(self, model, x0, y0, t, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        if self.symm_predict(x0 + np.array(lbd*theta, dtype=float)) != t:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while self.symm_predict(x0 + np.array(lbd_hi*theta, dtype=float)) != t:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 100:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while self.symm_predict(x0 + np.array(lbd_lo*theta, dtype=float)) == t:
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.symm_predict(x0 + np.array(lbd_mid*theta, dtype=float)) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid

#         temp_theta = np.abs(lbd_hi*theta)
#         temp_theta = np.clip(temp_theta - 0.15, 0.0, None)
#         loss = np.sum(np.square(temp_theta))
        return lbd_hi, nquery

    def fine_grained_binary_search_targeted(self, model, x0, y0, t, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best:
            if self.symm_predict(x0 + np.array(current_best*theta, dtype=float)) != t:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.symm_predict(x0 + np.array(lbd_mid*theta, dtype=float)) != t:
                lbd_lo = lbd_mid
            else:
                lbd_hi = lbd_mid
        return lbd_hi, nquery

    def __call__(self, input_xi, label_or_target, target=None, distortion=None, seed=None,
                 svm=False, query_limit=40000, momentum=0.0, stopping=0.0001, TARGETED=False):
        if target is not None:
            adv = self.attack_targeted(input_xi, label_or_target, target, distortion=distortion,
                                       seed=seed, svm=svm, query_limit=query_limit, stopping=stopping)
        else:
            return self.attack_untargeted(input_xi, label_or_target, distortion=distortion,
                                         seed=seed, svm=svm, query_limit=query_limit, momentum=momentum,
                                         stopping=stopping)
        return adv


