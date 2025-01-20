############################################################
### Forked from https://github.com/cmhcbb/attackbox
############################################################

import time
import numpy as np
from numpy import linalg as LA
import random
import torch
from torchvision import transforms

hflip = transforms.RandomHorizontalFlip(p=1.0)

def mulvt(v,t):
##################################
## widely used in binary search ##
## v is batch_size by 1         ##
## t is batch_size by any dim   ##
##################################
    batch_size, other_dim = t.size()[0], t.size()[1:]
    len_dim = len(other_dim)-1
    for i in range(len_dim):
        v = v.unsqueeze(len(v.size()))
    v = v.expand(t.size())
    return v*t

class OPT_attack_lf(object):
    def __init__(self, model, norm_order, inverted=False):
        self.model = model
        self.norm_order = norm_order
        self.inverted = inverted

    def norm(self, p):
        return LA.norm(p, self.norm_order)

    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.01, iterations = 1000):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model
        # y0 = y0[0]
        if (self.symm_predict(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return (False, x0)

        flipped_x0 = np.reshape(x0, (1, 1, 28, 28))
        flipped_x0 = torch.from_numpy(flipped_x0)
        flipped_x0 = hflip(flipped_x0)
        flipped_x0 = flipped_x0.cpu().detach().numpy()
        flipped_x0 = np.reshape(flipped_x0, (28*28))

        # ORIGINAL
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        query_count = 0
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        timestart = time.time()
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape)
            if self.symm_predict(x0+theta)!=y0:
                #l2norm = self.norm(theta)
                initial_lbd = self.norm(theta.flatten())
                theta /= initial_lbd     # might have problem on the defination of direction
                lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)

        # It's very likely that np.random.randn cannot find an adv point.
        if g_theta == np.inf:
            for i in range(num_directions * 10):
                query_count += 1
                theta = np.random.uniform(-1, 1, *x0.shape)
                # theta = np.random.randn(*x0.shape)
                if self.symm_predict(x0+theta)!=y0:
                    initial_lbd = self.norm(theta)
                    theta /= initial_lbd
                    lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                    query_count += count
                    if lbd < g_theta_1:
                        best_theta, g_theta = theta, lbd
                        print("--------> Found distortion %.4f in uniform generator %d" % (g_theta, i))
                        break

        timeend = time.time()
        time1 = timeend - timestart
        if g_theta == np.inf:
            print("Failed to find initial point!!!!")
            return (False, x0)
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))

        timestart = time.time()
        g1_1 = 1.0
        g1_2 = 1.0
        g1_3 = 1.0
        g1_4 = 1.0
        theta, g2 = best_theta, g_theta
        opt_count = 0
        stopping = 0.005
        prev_obj = 100000
        for i in range(iterations):
                
            gradient_1 = np.zeros(theta.shape)
            q = 5
            min_g1 = float('inf')
            for _ in range(q):
                u = np.random.randn(*theta.shape)
                u /= self.norm(u.flatten())
                ttt = theta+beta * u
                ttt /= self.norm(ttt.flatten())
                g1_1, count = self.fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd = g2, tol=beta/500)
                opt_count += count
                gradient_1 += (g1_1-g2)/beta * u

                if g1_1 < min_g1:
                    min_g1 = g1_1
                    min_ttt = ttt

            gradient_2 = np.zeros(theta.shape)
            q = 5
            #min_g1 = float('inf')
            for _ in range(q):
                u = np.random.randn(*theta.shape)
                u /= self.norm(u.flatten())
                ttt = theta+beta * u
                ttt /= self.norm(ttt.flatten())
                g1_2, count = self.fine_grained_binary_search_local(model, 1.0-x0, y0, ttt, initial_lbd = g2, tol=beta/500)
                opt_count += count
                gradient_2 += (g1_2-g2)/beta * u

                if g1_2 < min_g1:
                    min_g1 = g1_2
                    min_ttt = ttt

            gradient_3 = np.zeros(theta.shape)
            q = 5
            #min_g1 = float('inf')
            for _ in range(q):
                u = np.random.randn(*theta.shape)
                u /= self.norm(u.flatten())
                ttt = theta+beta * u
                ttt /= self.norm(ttt.flatten())
                g1_3, count = self.fine_grained_binary_search_local(model, flipped_x0, y0, ttt, initial_lbd = g2, tol=beta/500)
                opt_count += count
                gradient_3 += (g1_3-g2)/beta * u

                if g1_3 < min_g1:
                    min_g1 = g1_3
                    min_ttt = ttt

            gradient_4 = np.zeros(theta.shape)
            q = 5
            #min_g1 = float('inf')
            for _ in range(q):
                u = np.random.randn(*theta.shape)
                u /= self.norm(u.flatten())
                ttt = theta+beta * u
                ttt /= self.norm(ttt.flatten())
                g1_4, count = self.fine_grained_binary_search_local(model, 1.0-flipped_x0, y0, ttt, initial_lbd = g2, tol=beta/500)
                opt_count += count
                gradient_4 += (g1_4-g2)/beta * u

                if g1_4 < min_g1:
                    min_g1 = g1_4
                    min_ttt = ttt

            gradient = 1.0/q * gradient_1 + 1.0/q * gradient_2 + 1.0/q * gradient_3 + 1.0/q * gradient_4

            if (i+1)%1 == 0:
                print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1_1, g2, self.norm((g2*theta).flatten()), opt_count))
                if g2 > prev_obj-stopping:
                    print("stopping")
                    break
                prev_obj = g2

            min_theta = theta
            min_g2 = g2

            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta /= self.norm(new_theta.flatten())
                new_g2, count = self.fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                opt_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = theta - alpha * gradient
                    new_theta /= self.norm(new_theta.flatten())
                    new_g2, count = self.fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                    opt_count += count
                    if new_g2 < g2:
                        min_theta = new_theta
                        min_g2 = new_g2
                        break

            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1

            if g2 < g_theta:
                best_theta, g_theta = theta, g2

            #print(alpha)
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 0.00005):
                    break

        target = self.symm_predict(x0 + g_theta*best_theta)
        timeend = time.time()
        time2 = timeend - timestart
        print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        return (True, x0 + g_theta*best_theta)



    def symm_predict(self, x0):
    
        #print('mmm', x0.shape[0])
        
        if x0.shape[0] == 784:
            x0 = x0.reshape((1,784))
    
        #print('mmm', self.model.predict_label(x0))
        #print('mmm', preds.shape, x0.shape)
        
        flipped_x0 = np.reshape(x0, (x0.shape[0], 1, 28, 28))
        flipped_x0 = torch.from_numpy(flipped_x0)
        flipped_x0 = hflip(flipped_x0)
        flipped_x0 = flipped_x0.cpu().detach().numpy()
        flipped_x0 = np.reshape(flipped_x0, (x0.shape[0],28*28))
        
        preds             = self.model.predict_label(x0)
        preds_flipped     = self.model.predict_label(flipped_x0)
        preds_inv         = self.model.predict_label(1.0-x0)
        preds_inv_flipped = self.model.predict_label(1.0-flipped_x0)
        
        symm_preds = []
        for ii in range(x0.shape[0]):
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



    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        if self.symm_predict(x0+lbd*theta) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while self.symm_predict(x0+lbd_hi*theta) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return 99999, nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while self.symm_predict(x0+lbd_lo*theta) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.symm_predict(x0 + lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best:
            if self.symm_predict(x0+current_best*theta) == y0:
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
            if self.symm_predict(x0 + lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def attack_targeted(self, initial_xi, x0, y0, target, alpha = 0.2, beta = 0.001, iterations = 5000):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model
        #print(y0)
        #y0 = y0[0]
        if (self.symm_predict(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return x0,0,0

        num_samples = 100
        best_theta, g_theta = None, float('inf')
        query_count = 0
        sample_count = 0
        #print("Searching for the initial direction on %d samples: " % (num_samples))
        timestart = time.time()
        #samples = set(random.sample(range(len(train_dataset)), num_samples))
        #train_dataset = train_dataset[samples]
        #for i, (xi, yi) in enumerate(train_dataset):
        #    if i not in samples:
        #        continue
        #    if yi != target:
        #        continue
        #    query_count += 1
        #    if model.predict(xi) == target:
        #       theta = xi - x0
                #l2norm = self.norm(theta)
        #        initial_lbd = self.norm(theta.flatten())
        #        theta /= initial_lbd     # might have problem on the defination of direction
        #        lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
        #       query_count += count
        #        if lbd < g_theta:
        #            best_theta, g_theta = theta, lbd
        #           print("--------> Found distortion %.4f" % g_theta)
        #print(x0)
        xi = initial_xi
        xi = xi.numpy()
        theta = xi - x0
        initial_lbd = self.norm(theta.flatten())
        theta /= initial_lbd     # might have problem on the defination of direction
        lbd, count, lbd_g2 = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, theta)
        query_count += count
        if lbd < g_theta:
            best_theta, g_theta = theta, lbd
            print("--------> Found distortion %.4f" % g_theta)

        timeend = time.time()
        if g_theta == np.inf:
            return "NA", float('inf'), 0
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))

        timestart = time.time()
        g1 = 1.0
        theta, g2 = best_theta, g_theta
        opt_count = 0
        stopping = 1e-8
        prev_obj = 1000000
        for i in range(iterations):
            if g2==0.0:
                break
            gradient = np.zeros(theta.shape)
            q = 20
            min_g1 = float('inf')
            min_lbd = float('inf')
            for _ in range(q):
                u = np.random.randn(*theta.shape)
                u /= self.norm(u.flatten())
                ttt = theta+beta * u
                ttt /= self.norm(ttt.flatten())
                g1, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, ttt, initial_lbd = lbd_g2, tol=beta/500)
                #g1, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, ttt)
                opt_count += count
                gradient += (g1-g2)/beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
                    min_lbd_1 = lbd_hi
            gradient = 1.0/q * gradient

            if (i+1)%10 == 0:
                print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, self.norm((lbd_g2*theta).flatten()), opt_count))
                if g2 > prev_obj-stopping:
                    print("stopping")
                    break
                prev_obj = g2

            min_theta = theta
            min_g2 = g2
            min_lbd = lbd_g2

            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta /= self.norm(new_theta.flatten())
                new_g2, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta, initial_lbd = min_lbd, tol=beta/500)
                #new_g2, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta)
                opt_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    min_lbd = lbd_hi
                else:
                    break


            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = theta - alpha * gradient
                    new_theta /= self.norm(new_theta.flatten())
                    new_g2, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta, initial_lbd = min_lbd, tol=beta/500)
                    #new_g2, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta)
                    opt_count += count
                    if new_g2 < g2:
                        min_theta = new_theta
                        min_g2 = new_g2
                        min_lbd = lbd_hi
                        break

            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
                lbd_g2 = min_lbd
            else:
                theta, g2 = min_ttt, min_g1
                lbd_g2 = min_lbd_1
            if g2 < g_theta:
                best_theta, g_theta = theta, g2
                #lbd_g2 = min_lbd
            #print(alpha)
            if alpha < 1e-6:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 1e-8):
                    break
        g_theta, _ = self.fine_grained_binary_search_local_targeted_original(model, x0, y0, target, best_theta, initial_lbd = 1.0, tol=beta/500)
        dis = self.norm((g_theta*best_theta).flatten())
        target = self.symm_predict(x0 + g_theta*best_theta)
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (dis, target, query_count + opt_count, timeend-timestart))
        return x0 + g_theta*best_theta

    def fine_grained_binary_search_local_targeted(self, model, x0, y0, t, theta, initial_lbd= 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        if self.symm_predict(x0+lbd*theta) != t:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while self.symm_predict(x0+lbd_hi*theta) != t:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 100:
                    return float('inf'), nquery, 1.0
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while self.symm_predict(x0+lbd_lo*theta) == t:
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.symm_predict(x0 + lbd_mid*theta) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        temp_theta = np.abs(lbd_hi*theta)
        temp_theta = np.clip(temp_theta - 0.15, 0.0, None)
        loss = np.sum(np.square(temp_theta))
        #print(lbd_hi)
        return loss, nquery, lbd_hi

    def fine_grained_binary_search_local_targeted_original(self, model, x0, y0, t, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        if self.symm_predict(x0+lbd*theta) != t:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while self.symm_predict(x0+lbd_hi*theta) != t:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 100:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while self.symm_predict(x0+lbd_lo*theta) == t:
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.symm_predict(x0 + lbd_mid*theta) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search_targeted(self, model, x0, y0, t, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best:
            if self.symm_predict(x0+current_best*theta) != t:
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
            if self.symm_predict(x0 + lbd_mid*theta) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def __call__(self, input_xi, label_or_target, initial_xi=None, target=None, TARGETED=False):
        if TARGETED:
            adv = self.attack_targeted(initial_xi, input_xi, label_or_target, target)
            print('HERE')
            exit()
        else:
            adv = self.attack_untargeted(input_xi, label_or_target)
        return adv


