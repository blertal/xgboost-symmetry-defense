############################################################
### Forked from https://github.com/cmhcbb/attackbox
############################################################

from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from torchvision import transforms

hflip = transforms.RandomHorizontalFlip(p=1.0)

class HSJA(object):
    def __init__(self,model,constraint=2,inverted=False,num_iterations=40,gamma=1.0,stepsize_search='geometric_progression',max_num_evals=1e4,init_num_evals=100, verbose=True):
        self.model = model
        self.constraint = constraint
        self.inverted = inverted
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.stepsize_search = stepsize_search
        self.max_num_evals = max_num_evals
        self.init_num_evals = init_num_evals
        self.verbose = verbose

    def hsja(self,input_xi,label_or_target,initial_xi,TARGETED):
    
        if self.symm_predict(input_xi) != label_or_target: #####self.model.predict_label(input_xi) != label_or_target):
        #print(self.model.predict_label(input_xi))
        #exit()
        #if self.model.predict_label(input_xi) != label_or_target:
            print("Fail to classify the image. No need to attack.")
            return (False, None)

        # Set parameters
        # original_label = np.argmax(self.model.predict_label(input_xi))
        d = int(np.prod(input_xi.shape))
        # Set binary search threshold.
        if self.constraint == 2:
                theta = self.gamma / (np.sqrt(d) * d)
        else:
                theta = self.gamma / (d ** 2)

        # Initialize.
        perturbed = self.initialize(input_xi, label_or_target, initial_xi, TARGETED)
        if perturbed is None:
                print("Fail to find initial adversarial image.")
                return (False, None)

        # Project the initialization to the boundary.
        perturbed, dist_post_update = self.binary_search_batch(input_xi, np.expand_dims(perturbed, 0), label_or_target, theta, TARGETED)
        dist = self.compute_distance(perturbed, input_xi)

        for j in np.arange(self.num_iterations):
                #params['cur_iter'] = j + 1

                # Choose delta.
                if j==1:
                    delta = 0.1 * (self.model.bounds[1] - self.model.bounds[0])
                else:
                    if self.constraint == 2:
                            delta = np.sqrt(d) * theta * dist_post_update
                    elif self.constraint == np.inf:
                            delta = d * theta * dist_post_update


                # Choose number of evaluations.
                num_evals = int(self.init_num_evals * np.sqrt(j+1))
                num_evals = int(min([num_evals, self.max_num_evals]))

                flipped_xi = np.reshape(perturbed, (1, 1, 28, 28))
                flipped_xi = torch.from_numpy(flipped_xi)
                flipped_xi = hflip(flipped_xi)
                flipped_xi = flipped_xi.cpu().detach().numpy()
                flipped_xi = np.reshape(flipped_xi, (28*28))

                # approximate gradient.
                gradf = self.approximate_gradient(perturbed, label_or_target, num_evals, delta, TARGETED) + self.approximate_gradient(flipped_xi, label_or_target, num_evals, delta, TARGETED) + self.approximate_gradient(1.0-perturbed, label_or_target, num_evals, delta, TARGETED) + self.approximate_gradient(1.0-flipped_xi, label_or_target, num_evals, delta, TARGETED)
                
                if self.constraint == np.inf:
                        update = np.sign(gradf)
                else:
                        update = gradf

                # search step size.
                if self.stepsize_search == 'geometric_progression':
                        # find step size.
                        epsilon = self.geometric_progression_for_stepsize(perturbed, label_or_target,
                                update, dist, j+1, TARGETED)

                        # Update the sample.
                        perturbed = self.clip_image(perturbed + epsilon * update,
                                self.model.bounds[0], self.model.bounds[1])

                        # Binary search to return to the boundary.
                        perturbed, dist_post_update = self.binary_search_batch(input_xi,
                                perturbed[None], label_or_target, theta, TARGETED)

                elif params['stepsize_search'] == 'grid_search':
                        # Grid search for stepsize.
                        epsilons = np.logspace(-4, 0, num=20, endpoint = True) * dist
                        epsilons_shape = [20] + len(input_xi.shape) * [1]
                        perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
                        perturbeds = self.clip_image(perturbeds, self.model.bounds[0], self.model.bounds[1])
                        idx_perturbed = self.decision_function(perturbeds, label_or_target, TARGETED)

                        if np.sum(idx_perturbed) > 0:
                                # Select the perturbation that yields the minimum distance # after binary search.
                                perturbed, dist_post_update = self.binary_search_batch(input_xi,
                                        perturbeds[idx_perturbed], label_or_target, theta, TARGETED)

                # compute new distance.
                dist = self.compute_distance(perturbed, input_xi)
                if self.verbose:
                        print('iteration: {:d}, distance {:.4E}'.format(j+1, dist))

        return (True, perturbed)

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


    def decision_function(self, images, label, TARGETED):
            """
            Decision function output 1 on the desired side of the boundary,
            0 otherwise.
            """
            # images = torch.from_numpy(images).float().cuda()
            assert images is not None
            
            #print(images.shape)
            

            la = self.symm_predict(images)  ##### self.model.predict_label(images)
            #la = self.model.predict_label(images)
            
            #print('qqq',la,label)
            # la = la.cpu().numpy()

            if TARGETED:
                return (la==label)
            else:
                return (la!=label)

    def clip_image(self, image, clip_min, clip_max):
            # Clip an image, or an image batch, with upper and lower threshold.
            return np.minimum(np.maximum(clip_min, image), clip_max)


    def compute_distance(self, x_ori, x_pert):
            # Compute the distance between two images.
            if self.constraint == 2:
                    return np.linalg.norm(x_ori - x_pert)
            elif self.constraint == np.inf:
                    return np.max(abs(x_ori - x_pert))


    def approximate_gradient(self, sample, label_or_target, num_evals, delta, TARGETED):

            # Generate random vectors.
            noise_shape = [num_evals] + list(sample.shape)
            if self.constraint == 2:
                    rv = np.random.randn(*noise_shape)
            elif self.constraint == np.inf:
                    rv = np.random.uniform(low = -1, high = 1, size = noise_shape)

            rv = rv / np.sqrt(np.sum(rv ** 2, axis = (1), keepdims = True))
            perturbed = sample + delta * rv
            perturbed = self.clip_image(perturbed, self.model.bounds[0], self.model.bounds[1])
            rv = (perturbed - sample) / delta

            # query the model.
            decisions = self.decision_function(perturbed, label_or_target, TARGETED)
            decision_shape = [len(decisions)] + [1] * len(sample.shape)
            fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0

            # Baseline subtraction (when fval differs)
            if np.mean(fval) == 1.0: # label changes.
                    gradf = np.mean(rv, axis = 0)
            elif np.mean(fval) == -1.0: # label not change.
                    gradf = - np.mean(rv, axis = 0)
            else:
                    fval -= np.mean(fval)
                    gradf = np.mean(fval * rv, axis = 0)

            # Get the gradient direction.
            gradf = gradf / np.linalg.norm(gradf)

            return gradf


    def project(self, original_image, perturbed_images, alphas):
            # alphas_shape = [1] * len(original_image.shape)
            # alphas = alphas.reshape(alphas_shape)
            assert len(original_image.shape) == 1
            assert len(perturbed_images.shape) == 2
            if self.constraint == 2:
                    #print(alphas.shape,original_image.shape, perturbed_images.shape)
                    return (1-alphas) * original_image + alphas * perturbed_images
            elif self.constraint == np.inf:
                    out_images = self.clip_image(
                            perturbed_images,
                            original_image - alphas,
                            original_image + alphas
                            )
                    return out_images
            else:
                raise Exception(f"Unsupported constraint {self.constraint}")


    def binary_search_batch(self, original_image, perturbed_images, label_or_target, theta, TARGETED):
            """ Binary search to approach the boundar. """
            assert len(original_image.shape) == 1
            assert len(perturbed_images.shape) == 2

            # Compute distance between each of perturbed image and original image.
            dists_post_update = np.array([
                            self.compute_distance(
                                    original_image,
                                    perturbed_image
                            )
                            for perturbed_image in perturbed_images])
            #print(dists_post_update)
            # Choose upper thresholds in binary searchs based on constraint.
            if self.constraint == np.inf:
                    highs = dists_post_update
                    # Stopping criteria.
                    thresholds = np.minimum(dists_post_update * theta, theta)
            else:
                    highs = np.ones(len(perturbed_images))
                    thresholds = theta

            lows = np.zeros(len(perturbed_images))



            # Call recursive function.
            while np.max((highs - lows) / thresholds) > 1:
                    # projection to mids.
                    mids = (highs + lows) / 2.0
                    mid_images = self.project(original_image, perturbed_images, mids)
           #         print(mid_images.shape)
                    # Update highs and lows based on model decisions.
                    decisions = self.decision_function(mid_images, label_or_target, TARGETED)
                    lows = np.where(decisions == 0, mids, lows)
                    highs = np.where(decisions == 1, mids, highs)

            out_images = self.project(original_image, perturbed_images, highs)

            # Compute distance of the output image to select the best choice.
            # (only used when stepsize_search is grid_search.)
            dists = np.array([
                    self.compute_distance(
                            original_image,
                            out_image
                    )
                    for out_image in out_images])
            idx = np.argmin(dists)

            dist = dists_post_update[idx]
            out_image = out_images[idx]
            return out_image, dist


    def initialize(self, input_xi, label_or_target, initial_xi, TARGETED):
            """
            Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
            """
            success = 0
            num_evals = 0

            if initial_xi is None:
                    # Find a misclassified random noise.
                    while num_evals < 1e4:
                            random_noise = np.random.uniform(*self.model.bounds, size = input_xi.shape)
                            #print(random_noise[None].shape)
                            success = self.decision_function(random_noise, label_or_target, TARGETED)
                            self.model.num_queries += 1
                            if success:
                                    break

                    if not success:
                            print("Initialization failed! ")
                            return None

                    # Binary search to minimize l2 distance to original image.
                    low = 0.0
                    high = 1.0
                    while high - low > 0.001:
                            mid = (high + low) / 2.0
                            blended = (1 - mid) * input_xi + mid * random_noise
                            success = self.decision_function(blended, label_or_target, TARGETED)
                            if success:
                                    high = mid
                            else:
                                    low = mid

                    initialization = (1 - high) * input_xi + high * random_noise

            else:
                    initialization = initial_xi

            return initialization


    def geometric_progression_for_stepsize(self, x, label_or_target, update, dist, j, TARGETED):
            """
            Geometric progression to search for stepsize.
            Keep decreasing stepsize by half until reaching
            the desired side of the boundary,
            """
            epsilon = dist / np.sqrt(j)

            def phi(epsilon):
                    new = x + epsilon * update
                    success = self.decision_function(new, label_or_target, TARGETED)
                    return success

            while not phi(epsilon):
                    epsilon /= 2.0

            return epsilon


    def __call__(self, input_xi, label_or_target, initial_xi=None, target=None, TARGETED=False):
        # input_xi = input_xi.cpu().numpy()
        # label_or_target = label_or_target.cpu().numpy()
        adv = self.hsja(input_xi, label_or_target, initial_xi, TARGETED)
        # adv = torch.from_numpy(adv).float().cuda()
        return adv

