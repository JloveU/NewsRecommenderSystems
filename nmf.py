#!python3
"""
Created on 2015-12-12
@author: yuqiang
Modified from Chih-Jen Lin's NMF Algorithm
http://www.csie.ntu.edu.tw/~cjlin/nmf/

# NMF by alternative non-negative least squares using projected gradients
# Author: Chih-Jen Lin, National Taiwan University
# Python/numpy translation: Anthony Di Franco
"""

from numpy import *
from numpy.linalg import norm
from time import time
from sys import stdout


def nmf(V, K, tol=0.0000000000001, timelimit=25, maxiter=8000):
 """
 (W,H) = nmf(V,K,tol,timelimit,maxiter)
 W,H: output solution
 K: column number of W
 tol: tolerance for a relative stopping condition
 timelimit, maxiter: limit of time and iterations
 """

 print('NMF started.')
 time_start = time()

 N = len(V)
 M = len(V[0])
 W = random.rand(N, K)
 H = random.rand(K, M)
 initt = time()

 gradW = dot(W, dot(H, H.T)) - dot(V, H.T)
 gradH = dot(dot(W.T, W), H) - dot(W.T, V)
 initgrad = norm(r_[gradW, gradH.T])
 # print('Init gradient norm %f' % initgrad)
 tolW = max(0.001,tol)*initgrad
 tolH = tolW

 for iter in range(1,maxiter):
  # stopping condition
  projnorm = norm(r_[gradW[logical_or(gradW<0, W>0)],
                                 gradH[logical_or(gradH<0, H>0)]])
  if projnorm < tol*initgrad or time() - initt > timelimit: break

  (W, gradW, iterW) = nlssubprob(V.T,H.T,W.T,tolW,1000)
  W = W.T
  gradW = gradW.T

  if iterW==1: tolW = 0.1 * tolW

  (H,gradH,iterH) = nlssubprob(V,W,H,tolH,1000)
  if iterH==1: tolH = 0.1 * tolH

  if iter % 2 == 0: stdout.write('.'); stdout.flush()

 print('')

 # print('Iter = %d Final proj-grad norm %f' % (iter, projnorm))

 time_end = time()
 print('NMF ended. %fs cost.' % (time_end - time_start))

 return (W,H)


def nlssubprob(V,W,Hinit,tol,maxiter):
 """
 H, grad: output solution and gradient
 iter: #iterations used
 V, W: constant matrices
 Hinit: initial solution
 tol: stopping tolerance
 maxiter: limit of iterations
 """

 H = Hinit
 WtV = dot(W.T, V)
 WtW = dot(W.T, W)

 alpha = 1; beta = 0.1;
 for iter in range(1, maxiter):
  grad = dot(WtW, H) - WtV
  projgrad = norm(grad[logical_or(grad < 0, H >0)])
  if projgrad < tol: break

  # search step size
  for inner_iter in range(1,20):
   Hn = H - alpha*grad
   Hn = where(Hn > 0, Hn, 0)
   d = Hn-H
   gradd = sum(grad * d)
   dQd = sum(dot(WtW,d) * d)
   suff_decr = 0.99*gradd + 0.5*dQd < 0;
   if inner_iter == 1:
    decr_alpha = not suff_decr; Hp = H;
   if decr_alpha:
    if suff_decr:
     H = Hn; break;
    else:
     alpha = alpha * beta;
   else:
      if not suff_decr or (Hp == Hn).all():
       H = Hp; break;
      else:
       alpha = alpha/beta; Hp = Hn;

  if iter == maxiter:
   print('Max iter in nlssubprob')
 return (H, grad, iter)