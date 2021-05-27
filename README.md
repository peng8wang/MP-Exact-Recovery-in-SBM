# Non-Convex Exact Community Recovery in Stochastic Block Model

This folder contains the MATLAB source codes for the implementation of all the experiments in the paper

"Non-Convex Exact Community Recovery in Stochastic Block Model" (submitted to Mathematical Programming)
by Peng Wang, Zirui Zhou, Anthony Man-Cho So.

* Contact: Peng Wang
* If you have any questions, please feel free to contact "wp19940121@gmail.com".

===========================================================================

This package contains 3 experimental tests to output the results in the paper:

* In the folder named phase-transition, we conduct the experiment of phase transition to test recovery performance of our approach PPM and compare it with SDP-based approach in Amini et al. (2018), the manifold optimization (MFO) based approach in Bandeira et al. (2016), and the spectral clustering (SC) approach in Abbe et al. (2017).
  - phase_transition.m: Output the recovery performance and running time of above methods
  - PPM.m: Implement our two-stage method consisting of orthogonal iterations (OIs) + projected power iterations (PPIs)
  - manifold_GD: Implement MFO based approach by manifold gradient descent (MGD) in Bandeira et al. (2016)
  - sdp_admm1: Implement SDP-based approach by alternating direction method of multipliers (ADMM) in Amini et al. (2018)
  - SC: Implement vanilla spectral clustering in Abbe et al (2017) via MATLAB function eigs

* In the folder named convergence-performance, we conduct the experiments of convergence performance to test the number of iterations needed by our approach
PPM to exactly identify the underlying communities. For comparison, we also test the convergence performance of MGD, which is an iterative algorithm that has similar per-iteration
cost to our method.
  - convergence_demo.m: Output the convergence performance of our method PPM and MGD

* In the folder named computational-efficiency, we compare the computational efficiency of our proposed method with GPM, MGD, SDP, and SC on both synthetic and real data sets. 
  - GPM.m: Implement the two-stage method consisting of power iterations (PIs) + generalized power iterations (GPIs) in Wang et al. (2020). 
  - test_synthetic.m: Output the results on synthetic data sets
  - test_polbooks.m: Output the results on real data set polbooks
  - test_polblogs.m: Output the results on real data set polblogs
