%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  (a) Unconstrained smooth convex minimization                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%        (1a) worst-case of the optimized gradient method with respect to
%             objective value at the best iterate, 20 iterations. Solver
%             set to Mosek with tolerance 1e-10.

clear P A C S val Sol Prob;
P.L=1; P.mu=0; P.R=1;
A.name='OGM2'; A.N=20;
C.name='Obj'; S.solver='mosek'; S.tol=1e-10;
S.verb=0;
[val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val

%%
%        (2a) worst-case of the gradient method with h=1.5 with respect to
%             last gradient norm, 10 iterations. Default Yalmip solver and
%             tolerance.

clear P A C S val Sol Prob;
P.L=1; P.mu=0; P.R=1;
A.name='GM'; A.N=10; A.stepsize=1.5;
C.name='Grad';

[val, Sol, Prob]=pep_fixedsteps(P,A,C); format long; val

%%
%        (3a) worst-case of the fast gradient method with respect to
%             best gradient norm, 5 iterations. Solver set to Sedumi
%             tolerance 1e-9.

clear P A C S val Sol Prob;
P.L=1; P.mu=0; P.R=1;
A.name='FGM1'; A.N=5;
C.name='MinGrad';
S.solver='sedumi'; S.tol=1e-9;

[val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val

%%
%        (4a) worst-case of the unit-step gradient method with respect to
%             best gradient norm, 2 iterations. Solver set to Sedumi
%             with tolerance 1e-9. 2 ways of doing this: via the 'Custom'
%             and via the 'GM' options.
%
clear P A C S val Sol Prob;
P.L=1; P.mu=0; P.R=1;
A.name='Custom'; A.N=2; C.name='MinGrad';
S.solver='sedumi'; S.tol=1e-9; S.verb=0;
A.stepsize=[1 0 ; 1 1];
[val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val

clear P A C S val Sol Prob;
P.L=1; P.mu=0; P.R=1;
A.name='GM'; A.N=2; C.name='MinGrad';
S.solver='sedumi'; S.tol=1e-9; S.verb=0;
A.stepsize=1;
[val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  (b)  Projected/proximal smooth convex minimization                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%        (1b) worst-case of the unit-step projected gradient method with 
%             respect to objective accuracy, 5 iterations. 
%             Solver set to Mosek with tolerance 1e-10.
%

clear P A C S val Sol Prob;
P.L=1; P.mu=0; P.R=1; P.Proj=1; P.Prox=0;
A.name='GM'; A.N=5; A.stepsize=1;
C.name='Obj';
S.relax=1; S.solver='mosek'; S.tol=1e-10; S.verb=0; S.OnlyConjecture=0;
[val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val

%%
%        (2b) worst-case of the fast proximal gradient method (primary 
%             sequence) with respect to objective accuracy, 5 iterations.
%

clear P A C S val Sol Prob;
P.L=1; P.mu=0; P.R=1; P.Proj=0; P.Prox=1;
A.name='FGM1'; A.N=5; A.stepsize=1;
C.name='Obj';
S.relax=1; S.solver='mosek'; S.tol=1e-10; S.verb=0; S.OnlyConjecture=0;
[val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val

%%
%        (3b) worst-case of the fast proximal gradient method with
%        simplified inertial parameters (k-1)/(k+2) (primary sequence) 
%        with respect to objective accuracy, 5 iterations.
%

clear P A C S val Sol Prob;
P.L=1; P.mu=0; P.R=1; P.Proj=0; P.Prox=1;
A.name='FGM1alt'; A.N=5; A.OneSeq=0; C.name='Obj'; S.relax=0;
S.solver='mosek'; S.tol=1e-10; S.verb=0;
A.stepsize=1;S.OnlyConjecture=0;
[val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val

%%
%        (4b) worst-case of the fast proximal gradient method (FPGM) (with
%        simplified inertial parameters (k-1)/(k+2) (secondary sequence) 
%        with respect to objective accuracy, 5 iterations. Variant of FPGM
%        where the proximal operation takes place after the inertia (FPGM2)
%        see [THG2] for details.

clear P A C S val Sol Prob;
P.L=1; P.mu=0; P.R=1; P.Proj=0; P.Prox=1;
A.name='FGM2alt'; A.N=5; A.OneSeq=1; A.stepsize=1;
C.name='Obj';
S.solver='mosek'; S.tol=1e-10; S.verb=0; S.relax=0; S.OnlyConjecture=0;
[val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val


%%
%        (5b) worst-case of the proximal optimized gradient method (POGM)
%        (secondary sequence) with respect to objective accuracy,
%        5 iterations. 
%        The proximal operation takes place after the inertia
%        see [THG2] for details.

clear P A C S val Sol Prob;
P.L=1; P.mu=0; P.R=1; P.Proj=0; P.Prox=1;
A.name='OGM2'; A.N=5; A.OneSeq=1; A.stepsize=1;
C.name='Obj';
S.solver='mosek'; S.tol=1e-10; S.verb=0; S.relax=0; S.OnlyConjecture=0;
[val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val



