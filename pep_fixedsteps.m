function [val,Sol,Prob]=pep_fixedsteps(P,A,C,S)
%
%   Author: A. Taylor; Universite catholique de Louvain.
%   Date:   December 23, 2015
%
%
%---Performance Estimation Problem [THG][DT][KF] routine using YALMIP [Lof]
%
% [THG] A.B. Taylor, J.M. Hendrickx, F. Glineur. "Smooth Strongly Convex
%       Interpolation and Exact Worst-case Performance of First-order
%       Methods." (2015).
% [THG2] A.B. Taylor, J.M. Hendrickx, F. Glineur. "Exact Worst-case Performance
%	of First-order Algorithms for Composite Convex Optimization." (2015).
% [DT] Y. Drori, M. Teboulle. "Performance of first-order methods for
%      smooth convex minimization: a novel approach."
%      Mathematical Programming 145.1-2 (2014): 451-482.
% [KF] D. Kim, J.A. Fessler. "Optimized first-order methods for smooth
%      convex minimization", Mathematical Programming (2015).
% [Lof] J. Lofberg. "Yalmip A toolbox for modeling and optimization in
%       MATLAB". Proceedings of the CACSD Conference (2004).
%
%
%
%   Inputs:
%         - P: problem class structure  (P.L, P.mu, P.R, P.prox, P.proj)
%         - A: algorithm structure      (A.name, A.stepsize, A.N)
%         - C: criterion structure      (C.name)
%         - S: solver and problem attributes (S.tol, S.verb, S.structure,
%              S.relax, S.OnlyConjecture)
%         (see below for detailed descriptions)
%
%   Outputs:
%         - val: structure containing
%              val.primal: primal optimal value
%              val.dual:   dual optimal value
%              val.conj:   conjectured optimal value
%                          (NaN if no conjecture is available)
%           Primal and dual values may correspond to non-feasible points
%           (up to numerical and solver precision), hence the interval
%           [primal dual] may not contain the conjectured result.
%         - Sol: structure containing
%              Sol.G:      Gram matrix of gradients, subgradients and x0
%                          (primal PSD matrix)
%              Sol.f:      Function values (primal linear variables)
%              Sol.S :     Dual PSD matrix
%              Sol.lambda: multipliers of linear inequalities
%              Sol.err:    error code returned by the solver
%         - Prob: summary of the problem setting (to help diagnose problems
%           in case something went wrong in the input arguments)
%
%   Problem class structure:
%         - P.L:    Lipschitz constant (default L=1).
%         - P.mu:   Strong convexity constant mu (default mu=0). This
%                   option is ignored in the proximal and projected cases
%		    at the moment.
%         - P.R:    Bound on distance to optimal solution R (default R=1)
%         - P.Prox: [0 | 1] Proximal setting, problem is min f(x)+h(x) with
%                   f L-smooth and convex (no strong convexity yet) and
%                   h is a general convex function. This option is ignored
%                   in the projected case. 
%         - P.Proj: [0 | 1] Projected setting, problem is min f(x)+Ind(x)
%                   with f L-smooth and convex (no strong convexity yet)
%                   and Ind(x) is an indicator function. 
%
%   Algorithm structure:
%         - A.name: name of the algorithm to be chosen among
%              'GM' for the gradient method GM
%              'FGM1' for the fast gradient method FGM (primary sequence)
%              'FGM2' for the fast gradient method FGM (secondary sequence)
%              'FGM1alt' for the FGM with inertial parameter (k-1)/(k+2)
%              (primary sequence)
%              'FGM2alt' for the FGM with inertial parameter (k-1)/(k+2)
%              (secondary sequence)
%              'OGM1' for the optimized method OGM (primary* sequence)
%              'OGM2' for the optimized method OGM (secondary* sequence)
%              'Custom' for a custom fixed-step algorithm (see below).
%              Custom is only available in the non-proximal and
%              non-projected setting.
%           (* note that [KF] presents OGM in terms of its secondary seq.)
%         - A.stepsize: stepsize coefficients
%              'GM' method: scalar stepsize (default h=1.5)
%              'FGM' or 'OGM' method: not applicable
%              'Custom' method: contains an NxN matrix of coefficients H
%               such that each step of the method corresponds to
%               x_i = x_0 - 1/L * sum_{k=1}^{i-1} H(i,k) * g_{k-1}
%         - A.N: number of iteration (default N=1)
%         - A.OneSeq: [0 | 1] for projected/proximal algorithms, 0 means
%               that the projection/proximal step is taken just after the
%               explicit gradient step (standard). 1 correspond to have
%               only one sequence x_i on which both implicit and explicit
%               gradient are evaluated (thus, first implicit and then
%               explicit --- see FPGM2 and POGM in [THG2] for details).
%               Note that the algorithm ends after an implicit step 
%               (before the corresponding explicit step), and that the 
%               algorithm use exactly the same steps as for the
%               unconstrained case, on both implicit and explicit
%               sequences, which is very natural.
%               
%
%   Criterion structure:
%         - C.name: criterion name to be chosen among
%              'Obj' for the objective value of the last iterate
%             *'Grad' for the residual gradient norm of the last iterate
%                     or equivalently norm of approximate first-order
%                     optimality conditions at the last iterate (from a
%                     projection or proximal step) in the constrained
%                     and proximal cases.
%             *'MinGrad' for the smallest gradient norm among all iterates
%                     or equivalently norm of approximate first-order
%                     optimality conditions at the best iterate in 
%             *'Dist' for the distance between last iterate and opt.
%                     solution
%
%             *NOTE: at the moment, only the objective is usable in 
%                    the proximal and projected settings
%
%
%   Solver and problem attributes:
%         - S.tol: tolerance for SDP solver (default 1e-8)
%         - S.verb: verbose mode (0 or 1 ; default 1)
%         - S.relax: use relaxation proposed in [DT] (0 or 1 ; default 0)
%         - S.solver: 'sedumi' or 'mosek' (default: yalmip's default)
%         - S.OnlyConjecture: only conjectures are evaluated
%
%
%   Examples: 
%	----- (A) Unconstrained setting -----
%
%
%        (1A) worst-case of the optimized gradient method with respect to
%             objective value at the best iterate, 20 iterations. Solver
%             set to Mosek with tolerance 1e-10.
%
%           P.L=1; P.mu=0; P.R=1;
%           A.name='OGM2'; A.N=40;
%           C.name='Obj'; S.solver='mosek'; S.tol=1e-10;
%           S.verb=0;
%           [val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; 
%           1/val.primal
%
%        (2A) worst-case of the gradient method with h=1.5 with respect to
%             last gradient norm, 10 iterations. Default Yalmip solver and
%             tolerance.
%
%           P.L=1; P.mu=0; P.R=1;
%           A.name='GM'; A.N=10; A.stepsize=1.5;
%           C.name='Grad';
%
%           [val, Sol, Prob]=pep_fixedsteps(P,A,C); format long; val
%
%        (3A) worst-case of the fast gradient method with respect to
%             best gradient norm, 5 iterations. Solver set to Sedumi
%             tolerance 1e-9.
%
%           P.L=1; P.mu=0; P.R=1;
%           A.name='FGM1'; A.N=5;
%           C.name='MinGrad';
%           S.solver='sedumi'; S.tol=1e-9;
%
%           [val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val
%
%        (4A) worst-case of the unit-step gradient method with respect to
%             best gradient norm, 2 iterations. Solver set to Sedumi
%             with tolerance 1e-9. 2 ways of doing this: via the 'Custom'
%             and via the 'GM' options.
%
%           P.L=1; P.mu=0; P.R=1;
%           A.name='Custom'; A.N=2; C.name='MinGrad';
%           S.solver='sedumi'; S.tol=1e-9; S.verb=0;
%           A.stepsize=[1 0 ; 1 1];
%           [val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val
%
%           P.L=1; P.mu=0; P.R=1;
%           A.name='GM'; A.N=2; C.name='MinGrad';
%           S.solver='sedumi'; S.tol=1e-9; S.verb=0;
%           A.stepsize=1;
%           [val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val
%
%
%	----- (B) Proximal/projected setting -----
%
%
%        (1B) worst-case of the unit-step projected gradient method with 
%             respect to objective accuracy, 5 iterations. 
%             Solver set to Mosek with tolerance 1e-10.
%
% 
%           clear P A C S val Sol Prob;
%           P.L=1; P.mu=0; P.R=1; P.Proj=1; P.Prox=0;
%           A.name='GM'; A.N=5; A.stepsize=1;
%           C.name='Obj';
%           S.relax=0; S.solver='mosek'; S.tol=1e-10; S.verb=0;
%           [val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val
%
%        (2B) worst-case of the fast proximal gradient method (primary 
%             sequence) with respect to objective accuracy, 5 iterations.
%
% 
%           clear P A C S val Sol Prob;
%           P.L=1; P.mu=0; P.R=1; P.Proj=0; P.Prox=1;
%           A.name='FGM1'; A.N=5; A.stepsize=1;
%           C.name='Obj';
%           S.relax=0; S.solver='mosek'; S.tol=1e-10; S.verb=0;
%           [val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val
%
%        (3B) worst-case of the fast proximal gradient method with
%        simplified inertial parameters (k-1)/(k+2) (primary sequence) 
%        with respect to objective accuracy, 5 iterations.
%
% 
%           clear P A C S val Sol Prob;
%           P.L=1; P.mu=0; P.R=1; P.Proj=0; P.Prox=1;
%           A.name='FGM1alt'; A.N=5; A.OneSeq=0; C.name='Obj'; S.relax=0;
%           S.solver='mosek'; S.tol=1e-10; S.verb=0;
%           A.stepsize=1;
%           [val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val
%
%        (4B) worst-case of the fast proximal gradient method (FPGM) (with
%        simplified inertial parameters (k-1)/(k+2) (secondary sequence) 
%        with respect to objective accuracy, 5 iterations. Variant of FPGM
%        where the proximal operation takes place after the inertia (FPGM2)
%        see [THG2] for details.
% 
%           clear P A C S val Sol Prob;
%           P.L=1; P.mu=0; P.R=1; P.Proj=0; P.Prox=1;
%           A.name='FGM2alt'; A.N=5; A.OneSeq=1; A.stepsize=1;
%           C.name='Obj';
%           S.solver='mosek'; S.tol=1e-10; S.verb=0; S.relax=0;
%           [val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val
%
%        (5B) worst-case of the proximal optimized gradient method (POGM)
%        (secondary sequence) with respect to objective accuracy,
%        5 iterations. 
%        The proximal operation takes place after the inertia
%        see [THG2] for details.
% 
%           clear P A C S val Sol Prob;
%           P.L=1; P.mu=0; P.R=1; P.Proj=0; P.Prox=1;
%           A.name='OGM2'; A.N=5; A.OneSeq=1; A.stepsize=1;
%           C.name='Obj';
%           S.solver='mosek'; S.tol=1e-10; S.verb=0; S.relax=0;
%           [val, Sol, Prob]=pep_fixedsteps(P,A,C,S); format long; val
%
%% Parameter association:
if nargin>=1
    if isfield(P,'L')
        L=P.L;
    else
        L=1;
    end
    if isfield(P,'mu')
        mu=P.mu;
    else
        mu=0;
    end
    if isfield(P,'R')
        R=P.R;
    else
        R=1;
    end
    if isfield(P,'Prox')
        Prox=P.Prox;
    else
        Prox=0;
    end
    if isfield(P,'Proj')
        Proj=P.Proj;
        if (Proj)
            Prox=0;
        end
    else
        Proj=0;
    end
else
    L=1;R=1;mu=0;Prox=0;Proj=0;
end

if nargin>=2
    if isfield(A,'name')
        method=A.name;
    else
        method='GM';
    end
    if isfield(A,'stepsize')
        h=A.stepsize;
    else
        h=1;A.stepsize=h;
    end
    if isfield(A,'N')
        N=A.N;
    else
        N=1;
    end
    if isfield(A,'OneSeq')
        OneSeq=A.OneSeq;
    else
        OneSeq=0;
    end
else
    method='GM';h=1.5;N=1;OneSeq=0;
end
if nargin>=3
    if isfield(C,'name')
        criterion=C.name;
    else
        criterion='Obj';
    end
else
    criterion='Obj';
end
if nargin>=4
    if isfield(S,'tol')
        tol_spec=S.tol;
    else
        tol_spec=1e-8;
    end
    if isfield(S,'verb')
        verb_spec=S.verb;
    else
        verb_spec=1;
    end
    if isfield(S,'relax')
        relax=S.relax;
    else
        relax=0;
    end
    if isfield(S,'solver')
        solver=S.solver;
    else
        solver='Yalmip''s default';
    end
    if isfield(S,'OnlyConjecture')
        OnlyConjecture=S.OnlyConjecture;
    else
        OnlyConjecture=0;
    end
else
    tol_spec=1e-8;verb_spec=1;relax=0;OnlyConjecture=0;
    solver='Yalmip''s default';
end

%% Method' choice

steps_h=zeros(N+1,N);
switch method
    case 'FGM1alt'
        steps_h(2,1)=1; %step for x1
        for i=2:N-1
            cur_step_param=(i-1)/(i+2);
            steps_h(i+1,:)=steps_h(i,:)+cur_step_param*(steps_h(i,:)-steps_h(i-1,:));
            steps_h(i+1,i)=1+cur_step_param;
            steps_h(i+1,i-1)=steps_h(i+1,i-1)-cur_step_param;
        end
        steps_h(end,:)=steps_h(end-1,:);
        steps_h(end,end)=1;
        steps_h2=steps_h;
    case 'FGM2alt'
        steps_h=zeros(N+1,N);
        steps_h(2,1)=1; %step for x1
        for i=2:N
            cur_step_param=(i-1)/(i+2);
            steps_h(i+1,:)=steps_h(i,:)+cur_step_param*(steps_h(i,:)-steps_h(i-1,:));
            steps_h(i+1,i)=1+cur_step_param;
            steps_h(i+1,i-1)=steps_h(i+1,i-1)-cur_step_param;
        end
        steps_h2=steps_h;
    case 'FGM1'
        t=zeros(N-1,1);
        t(1,1)=1;
        for i=1:N-1
            t(i+1,1)=(1+sqrt(1+4*t(i,1)^2))/2;
        end
        steps_h(2,1)=1; %step for x1
        for i=2:N-1
            cur_step_param=(t(i,1)-1)/t(i+1,1);
            steps_h(i+1,:)=steps_h(i,:)+cur_step_param*(steps_h(i,:)-steps_h(i-1,:));
            steps_h(i+1,i)=1+cur_step_param;
            steps_h(i+1,i-1)=steps_h(i+1,i-1)-cur_step_param;
        end
        steps_h(end,:)=steps_h(end-1,:);
        steps_h(end,end)=1;
        steps_h2=steps_h;
    case 'FGM2'
        t=zeros(N+1,1);
        t(1,1)=1;
        
        for i=1:N
            t(i+1,1)=(1+sqrt(1+4*t(i,1)^2))/2;
        end
        
        steps_h=zeros(N+1,N);
        steps_h(2,1)=1; %step for x1
        for i=2:N
            cur_step_param=(t(i,1)-1)/t(i+1,1);
            steps_h(i+1,:)=steps_h(i,:)+cur_step_param*(steps_h(i,:)-steps_h(i-1,:));
            steps_h(i+1,i)=1+cur_step_param;
            steps_h(i+1,i-1)=steps_h(i+1,i-1)-cur_step_param;
        end
        steps_h2=steps_h;
    case 'StrCvxFGM1'
        steps_h(2,1)=1; %step for x1
        gamma=(1-sqrt(mu/L))/(1+sqrt(mu/L));
        for i=2:N-1
            cur_step_param=gamma;
            steps_h(i+1,:)=steps_h(i,:)+cur_step_param*(steps_h(i,:)-steps_h(i-1,:));
            steps_h(i+1,i)=1+cur_step_param;
            steps_h(i+1,i-1)=steps_h(i+1,i-1)-cur_step_param;
        end
        steps_h(end,:)=steps_h(end-1,:);
        steps_h(end,end)=1;
        steps_h2=steps_h;
    case  'OGM2'
        t(1,1)=1;
        for i=1:N
            if (i<=N-1)
                t(i+1,1)=(1+sqrt(1+4*t(i,1)^2))/2;
            else
                t(i+1,1)=(1+sqrt(1+8*t(i,1)^2))/2;
            end
        end
        steps_h=zeros(N+2,N+1);
        for i=0:N-1
            cur_step_param=(t(i+1,1)-1)/t(i+2,1);
            cur_step_param2=(2*t(i+1,1)-1)/t(i+2,1);
            steps_h(i+3,:)=steps_h(i+2,:)+cur_step_param*(steps_h(i+2,:)-steps_h(i+1,:));
            steps_h(i+3,i+2)=1+cur_step_param2;
            steps_h(i+3,i+1)=steps_h(i+3,i+1)-cur_step_param;
        end
        steps_h=steps_h(2:end,2:end);
        steps_h2=steps_h;
    case  'OGM1'
        t(1,1)=1;
        for i=1:N
            if (i<=N-1)
                t(i+1,1)=(1+sqrt(1+4*t(i,1)^2))/2;
            else
                t(i+1,1)=(1+sqrt(1+8*t(i,1)^2))/2;
            end
        end
        steps_h=zeros(N+2,N+1);
        for i=0:N-2
            cur_step_param=(t(i+1,1)-1)/t(i+2,1);
            cur_step_param2=(2*t(i+1,1)-1)/t(i+2,1);
            steps_h(i+3,:)=steps_h(i+2,:)+cur_step_param*(steps_h(i+2,:)-steps_h(i+1,:));
            steps_h(i+3,i+2)=1+cur_step_param2;
            steps_h(i+3,i+1)=steps_h(i+3,i+1)-cur_step_param;
        end
        steps_h(N+2,:)=steps_h(N+1,:);
        steps_h(N+2,N+1)=1;
        steps_h=steps_h(2:end,2:end);
        steps_h2=steps_h;
    case 'Custom'
        if size(A.stepsize,1)==N && size(A.stepsize,2)==N
            steps_h=[zeros(1,N); A.stepsize];
        else
            error('Wrong use of the Custom option');
        end
        if (Prox==1 || Proj)
            error('Wrong use of the Custom option');
        end
    otherwise %GM
        steps_h(2,1)=h;
        for i=2:N
            steps_h(i+1,:)=steps_h(i,:);
            steps_h(i+1,i)=h;
        end
        steps_h2=steps_h;
        method='GM';
end
%% Passing the parsed input to specialized methods
if (~OnlyConjecture)
    switch solver
        case 'sedumi'
            ops = sdpsettings('verbose',verb_spec,'solver','sedumi','sedumi.eps',tol_spec);
            tolerance=tol_spec;
        case 'mosek'
            ops = sdpsettings('verbose',verb_spec,'solver','mosek','mosek.MSK_DPAR_INTPNT_CO_TOL_PFEAS',tol_spec);
            tolerance=tol_spec;
        case 'sdpt3'
            sqlparameters.gaptol=1e-12; sqlparameters.inftol=1e-12; sqlparameters.steptol=1e-12; %#ok<STRNU>
            ops = sdpsettings('solver','sdpt3','verbose',verb_spec,'debug',1);
            tolerance=tol_spec;
        case 'sdplr'
            ops=sdpsettings('solver','sdplr','sdplr.maxrank',2,'sdplr.feastol',tol_spec,'savesolveroutput',1,'sdplr.dir',2,'sdplr.centol',1e-2);
            tolerance=tol_spec;
        otherwise
            ops=sdpsettings('verbose',verb_spec);
            solver='Yalmip default';
            tolerance='Yalmip default';
    end
    
    if (Proj)
        [val, Sol, Prob]=pep_proj(steps_h,steps_h2,method,criterion,solver,relax,L,mu,R,N,h,ops,tolerance,OneSeq);
    elseif (Prox)
        [val, Sol, Prob]=pep_prox(steps_h,steps_h2,method,criterion,solver,relax,L,mu,R,N,h,ops,tolerance,OneSeq);
    else
        [val, Sol, Prob]=pep_unc(steps_h,method,criterion,solver,relax,L,mu,R,N,h,ops,tolerance);
    end
    Prob.steps_h=steps_h;
end
%% Conjectures
outth=NaN;
if (~Prox && ~Proj)
    switch method
        case 'FGM1'
            sumgamma=sum(steps_h(end,:));
            switch criterion
                case 'Obj'
                    tau=1/(2*sumgamma+1);
                    outth=L*R^2/2*tau;
                case 'Grad'
                    outth=NaN;
                case 'MinGrad'
                    outth=NaN;
                case 'Dist'
                    outth=NaN;
            end
        case 'FGM2'
            sumgamma=sum(steps_h(end,:));
            switch criterion
                case 'Obj'
                    tau=1/(2*sumgamma+1);
                    outth=L*R^2/2*tau;
                case 'Grad'
                    outth=NaN;
                case 'MinGrad'
                    outth=NaN;
                case 'Dist'
                    outth=NaN;
            end
        case 'FGM1alt'
            switch criterion
                case 'Obj'
                    outth=L*R^2*4/( 5*N + N^2 +6)/2;
                case 'Grad'
                    outth=NaN;
                case 'MinGrad'
                    outth=NaN;
                case 'Dist'
                    outth=NaN;
            end            
        case 'FGM2alt'
            switch criterion
                case 'Obj'
                    outth=L*R^2*4/(5*(N+1) + (N+1)^2-2)/2;
                case 'Grad'
                    outth=NaN;
                case 'MinGrad'
                    outth=NaN;
                case 'Dist'
                    outth=NaN;
            end                      
        case 'StrCvxFGM1'
            outth=NaN;
        case  'OGM2'
            sumgamma=sum(steps_h(end,:));
            switch criterion
                case 'Obj'
                    tau=1/(2*sumgamma+1);
                    outth=L*R^2/2*tau;
                case 'Grad'
                    outth=NaN;
                case 'MinGrad'
                    outth=NaN;
                case 'Dist'
                    outth=NaN;
            end
        case  'OGM1'
            sumgamma=sum(steps_h(end,:));
            switch criterion
                case 'Obj'
                    tau=1/(2*sumgamma+1);
                    outth=L*R^2/2*tau;
                case 'Grad'
                    outth=NaN;
                case 'MinGrad'
                    outth=NaN;
                case 'Dist'
                    outth=NaN;
            end
        case 'GM'
            if (mu==0)
                switch criterion
                    case 'Grad'
                        outth=(L*R*max(1/(N*h+1),abs(1-h)^(N)))^2;
                    case 'MinGrad'
                        outth=(L*R*max(1/(N*h+1),abs(1-h)^(N)))^2;
                    case 'Dist'
                        outth=R;
                    otherwise%case 'Obj'
                        outth=L*R^2/2*max(1/(2*N*h+1),(1-h)^(2*N));
                end
            else
                kappa=mu/L;
                switch criterion
                    case 'Grad'
                        outth=(L*R*max(kappa*(1-h*kappa)^N/((kappa-1)*(1-h*kappa)^N+1),abs(1-h)^(N)))^2;
                    case 'MinGrad'
                        outth=(L*R*max(kappa*(1-h*kappa)^N/((kappa-1)*(1-h*kappa)^N+1),abs(1-h)^(N)))^2;
                    case 'Dist'
                        outth=NaN;
                    otherwise %case 'Obj'
                        outth=L*R^2/2*max(kappa*(1-h*kappa)^(2*N)/((kappa-1)*(1-h*kappa)^(2*N)+1),(1-h)^(2*N));
                end
            end
    end
end

if (Proj) %Crit=Obj
    sumgamma=sum(steps_h(end,:));
    tau=1/(2*sumgamma)/2;
    if (mu==0)
        switch method
            case 'FGM1'
                switch criterion
                    case 'Obj'
                        outth=L*R^2*tau;
                    otherwise
                        outth=NaN;
                end
            case 'FGM2'
                switch criterion
                    case 'Obj'
                        outth=L*R^2*tau;
                    otherwise
                        outth=NaN;
                end
                case 'FGM1alt'
                switch criterion
                    case 'Obj'
                        outth=L*R^2*4/( 5*N + N^2 +2)/2;
                    otherwise
                        outth=NaN;
                end
            case 'FGM2alt'
                switch criterion
                    case 'Obj'
                        outth=L*R^2*4./(5*(N+1) + (N+1)^2-6)/2;
                    otherwise
                        outth=NaN;
                end
            case 'GM'
                switch criterion
                    case 'Obj'
                        outth=L*R^2*max(tau,(1-h)^(2*N)/2);
                    case 'Grad'
                        outth=max(1/(A.stepsize*A.N)^2,(1-h)^(2*N));%smooth str cvx opt. cond
                    otherwise
                        outth=NaN;
                end
                
            otherwise
                outth=NaN;
        end
    else
        kappa=mu/L;
        switch method
            case 'GM'
                switch criterion
                    case 'Obj'
                        outth=max(- kappa/2 - kappa/(2*((1 - h*kappa)^(2*N) - 1)),(1-h)^(2*N)/2);
                    case 'Grad'
                        outth=max((-kappa - kappa/(((1 - h*kappa)^(N) - 1)))^2,(1-h)^(2*N));%smooth str cvx opt. cond
                    otherwise
                        outth=NaN;
                end
            otherwise
                outth=NaN;
        end
    end
end
if (Prox)
    if (mu==0)
        switch method
            case 'FGM1'
                switch criterion
                    case 'Obj'
                        sumgamma=sum(steps_h(end,:));
                        tau=1/(2*sumgamma)/2;
                        outth=L*R^2*tau;
                    otherwise
                        outth=NaN;
                end
            case 'FGM1alt'
                switch criterion
                    case 'Obj'
                        outth=L*R^2*4/( 5*N + N^2 +2)/2;
                    otherwise
                        outth=NaN;
                end
            case 'GM'
                sumgamma=sum(steps_h(end,:));
                tau=1/(2*sumgamma)/2;
                switch criterion
                    case 'Obj'
                        outth=L*R^2*max(tau,(1-h)^(2*N)/2);
                    case 'Grad'
                        outth=max(1/(A.stepsize*A.N)^2,(1-h)^(2*N));%smooth str cvx opt. cond
                    otherwise
                        outth=NaN;
                end
            otherwise
                outth=NaN;
        end
    else
        kappa=mu/L;
        switch method
            case 'GM'
                switch criterion
                    case 'Obj'
                        outth=max(- kappa/2 - kappa/(2*((1 - h*kappa)^(2*N) - 1)),(1-h)^(2*N)/2);
                    case 'Grad'
                        outth=max((-kappa - kappa/(((1 - h*kappa)^(N) - 1)))^2,(1-h)^(2*N));%smooth str cvx opt. cond
                    otherwise
                        outth=NaN;
                end
            otherwise
                outth=NaN;
        end
    end
end
val.conj=outth;
if (OnlyConjecture)
    Sol.Status='Only Conjecture Mode';
    Prob.Status='Only Conjecture Mode';
    Prob.mu=mu;Prob.R=R;Prob.L=L;Prob.nbIter=N;Prob.method=method;Prob.criterion=criterion;
end

%% Summary

if verb_spec
    fprintf('\nWorst-case estimation of criterion %s for method %s on an (mu=%g,L=%g)-function after %d iterations:', C.name, A.name, P.mu, P.L, A.N);
    if isnan(val.conj)
        conj_str = 'no conjectured value available.';
    else
        conj_str = sprintf('conjectured value = %g', val.conj);
    end
    fprintf('-> primal-dual interval found = [%g %g] ; %s\n', val.primal, val.dual, conj_str);
end
end

function [val, Sol, Prob]=pep_unc(steps_h,method,criterion,solver,relax,L,mu,R,N,h,ops,tolerance) %#ok<INUSL>
steps_c=[-steps_h/L zeros(N+1,1) ones(N+1,1)];

% RELAXATION SCHEMES:
%   - 1 standard from [DT]
%
%% Matrices Generation:
%
% G= [g0 g1 ... gN x0]^T[g0 g1 ... gN x0]
%
%

%Starting condition
AR=zeros(N+2,N+2);
AR(N+2,N+2)=1;

% Functional class (along with iterations)
% fi >= fj + gj^T(xi-xj) + 1/(2L) ||gi-gj||^2_2 (+str cvx)
% with j>i

AF=zeros(N+2,N+2,(N+1)*N/2);
BF=zeros(N+1,1,(N+1)*N/2);
count=0;
for j=2:N+1
    for i=1:j-1
        count=count+1;
        BF(i,1,count)=-1;
        BF(j,1,count)=1;
        AF(j,j,count)=1/(4*(L-mu)); %True  value/2 because we sum with transpose afterwards
        AF(i,i,count)=1/(4*(L-mu)); %True  value/2 because we sum with transpose afterwards
        AF(i,j,count)=-1/(2*(L-mu));
        
        ci=steps_c(i,:);
        cj=steps_c(j,:);
        ei=zeros(N+2,1);
        ej=ei;
        ei(i)=1;
        ej(j)=1;
        
        AF(:,:,count)=AF(:,:,count)+(L/(L-mu))*1/2*(ej*(ci-cj));
        AF(:,:,count)=AF(:,:,count)+(mu/(L-mu))*1/2*(ei*(cj-ci));
        
        c=(ci-cj);
        AF(:,:,count)=AF(:,:,count)+L*mu/(2*(L-mu))*(c.'*c)/2;
        
        AF(:,:,count)=AF(:,:,count).'+AF(:,:,count);
        
    end
end

% fj >= fi + gi^T(xj-xi) + 1/(2L) ||gj-gi||^2_2 (+str cvx)
% with j>i
AF2=zeros(N+2,N+2,(N+1)*N/2);
BF2=zeros(N+1,1,(N+1)*N/2);
count=0;
for i=1:N
    for j=i+1:N+1
        count=count+1;
        BF2(i,1,count)=1;
        BF2(j,1,count)=-1;
        AF2(j,j,count)=1/(4*(L-mu)); %True  value/2 because we sum with transpose afterwards
        AF2(i,i,count)=1/(4*(L-mu)); %True  value/2 because we sum with transpose afterwards
        AF2(i,j,count)=-1/(2*(L-mu));
        
        ci=steps_c(i,:);
        cj=steps_c(j,:);
        ei=zeros(N+2,1);
        ej=ei;
        ei(i)=1;
        ej(j)=1;
        
        
        AF2(:,:,count)=AF2(:,:,count)+(L/(L-mu))*1/2*(ei*(cj-ci));
        AF2(:,:,count)=AF2(:,:,count)+(mu/(L-mu))*1/2*(ej*(ci-cj));
        
        c=(ci-cj);
        AF2(:,:,count)=AF2(:,:,count)+L*mu/(2*(L-mu))*(c.'*c)/2;
        
        AF2(:,:,count)=AF2(:,:,count).'+AF2(:,:,count);
    end
end

% f* >= fj + gj^T(x*-xj) + 1/(2L) ||g*-gj||^2_2 (+str cvx)

AFopt=zeros(N+2,N+2,N+1);
BFopt=zeros(N+1,1,N+1);
count=0;
for j=1:N+1
    count=count+1;
    BFopt(j,1,count)=1;
    AFopt(j,j,count)=1/(4*(L-mu));
    
    cj=steps_c(j,:);
    ej=zeros(N+2,1);
    ej(j)=1;
    
    AFopt(:,:,count)=AFopt(:,:,count)-(L/(L-mu))*1/2*(ej*cj);
    
    AFopt(:,:,count)=AFopt(:,:,count)+L*mu/(2*(L-mu))*(cj.'*cj)/2;
    AFopt(:,:,count)=AFopt(:,:,count).'+AFopt(:,:,count);
    
end

%fj >= f* + 1/(2L) ||g*-gj||^2_2  (+str cvx)

AFopt2=zeros(N+2,N+2,N+1);
BFopt2=zeros(N+1,1,N+1);
count=0;
for j=1:N+1
    count=count+1;
    BFopt2(j,1,count)=-1;
    AFopt2(j,j,count)=1/(4*(L-mu));
    cj=steps_c(j,:);
    ej=zeros(N+2,1);
    ej(j)=1;
    AFopt2(:,:,count)=AFopt2(:,:,count)-(mu/(L-mu))*1/2*(ej*cj);
    
    
    AFopt2(:,:,count)=AFopt2(:,:,count)+L*mu/(2*(L-mu))*(cj.'*cj)/2;
    
    AFopt2(:,:,count)=AFopt2(:,:,count).'+AFopt2(:,:,count);
    
end

%% Complete primal problem

count=0;
const_count_tot=0;

G=sdpvar(N+2);
F=sdpvar(N+1,1);
cons=(G>=0);
cons=cons+(trace(AR*G)-R^2<=0);


for j=2:N+1
    for i=1:j-1
        count=count+1;
        const_count_tot=const_count_tot+1;
        if (~relax || (j==i+1 && relax) )
            cons=cons+(trace(AF(:,:,count)*G)+BF(:,:,count).'*F<=0);
        end
    end
end
count=0;
for i=1:N
    for j=i+1:N+1
        count=count+1;
        const_count_tot=const_count_tot+1;
        if (~relax)
            cons=cons+(trace(AF2(:,:,count)*G)+BF2(:,:,count).'*F<=0);
        end
    end
end
count=0;
for j=1:N+1
    count=count+1;
    const_count_tot=const_count_tot+1;
    cons=cons+(trace(AFopt(:,:,count)*G)+BFopt(:,:,count).'*F<=0);
end
count=0;
for j=1:N+1
    const_count_tot=const_count_tot+1;
    count=count+1;
    if (~relax)
        cons=cons+(trace(AFopt2(:,:,count)*G)+BFopt2(:,:,count).'*F<=0);
    end
end

switch criterion
    case 'Grad'
        obj=-G(end-1,end-1);
    case 'MinGrad'
        tau_slack=sdpvar(1,1);
        obj=-tau_slack;
        for i=1:N+1
            cons=cons+(tau_slack<=G(i,i));
        end
    case 'Dist'
        c=steps_c(end,:);
        obj=-trace((c.'*c)*G);
    case 'AttemptObj'
        tau_slack=sdpvar(1,1);
        obj=-tau_slack;
        cons=cons+(tau_slack-1/F(end)<=0);
        criterion='AttemptObj';
    otherwise %case 'Obj'
        obj=-F(end);
        criterion='Obj';
end
outth=NaN; %#ok<NASGU>
saveYMdetails=optimize(cons,obj,ops);
outp=-double(obj);
outd=dual(cons(2))*R^2;


err=saveYMdetails.problem;

%% Outputs

val.primal=outp;
val.dual=outd;

Sol.G=double(G);
Sol.S=dual(cons(1));
Sol.lambda=dual(cons(2:end));
Sol.f=double(F);
Sol.err=err;

Prob.criterion=criterion;
Prob.solver=solver;
Prob.solvertolerance=tolerance;
Prob.method=method;
Prob.nbIter=N;
Prob.L=L;
Prob.R=R;
Prob.mu=mu;
Prob.relax=relax;
Prob.H=steps_c;
end
function [val, Sol, Prob]=pep_proj(steps_h,steps_h2,method,criterion,solver,relax,L,mu,R,N,h,ops,tolerance,opt_iterate)

%% General stepsize input:
% P=[g0 g1 ... gN g* s1 s2 ... sN x0]
%
% (explicit steps)    xi=P*steps_c1(i,:).'
% (projections steps) yi=P*steps_c2(i,:).'
%
% with (xi,gi,fi) to be interpolated by a L-smooth convex function
%      and an indicator function can be interpolated such that yi=Proj(xi)
%      si are some subgradients of the indicator function (si=0 if we are
%      in the interior of the domain and si points to the exterior of the
%      set if we are on the boundary (i.e. <si;y-yi><=0 for all feasible y)
%
%
% RELAXATION SCHEMES:
%   - 1 standard from [DT] (for both functions) (that is, we only consider
%   interpolation constraint between consecutive iterates, and only in the 
%   direction f_i>=f_{i+1}+...)

if (opt_iterate)
    steps_c1=[-steps_h/L zeros(N+1,2) -steps_h/L ones(N+1,1)];
    steps_c2=steps_c1(2:end,:);
else
    steps_c1=[-steps_h/L zeros(N+1,2) -steps_h2/L ones(N+1,1)];
    
    steps_c2=zeros(size(steps_c1)-[1 0]);
    steps_c2(:,end)=1;
    for i=1:N
        steps_c2(i,:)=steps_c1(i,:);
        steps_c2(i,i)=-h/L;
        steps_c2(i,N+2+i)=-h/L;
    end
end

%% Optim.

% notations: min f + h; g=grad of f s=subgrad of h
%            g*+s*=0; x*=0; f*=h*=0;
%
% Gram matrix: G=P.'P with P=[g0 g1 ... gN g* s1 s2 ... sN x0]

G=sdpvar(2*N+3);
f=sdpvar(N+1,1);

cons=(G>=0);
cons=cons+(G(2*N+3,2*N+3)<=R^2);
% interp. between iterates
for i=1:N+1
    for j=1:N+1
        if (i~=j)
            if (~relax || (j==i+1 && relax))
                cons=cons+(-(f(i)-f(j)-G(j,:)*(steps_c1(i,:)-steps_c1(j,:)).'-(1/(2*(1-mu/L)))*((G(i,i)+G(j,j)-2*G(i,j))/L+...
                    mu*(steps_c1(i,:)-steps_c1(j,:))*G*(steps_c1(i,:)-steps_c1(j,:)).'-2*mu/L*(G(j,:)-G(i,:))*(steps_c1(j,:)-steps_c1(i,:)).'))<=0);%smooth+str cvx
            end
            if (~relax || (i==j+1 && relax)) && (i~=N+1 && j~=N+1 )
                cons=cons+(G(N+2+i,:)*(steps_c2(j,:)-steps_c2(i,:)).'<=0);
            end
        end
    end
end

% interp. with optimal point
for i=1:N+1
    cons=cons+(-(-f(i)-G(i,:)*(-steps_c1(i,:)).'-(1/(2*(1-mu/L)))*((G(N+2,N+2)+G(i,i)-2*G(N+2,i))/L+mu*(-steps_c1(i,:))*G*(-steps_c1(i,:)).'-2*mu/L*(G(i,:)-G(N+2,:))*(steps_c1(i,:)).'))<=0);%smooth+str cvx
    if (~relax)
        cons=cons+(-(f(i)-G(N+2,:)*(steps_c1(i,:)).'-(1/(2*(1-mu/L)))*((G(i,i)+G(N+2,N+2)-2*G(i,N+2))/L+mu*(steps_c1(i,:))*G*(steps_c1(i,:)).'-2*mu/L*(G(N+2,:)-G(i,:))*(-steps_c1(i,:)).'))<=0);%smooth+str cvx
    end
end
for i=1:N
    cons=cons+(-(G(N+2+i,:)*(steps_c2(i,:)).')<=0); % h_*>=h_i+...
    if (~relax)
        cons=cons+(G(N+2,:)*(-steps_c2(i,:)).'<=0); % h_i>=h_*+...
    end
end


% opt. condition

switch criterion
    otherwise
        obj=-f(N+1);
end


outth=NaN;
saveYMdetails=optimize(cons,obj,ops);
outp=-double(obj);
outd=dual(cons(2))*R^2;
err=saveYMdetails.problem;

%% Conjectures

%% Outputs
val.primal=outp;
val.dual=outd;
val.conj=outth;

Sol.G=double(G);
Sol.S=dual(cons(1));
Sol.lambda=dual(cons(2:end));
Sol.f=double(f);
Sol.err=err;

Prob.criterion=criterion;
Prob.solver=solver;
Prob.solvertolerance=tolerance;
Prob.method=method;
Prob.nbIter=N;
Prob.L=L;
Prob.R=R;
Prob.mu=mu;
Prob.relax=relax;
Prob.H1=steps_c1;
Prob.H2=steps_c2;

end

function [val, Sol, Prob]=pep_prox(steps_h,steps_h2,method,criterion,solver,relax,L,mu,R,N,h,ops,tolerance,opt_iterate)
%% General stepsize input:
% P=[g0 g1 ... gN g* s0 s1 ... sN x0]
%
% (explicit steps) xi=P*steps_c1(i,:).'
% (implicit steps) yi=P*steps_c2(i,:).'
%
% with (xi,gi,fi) to be interpolated by a L-smooth convex function
%      (yi,si,hi) to be interpolated by a convex function
%
% RELAXATION SCHEMES:
%   - 1 standard from [DT] (for both functions) (that is, we only consider
%   interpolation constraint between consecutive iterates, and only in the 
%   direction f_i>=f_{i+1}+...)

if (opt_iterate)
    steps_c1=[-steps_h/L zeros(N+1,3) -steps_h/L ones(N+1,1)];
    
    steps_c2=steps_c1;
else
    steps_c1=[-steps_h/L zeros(N+1,2) -steps_h2/L zeros(N+1,1) ones(N+1,1)];
    
    steps_c2=zeros(size(steps_c1));
    for i=1:N
        steps_c2(i,:)=steps_c1(i,:);
        steps_c2(i,i)=-h/L;
        steps_c2(i,N+2+i)=-h/L;
    end
    steps_c2(N+1,:)=steps_c1(N+1,:);
end
%% Optim.

% notations: min f + h; g=grad of f; s=subgrad of h
%            g*+s*=0; x*=0; f*=h*=0; (s*=-g*)
%
% Gram matrix: G=P.'P with P=[g0 g1 ... gN g* s0 s1 ... sN x0]

G=sdpvar(2*N+4);
f=sdpvar(N+1,1);
h=sdpvar(N+1,1);

cons=(G>=0);
cons=cons+(G(2*N+4,2*N+4)-R^2<=0);


% interp. between iterates
for i=1:N+1
    for j=1:N+1
        if (i~=j)
            if (~relax || (j==i+1 && relax))
                cons=cons+(-(h(i)-h(j)-G(N+2+j,:)*(steps_c2(i,:)-steps_c2(j,:)).')<=0);
                
                cons=cons+(-(f(i)-f(j)-G(j,:)*(steps_c1(i,:)-steps_c1(j,:)).'-(1/(2*(1-mu/L)))*((G(i,i)+G(j,j)-2*G(i,j))/L+...
                    mu*(steps_c1(i,:)-steps_c1(j,:))*G*(steps_c1(i,:)-steps_c1(j,:)).'-2*mu/L*(G(j,:)-G(i,:))*(steps_c1(j,:)-steps_c1(i,:)).'))<=0);%smooth+str cvx
            end
        end
    end
end

% interp. with optimal point
for i=1:N+1
    cons=cons+(-(-h(i)-G(N+2+i,:)*(-steps_c2(i,:)).')<=0); % h_*>=h_i+...
    if (~relax)
        cons=cons+(-(h(i)+G(N+2,:)*(steps_c2(i,:)).')<=0); % h_i>=h_*+...
    end
    cons=cons+(-(-f(i)-G(i,:)*(-steps_c1(i,:)).'-(1/(2*(1-mu/L)))*((G(N+2,N+2)+G(i,i)-2*G(N+2,i))/L+mu*(-steps_c1(i,:))*G*(-steps_c1(i,:)).'-2*mu/L*(G(i,:)-G(N+2,:))*(steps_c1(i,:)).'))<=0);%smooth+str cvx
    
    if (~relax)
        cons=cons+(-(f(i)-G(N+2,:)*(steps_c1(i,:)).'-(1/(2*(1-mu/L)))*((G(i,i)+G(N+2,N+2)-2*G(i,N+2))/L+mu*(steps_c1(i,:))*G*(steps_c1(i,:)).'-2*mu/L*(G(N+2,:)-G(i,:))*(-steps_c1(i,:)).'))<=0);%smooth+str cvx
    end
end

switch criterion
    otherwise
        obj=-f(N+1)-h(N+1);
end
outth=NaN;
saveYMdetails=optimize(cons,obj,ops);
outp=-double(obj);
outd=dual(cons(2))*R^2;
err=saveYMdetails.problem;

%% Conjectures

%% Outputs
val.primal=outp;
val.dual=outd;
val.conj=outth;

Sol.G=double(G);
Sol.S=dual(cons(1));
Sol.lambda=dual(cons(2:end));
Sol.f=double(f);
Sol.h=double(h);
Sol.err=err;

Prob.criterion=criterion;
Prob.solver=solver;
Prob.solvertolerance=tolerance;
Prob.method=method;
Prob.nbIter=N;
Prob.L=L;
Prob.R=R;
Prob.mu=mu;
Prob.relax=relax;
Prob.H1=steps_c1;
Prob.H2=steps_c2;
end




