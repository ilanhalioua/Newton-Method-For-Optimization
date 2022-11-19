% Ilan Halioua - 100472908
% Claudia Del Campo - 100472925


%PROBLEM 1:

clear
clc
format long

% Function g(x,y) to study:

syms x y  
g = 2*(x^2) - (1.05)*(x^4) + (x^6)/6 + (x)*(y) + (y)^2;

% f(X): Compute gradient of g(x,y):

f1 = diff(g, x);
f2 = diff(g, y);
    
f = [f1;f2];

% Compute the Jacobian of f(X):
    
J = jacobian(f);

% x coordinates s.t: Det(J) = 0: ***USFULL FOR AREAS OF CONVERGENCE:

S = solve(det(J) == 0,x);

% fprintf('x coordinates of X0 from where there won''t be convergence (make det(J) = 0)\n\n');
% fprintf('x1 = %s ≈ 0.5638\n',S(1,1));
% fprintf('x2 = %s ≈ 1.4840\n',S(2,1));
% fprintf('x3 = %s ≈ -0.5638\n',S(3,1));
% fprintf('x4 = %s ≈ -1.4840\n\n',S(4,1));

% Starting point X0 ∈ S:  ***FOR UNDERSTANDING OF THE PROCESS

% x1 = input('Introduce a value for the x coordinate of the initial estimate [-5,5]: ');
% x2 = input('And for the y coordinate [-5,5]: ');
% X0 = [x1;x2];

% [XK, iters] = newton(X0);

% fprintf("\nX%d:\n", iters);
% disp(XK)

% Plot 1 

figure(1);
u = -5:0.05:5;
v = -5:0.05:5;

[X,Y] = meshgrid(u, v);
z = 2*(X.^2) - (1.05).*(X.^4) + (X.^6)./6 + (X).*(Y) + (Y).^2;
surf(X, Y, log(z), 'edgecolor', 'none');

title('Plot of the function');
subtitle('Log scale applied to z axis: Clearer extrema.');

% Plot 2 

tic

figure(2);

step_size = 0.25;

for i = -6:step_size:6
    for j = -6:step_size:6
        X0 = [i;j];
        [XK, iters] = newton(X0);
        if iters < 4
            L(1) = plot(i, j, 'g.'); % Green
        elseif (iters >= 4) && (iters < 7)
            L(2) = plot(i, j, 'y.'); % Yellow
        elseif (iters >= 7) && (iters < 9)
            L(3) = plot(i, j, 'b.'); % Blue
        elseif (iters >= 9) && (iters < 11)
            L(4) = plot(i, j, 'r.'); % Red
        else
            L(5) = plot(i, j, 'k.'); % Black
        end
        hold on
    end
end
hold off
grid on

toc

title({
    'Areas of convergence according to number of iterations'  
    'to arrive to desired accuracy'
    });
txt = ['Equispaced X0 (= [x init., y init.]) with step size = ' num2str(step_size)];
subtitle(txt);
xlabel('x initial');
ylabel('y initial');
legend(L, {'0 <= iters < 4', '4 <= iters < 7', '7 <= iters < 9', '9 <= iters < 11'})

% Plot 3

figure(3);

n = 1;
for i = -5:0.005:5
    Xv(n) = i;
    n = n+1;
end

i = 1;
j = i;

for n = 1:length(Xv)
    C(n) = cond(JAct(Xv(i)));
    i = i + 1;
    j = j + 1;
end

plot(Xv,log(C), 'k')

title('Iterations from each point using Cond(J(Xk))');
subtitle('Only x coordinates affect Cond(J(Xk))');
xlabel('x');
ylabel('Cond(J(Xk))');

% Questions

fprintf("\nFROM OUR OBSERVATIONS:\n");
 
fprintf("\nWe know that the process (Ax = b) is well conditioned when cond(A) is more or less 1.\n"); 
fprintf("In our case, the matrix A is the Jacobian matrix used to iterate from each point.\n");
fprintf("\nAnalyzing figure 3, we can see that in the interval (-5, -1.6) ∪ (+1.6, +5), the process tends to get ill-conditioned as Cond(J(Xk)) tends to ∞.\n")
fprintf("So that the system Ax = b gets very sensitive to perturbation as x tends to ∞.\n");
fprintf("Also, in the restriction -5<=x,y<=5, although one could say that Cond(J(Xk)) takes values close to 1, the rapid change of convergence\n");
fprintf("within the coordinates in the restriction, led us to consider the process ill-conditioned as a whole.\n");
fprintf("And in particular, in the points that make the determinant of the Jacobian equal to 0.\n");

fprintf("\nQUESTIONS:\n");

% [1]
fprintf("\n1. What happens if the point's orbit does, at some step, leave the square S?\n"); 
fprintf("\nFrom what we can see on the second plot, and according to our observations, if the point's orbit leaves the square S, at some step,\n");
fprintf("there won't be convergence to the minimum, so, it will diverge.\n");
fprintf("The second plot goes beyond the restriction square S in order to answer this question.\n");

% [2]
fprintf("\n2. Criteria used to stop:\n"); 
fprintf("\n- Firstly, that the distance with the correct minimum was less than 10^(-6).\n");
fprintf("- Secondly, that the number of iterations did not exceed 10.\n");
fprintf("- Lastly, that the difference between one iteration and the following one was less than (10^(-6))/2.\n");

fprintf("\nThe coordinates that make det(J(Xk)) == 0, were also considered for the areas of convergence.");

% FUNCTIONS:

function [Xk, n] = newton(X0)
    
    % ---------------------------------------------------------------
    % let g(x,y) = 2*(x^2) - 1.05*(x^4) + (x^6)/6 + (x)*(y) + (y)^2
    
    % Real Mins of g(x,y): (3.4951, -1.7476), (-3.4951, 1.7476), (2.1411, -1.0705), (-2.1411, 1.0705), (0, 0)
    
    % Approx. Min of g(x,y) by Newton-Raphson Method:
    
    % gradient(g) -> f(x)
    
    % Gaussian elimination -> Ax = b,   where: A is Jacobian of f(Xk), x is
    % δXk, b is -f(Xk).
    
    % When δXk (= X(k+1) - Xk) is found -> X(k+1) = δXk + Xk

    % *** https://www.youtube.com/watch?v=zPDp_ewoyhM *** 
    % (Aula Global\Numerical Methods)
    % ---------------------------------------------------------------

    Xk = X0;
    distance = checkAccuracy(Xk);
    difference = 1; % 1 >= 10^(-6)
    n = 0;
    nmax = 10;

    while (distance >= 10^(-6)) && (n < nmax) && (difference >= (10^(-6))/2)
        if (abs(Xk(1))==abs(0.5638)) || (abs(Xk(1))==abs(1.4840)) || (abs(Xk(1))==abs(0.5638)) || (abs(Xk(1))==abs(0.5638))
            n = 999;
            break
        end
        temp = Xk;
        DeltaXk = solveGauss(JAct(Xk(1)),-fAct(Xk));
        Xk = DeltaXk + Xk;
        distance = checkAccuracy(Xk);
        difference = norm(Xk - temp);
        n = n+1;
    end
end


function y = fAct(Xk)
    y = [Xk(1).^5 - (21.*Xk(1).^3)./5 + 4.*Xk(1) + Xk(2); Xk(1) + 2.*Xk(2)];
end


function y = JAct(x)
    y = [5*x^4 - (63*x^2)/5 + 4 1; 1 2];
end

function x = solveGauss(A,b)
    r = length(A);
    for j = 1:(r-1)
        for i = r:-1:j+1
            m = A(i,j)/A(j,j);
            A(i,:) = A(i,:) - m*A(j,:);
            b(i) = b(i) - m*b(j);
        end
    end 
    x = zeros(r,1);
    x(r) = b(r)/A(r,r);               
    for i = r-1:-1:1                    
        sum = 0;
        for j = r:-1:i+1                
            sum = sum + A(i,j)*x(j);    
        end 
        x(i) = (b(i)- sum)/A(i,i);
    end
end

function d = checkAccuracy(Xk)
    realMins_g = [sqrt((2/5)*(21+sqrt(91))) -sqrt((2/5)*(21+sqrt(91))) sqrt((2/5)*(21-sqrt(91))) -sqrt((2/5)*(21-sqrt(91))) 0
                 -sqrt((1/10)*(21+sqrt(91))) sqrt((1/10)*(21+sqrt(91))) -sqrt((1/10)*(21-sqrt(91))) sqrt((1/10)*(21-sqrt(91))) 0]; % 5x2
    for i = 1:5
        distv(i) = norm(realMins_g(:,i) - Xk);
    end
    d = min(distv);
end
