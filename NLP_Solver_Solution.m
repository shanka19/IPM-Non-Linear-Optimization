%% Using fmincon to evaluate problem 
fun = @(x)0.01*x(1)^2 + x(2)^2 - 100;
x0 = [-1,-1];
A = [-10,1];
lb = [2,-50];
ub = [50,50];
b = 10;
Aeq =[];
beq =[];
x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
disp(x)
disp(fun(x))
