close all;
clear;

%% TASK 1

%% 1b
p_to_test = [1.0001, 1.5, 2, 2.5, 3, 3.5, 4];

x_opt = zeros(length(p_to_test), 2);

for ii = 1:length(p_to_test)
    x_to_opt = [0, 0]; %We initialize at zero

    fun = @(x) x(1)^(p_to_test(ii)) + x(2)^(p_to_test(ii)); %Using the inline notation to be used by fmincon

    x_res = fmincon(fun, x_to_opt, [], [], [1, 2], 5); %Optimizing

    x_opt(ii, 1) = x_res(1); %Storing in our result array
    x_opt(ii, 2) = x_res(2);
end

%disp(x_opt);


%% 1c

x1 = linspace(-0.5, 2);
x2 = (5-x1)/2;
figure(1);
plot(x1, x2, "HandleVisibility", "off");
title("Underdetermined problem solved with fmincon");
xlabel("x1");
ylabel("x2");
ylim([1, 3]);
hold on;
for ii = 1:length(p_to_test)
    scatter(x_opt(ii,1), x_opt(ii,2), "DisplayName", string(p_to_test(ii)));
end
%scatter(x_opt(:,1), x_opt(:,2))
hold off;
legend;

%saveas(gcf, 'task1_c', 'png'); %Saving the figure


%% 1d
A = [1, 2];
b = 5;
A_MP = transpose(A) * inv(A * transpose(A));
x_MP = A_MP * b;

hold on;
scatter(x_MP(1), x_MP(2), "filled", "DisplayName", "MP inverse");
hold off;

saveas(gcf, 'task1_d', 'png'); %Saving the figure


%% FUNCTIONS

%Question 1a 
function phi = compute_phi(x, p)

phi = x(1).^(p) + x(2).^(p);

end





