clc,clear
close all
gens = 2:100;
dir = './task_space/task0263/';
figure
a = load(sprintf([dir,'pop_objs_valid%d.txt'],2));
axis_values = [min(a,[],1);max(a,[],1)];
for gen = gens
    a = load(sprintf([dir,'pop_objs_valid%d.txt'],gen));
    plot(a(:,1),a(:,2),'.','MarkerSize',10)
    axis([0 axis_values(2,1) 0 axis_values(2,2)])   
    title(['Optimisation Process in generation ' num2str(gen)] )
    xlabel('Error')
    ylabel('Individual unfairness')
    drawnow
    pause(0.1)
end

