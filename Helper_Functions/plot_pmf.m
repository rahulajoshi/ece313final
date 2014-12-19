function [ ] = plot_pmf( H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, plot_num, patient_num)

    X_H1 = H1_tabulate(1:H1_tab_ptr-1, 1);
    X_H0 = H0_tabulate(1:H0_tab_ptr-1, 1);
    
    Y_H1 = H1_tabulate(1:H1_tab_ptr-1, 3);
    Y_H0 = H0_tabulate(1:H0_tab_ptr-1, 3);
    
    figure(patient_num);
    subplot(7,1,plot_num);
    bar(X_H0(:,1), Y_H0(:,1), 'blue', 'grouped');
    hold on;
    bar(X_H1(:,1), Y_H1(:,1), 'green', 'grouped');
    hold off;
    xlabel('X');
    ylabel('pmf(X)');
    legend('H0','H1', 'Location', 'northeast');

end
