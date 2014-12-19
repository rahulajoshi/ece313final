function [ H1_tabulate, H0_tabulate, H1_tab_ptr, H0_tab_ptr ] = Format_matrix( H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, min_x, range )

    range_check = min_x;
    flag = 0;
    
    for i = 1:range;
        for j = 1:H1_tab_ptr-1
            if (H1_tabulate(j,1) == range_check)
                flag = 1;       
            end
        end
        if (flag == 0)
                H1_tabulate(H1_tab_ptr,1) = range_check;
                H1_tabulate(H1_tab_ptr,3) = 0;
                H1_tab_ptr = H1_tab_ptr + 1;
        end
        flag = 0;
        range_check = range_check + 1;
    end
    %----------------------------------------%

    range_check = min_x;
    flag = 0;
    
    for i = 1:range;
        for j = 1:H0_tab_ptr-1
            if (H0_tabulate(j,1) == range_check)
                flag = 1;       
            end
        end
        if (flag == 0)
                H0_tabulate(H0_tab_ptr,1) = range_check;
                H0_tabulate(H0_tab_ptr,3) = 0;
                H0_tab_ptr = H0_tab_ptr + 1;
        end
        flag = 0;
        range_check = range_check + 1;
    end
    %-----------------------------------
    
    % Convert percentages to a number between 0 and 1
    for i = 1:H1_tab_ptr-1
       H1_tabulate(i,3) = H1_tabulate(i,3) * 0.01; 
    end
    
    for i = 1:H0_tab_ptr-1
       H0_tabulate(i,3) = H0_tabulate(i,3) * 0.01; 
    end

end

