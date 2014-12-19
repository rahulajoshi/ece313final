function [ ML_Vector ] = fill_ML_Vector( range, H1_tab_ptr, H1_tabulate, H0_tabulate, min_x, max_x )

    ML_Vector(1:range) = 0;
    idx = 1;

    for i = 1:H1_tab_ptr-1
        if ( H1_tabulate(i,1) >= min_x && H1_tabulate(i,1) <= max_x && H0_tabulate(i,1) >= min_x && H1_tabulate(i,1) <= max_x )  
            if (H1_tabulate(i,3) >= H0_tabulate(i,3))
                ML_Vector(idx) = 1;
                idx = idx + 1;
            else
                ML_Vector(idx) = 0;
                idx = idx + 1;
            end
        end
    end
    
end

