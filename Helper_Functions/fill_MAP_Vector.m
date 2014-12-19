function [ MAP_Vector ] = fill_MAP_Vector( range, H1_tab_ptr, H1_tabulate, H0_tabulate, Prob_H1, Prob_H0, min_x, max_x )

    MAP_Vector(1:range) = 0;
    idx = 1;

    for i = 1:H1_tab_ptr-1
        if ( H1_tabulate(i,1) >= min_x && H1_tabulate(i,1) <= max_x && H0_tabulate(i,1) >= min_x && H1_tabulate(i,1) <= max_x )
            if ((Prob_H1*H1_tabulate(i,3)) >= (Prob_H0*H0_tabulate(i,3)))
                MAP_Vector(idx) = 1;
                idx = idx + 1;
            else
                MAP_Vector(idx) = 0;
                idx = idx + 1;
            end
        end
    end

end

