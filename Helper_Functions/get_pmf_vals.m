function [ H1_val, H0_val ] = get_pmf_vals( x, H1_tabulate, H0_tabulate )

H1_val = -1;
H0_val = -1;
i = 1;

while (H1_val == -1)
    if (x == H1_tabulate(i,1))
        H1_val = H1_tabulate(i,3);
    end
    i = i + 1;
end

i = 1;
while (H0_val == -1)
    if (x == H0_tabulate(i,1))
        H0_val = H0_tabulate(i,3);
    end
    i = i + 1;
end

end

