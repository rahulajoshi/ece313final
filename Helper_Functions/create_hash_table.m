function [ hash_table ] = create_hash_table( range, HT_table, rule_vector )

hash_table = containers.Map;

for i = 1:range
    hash_table(int2str(HT_table(i,1))) = rule_vector(i);
end

end

