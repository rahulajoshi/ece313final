function [ mesh_z_param_H1, mesh_z_param_H0 ] = reformat_joint_matrix( joint_liklihood_matrix, feature_X_liklihood, feature_Y_liklihood)

mesh_z_param_H1 = 0;
mesh_z_param_H0 = 0;
size_of_feature_X = size(feature_X_liklihood);
size_of_feature_Y = size(feature_Y_liklihood);

idx = 1;
for i = 1:size_of_feature_X(1,1)
   for j = 1:size_of_feature_Y(1,1)
       mesh_z_param_H1(i,j) = joint_liklihood_matrix(idx,3);
       mesh_z_param_H0(i,j) = joint_liklihood_matrix(idx,4);
       idx = idx + 1;
   end
end

end

