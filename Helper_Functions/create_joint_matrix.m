function [ joint_liklihood_matrix ] = create_joint_matrix( feature_X_liklihood, feature_Y_liklihood, priors, patient_num )

    joint_liklihood_matrix = 0;
    size_of_feature_X = size(feature_X_liklihood);
    size_of_feature_Y = size(feature_Y_liklihood);
    
    idx = 1;
    for i = 1:size_of_feature_X(1,1)
       for j = 1:size_of_feature_Y(1,1)
          joint_liklihood_matrix(idx,1) = feature_X_liklihood(i,1);
          joint_liklihood_matrix(idx,2) = feature_Y_liklihood(j,1); 
          joint_liklihood_matrix(idx,3) = feature_X_liklihood(i,2) * feature_Y_liklihood(j,2);
          joint_liklihood_matrix(idx,4) = feature_X_liklihood(i,3) * feature_Y_liklihood(j,3);
          
          if (joint_liklihood_matrix(idx,3) >= joint_liklihood_matrix(idx,4))
             joint_liklihood_matrix(idx,5) = 1;
          else
             joint_liklihood_matrix(idx,5) = 0;
          end
          
          if (joint_liklihood_matrix(idx,3)*priors(1,patient_num) >= joint_liklihood_matrix(idx,4)*priors(2,patient_num))
             joint_liklihood_matrix(idx,6) = 1;
          else
             joint_liklihood_matrix(idx,6) = 0;
          end
          
          idx = idx + 1;
       end
    end

end

