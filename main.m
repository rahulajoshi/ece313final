patient_list = {'Patient_Data/1_a41178.mat', 'Patient_Data/2_a42126.mat', 'Patient_Data/3_a40076.mat', 'Patient_Data/4_a40050.mat', 'Patient_Data/5_a41287.mat', 'Patient_Data/6_a41846.mat', 'Patient_Data/7_a41846.mat', 'Patient_Data/8_a42008.mat', 'Patient_Data/9_a41846.mat'};

HT_table_array = cell(9,7);
Error_table_array = cell(9,7);
joint_HT_table = cell(9,1);
joint_error_table_array = cell(3);
mismatches_ML = 0;
mismatches_MAP = 0;
priors = 0;


for p = 1:9
    load(patient_list{p});
        
    floored_data = floor(all_data);
    size_of_data = size(floored_data);
    cutoff = floor((2/3)*size_of_data(1,2));

    training_data = floored_data(:,1:cutoff);
    training_labels = all_labels(:,1:cutoff);

    testing_data = floored_data(:,cutoff+1:size_of_data(1,2));
    testing_labels = all_labels(:,cutoff+1:size_of_data(1,2));
        
    % Calculate Prior Probabilities H1 & H0 
    num_golden_alarms = 0;
    for i = 1:cutoff
        if (training_labels(1,i) == 1)
           num_golden_alarms = num_golden_alarms + 1; 
        end
    end

    Prob_H1 = num_golden_alarms / cutoff;
    Prob_H0 = 1 - Prob_H1;
    
    priors(1,p) = Prob_H1;
    priors(2,p) = Prob_H0;

    % Construct likelihood matrices for each feature

    H1_samples(7,1:cutoff) = 0;
    H0_samples(7,1:cutoff) = 0;
    
    H1_ptr = 1;
    H0_ptr = 1;
    
    for i = 1:cutoff
       if (training_labels(i) == 1)
           H1_samples(:,H1_ptr) = training_data(:,i);
           H1_ptr = H1_ptr + 1;
       else
           H0_samples(:,H0_ptr) = training_data(:,i);
           H0_ptr = H0_ptr + 1;
       end
    end
    
    H1_samples(:,H1_ptr:cutoff) = [];
    H0_samples(:,H0_ptr:cutoff) = [];
    
    % feature_1: Area under HR
    plot_num = 1;
    feature_num = 1;
    
    min_x = min(floored_data(1,:));
    max_x = max(floored_data(1,:));
    range = max_x - min_x + 1;

    H1_tabulate = tabulate(H1_samples(1,1:H1_ptr-1));
    H0_tabulate = tabulate(H0_samples(1,1:H0_ptr-1));

    H1_tabulate_size = size(H1_tabulate);
    H1_tab_ptr = H1_tabulate_size(1,1) + 1;
    H0_tabulate_size = size(H0_tabulate);
    H0_tab_ptr = H0_tabulate_size(1,1) + 1;
    
    % format matrix appropriately
    [H1_tabulate, H0_tabulate, H1_tab_ptr, H0_tab_ptr] = Format_matrix(H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, min_x, range);
    
    % sort the data
    H1_tabulate = sortrows(H1_tabulate,1);
    H0_tabulate = sortrows(H0_tabulate,1);
    
    % plot the data
    plot_pmf(H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, plot_num, p);
    plot_num = plot_num + 1;

    % Calculate ML and MAP rule values
    [ML_Vector] = fill_ML_Vector(range, H1_tab_ptr, H1_tabulate, H0_tabulate, min_x, max_x);
    [MAP_Vector] = fill_MAP_Vector(range, H1_tab_ptr, H1_tabulate, H0_tabulate, Prob_H1, Prob_H0, min_x, max_x); 
    
    % fill HT_table_array
    HT_table(1:range,1:5) = 0;
    
    x = min_x;
    H1_val = 0;
    H0_val = 0;
    HT_table = 0;
    
    for i = 1:range
        HT_table(i,1) = x;  % insert x value
        [H1_val, H0_val] = get_pmf_vals(x, H1_tabulate, H0_tabulate); 
        HT_table(i,2) = H1_val;    % insert pmf @ X given H1
        HT_table(i,3) = H0_val;    % insert pmf @ X given H0
        HT_table(i,4) = ML_Vector(i);
        HT_table(i,5) = MAP_Vector(i);
        x = x + 1;
    end
    
    HT_table_array{p,feature_num} = HT_table;
    
    % task 1.2 stuff
    size_of_alarm_vec = size(testing_labels);
    test_alarms(1:2,1:size_of_alarm_vec(1,2)) = 0;
    size_of_test_data = size(testing_data);
    
    ML_rule_vector = HT_table(:,4);
    MAP_rule_vector = HT_table(:,5);
 
    [ML_table] = create_hash_table(range, HT_table, ML_rule_vector);
    [MAP_table] = create_hash_table(range, HT_table, MAP_rule_vector);
    
    mismatch_ML = 0;
    mismatch_MAP = 0;
    
    for i = 1:size_of_test_data(1,2)
       test_alarms(1,i) = ML_table(int2str(testing_data(feature_num,i)));
       test_alarms(2,i) = MAP_table(int2str(testing_data(feature_num,i)));
       if (test_alarms(1,i) ~= testing_labels(i))
           mismatch_ML = mismatch_ML + 1;
       end
       
       if (test_alarms(2,i) ~= testing_labels(i))
           mismatch_MAP = mismatch_MAP + 1;
       end
    end
    
    mismatches_ML(feature_num,p) = mismatch_ML/size_of_test_data(1,2);
    mismatches_MAP(feature_num,p) = mismatch_MAP/size_of_test_data(1,2);
    
    [Prob_false_alarm_ML, Prob_false_alarm_MAP, Prob_Decision_alarm_ML, Prob_Decision_alarm_MAP] = calculate_prob_false_alarm( size_of_test_data(1,2), test_alarms, testing_labels);
    [Prob_miss_detection_ML, Prob_miss_detection_MAP, Prob_Decision_no_alarm_ML, Prob_Decision_no_alarm_MAP] = calculate_prob_miss_detection( size_of_test_data(1,2), test_alarms, testing_labels);
    Prob_Error_ML = Prob_Decision_alarm_ML + Prob_Decision_no_alarm_ML;
    Prob_Error_MAP = Prob_Decision_alarm_MAP + Prob_Decision_no_alarm_MAP;
    
    Error_table(1:2,1:3) = 0;
    Error_table(1,1) = Prob_false_alarm_ML;
    Error_table(1,2) = Prob_miss_detection_ML;
    Error_table(1,3) = Prob_Error_ML;
    Error_table(2,1) = Prob_false_alarm_MAP;
    Error_table(2,2) = Prob_miss_detection_MAP;
    Error_table(2,3) = Prob_Error_MAP;
    
    Error_table_array{p, feature_num} = Error_table;
        
    feature_num = feature_num + 1;
    
    % feature_2: Mean R-to-R Peak Interval
    min_x = min(floored_data(2,:));
    max_x = max(floored_data(2,:));
    range = max_x - min_x + 1;
    
    H1_tabulate = tabulate(H1_samples(2,1:H1_ptr-1));
    H0_tabulate = tabulate(H0_samples(2,1:H0_ptr-1));
        
    H1_tabulate_size = size(H1_tabulate);
    H1_tab_ptr = H1_tabulate_size(1,1) + 1;
    H0_tabulate_size = size(H0_tabulate);
    H0_tab_ptr = H0_tabulate_size(1,1) + 1;
    
    % format matrix appropriately
    [H1_tabulate, H0_tabulate, H1_tab_ptr, H0_tab_ptr] = Format_matrix(H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, min_x, range);
    
    % sort the data
    H1_tabulate = sortrows(H1_tabulate,1);
    H0_tabulate = sortrows(H0_tabulate,1);
    
    % plot the data
    plot_pmf(H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, plot_num, p);
    plot_num = plot_num + 1;

    % Calculate ML and MAP rule values
    [ML_Vector] = fill_ML_Vector(range, H1_tab_ptr, H1_tabulate, H0_tabulate, min_x, max_x);
    [MAP_Vector] = fill_MAP_Vector(range, H1_tab_ptr, H1_tabulate, H0_tabulate, Prob_H1, Prob_H0, min_x, max_x);

    % fill HT_table_array
    HT_table(1:range,1:5) = 0;
    
    x = min_x;
    H1_val = 0;
    H0_val = 0;
    HT_table = 0;
    
    for i = 1:range
        HT_table(i,1) = x;  % insert x value
        [H1_val, H0_val] = get_pmf_vals(x, H1_tabulate, H0_tabulate); 
        HT_table(i,2) = H1_val;    % insert pmf @ X given H1
        HT_table(i,3) = H0_val;    % insert pmf @ X given H0
        HT_table(i,4) = ML_Vector(i);
        HT_table(i,5) = MAP_Vector(i);
        x = x + 1;
    end
    
    HT_table_array{p,feature_num} = HT_table;
    
    % task 1.2 stuff
    size_of_alarm_vec = size(testing_labels);
    test_alarms(1:2,1:size_of_alarm_vec(1,2)) = 0;
    size_of_test_data = size(testing_data);
    
    ML_rule_vector = HT_table(:,4);
    MAP_rule_vector = HT_table(:,5);
 
    [ML_table] = create_hash_table(range, HT_table, ML_rule_vector);
    [MAP_table] = create_hash_table(range, HT_table, MAP_rule_vector);
    
    mismatch_ML = 0;
    mismatch_MAP = 0;
    
    for i = 1:size_of_test_data(1,2)
       test_alarms(1,i) = ML_table(int2str(testing_data(feature_num,i)));
       test_alarms(2,i) = MAP_table(int2str(testing_data(feature_num,i)));
       if (test_alarms(1,i) ~= testing_labels(i))
           mismatch_ML = mismatch_ML + 1;
       end
       
       if (test_alarms(2,i) ~= testing_labels(i))
           mismatch_MAP = mismatch_MAP + 1;
       end
    end
    
    mismatches_ML(feature_num,p) = mismatch_ML/size_of_test_data(1,2);
    mismatches_MAP(feature_num,p) = mismatch_MAP/size_of_test_data(1,2);
    
    [Prob_false_alarm_ML, Prob_false_alarm_MAP, Prob_Decision_alarm_ML, Prob_Decision_alarm_MAP] = calculate_prob_false_alarm( size_of_test_data(1,2), test_alarms, testing_labels);
    [Prob_miss_detection_ML, Prob_miss_detection_MAP, Prob_Decision_no_alarm_ML, Prob_Decision_no_alarm_MAP] = calculate_prob_miss_detection( size_of_test_data(1,2), test_alarms, testing_labels);
    Prob_Error_ML = Prob_Decision_alarm_ML + Prob_Decision_no_alarm_ML;
    Prob_Error_MAP = Prob_Decision_alarm_MAP + Prob_Decision_no_alarm_MAP;
    
    Error_table(1:2,1:3) = 0;
    Error_table(1,1) = Prob_false_alarm_ML;
    Error_table(1,2) = Prob_miss_detection_ML;
    Error_table(1,3) = Prob_Error_ML;
    Error_table(2,1) = Prob_false_alarm_MAP;
    Error_table(2,2) = Prob_miss_detection_MAP;
    Error_table(2,3) = Prob_Error_MAP;
    
    Error_table_array{p, feature_num} = Error_table;
        
    feature_num = feature_num + 1;
                
    % feature 3: Heart Rate
    min_x = min(floored_data(3,:));
    max_x = max(floored_data(3,:));
    range = max_x - min_x + 1;
    
    H1_tabulate = tabulate(H1_samples(3,1:H1_ptr-1));
    H0_tabulate = tabulate(H0_samples(3,1:H0_ptr-1));
        
    H1_tabulate_size = size(H1_tabulate);
    H1_tab_ptr = H1_tabulate_size(1,1) + 1;
    H0_tabulate_size = size(H0_tabulate);
    H0_tab_ptr = H0_tabulate_size(1,1) + 1;

    % format matrix appropriately
    [H1_tabulate, H0_tabulate, H1_tab_ptr, H0_tab_ptr] = Format_matrix(H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, min_x, range);
    
    % sort the data
    H1_tabulate = sortrows(H1_tabulate,1);
    H0_tabulate = sortrows(H0_tabulate,1);
    
    % plot the data
    plot_pmf(H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, plot_num, p);
    plot_num = plot_num + 1;
    
    % Calculate ML and MAP rule values
    [ML_Vector] = fill_ML_Vector(range, H1_tab_ptr, H1_tabulate, H0_tabulate, min_x, max_x);
    [MAP_Vector] = fill_MAP_Vector(range, H1_tab_ptr, H1_tabulate, H0_tabulate, Prob_H1, Prob_H0, min_x, max_x);
    
    % fill HT_table_array
    HT_table(1:range,1:5) = 0;
    
    x = min_x;
    H1_val = 0;
    H0_val = 0;
    HT_table = 0;
    
    for i = 1:range
        HT_table(i,1) = x;  % insert x value
        [H1_val, H0_val] = get_pmf_vals(x, H1_tabulate, H0_tabulate); 
        HT_table(i,2) = H1_val;    % insert pmf @ X given H1
        HT_table(i,3) = H0_val;    % insert pmf @ X given H0
        HT_table(i,4) = ML_Vector(i);
        HT_table(i,5) = MAP_Vector(i);
        x = x + 1;
    end
    
    HT_table_array{p,feature_num} = HT_table;
    
    % task 1.2 stuff
    size_of_alarm_vec = size(testing_labels);
    test_alarms(1:2,1:size_of_alarm_vec(1,2)) = 0;
    size_of_test_data = size(testing_data);
    
    ML_rule_vector = HT_table(:,4);
    MAP_rule_vector = HT_table(:,5);
 
    [ML_table] = create_hash_table(range, HT_table, ML_rule_vector);
    [MAP_table] = create_hash_table(range, HT_table, MAP_rule_vector);
    
    mismatch_ML = 0;
    mismatch_MAP = 0;
    
    for i = 1:size_of_test_data(1,2)-1
       test_alarms(1,i) = ML_table(int2str(testing_data(feature_num,i)));
       test_alarms(2,i) = MAP_table(int2str(testing_data(feature_num,i)));
       if (test_alarms(1,i) ~= testing_labels(i))
           mismatch_ML = mismatch_ML + 1;
       end
       
       if (test_alarms(2,i) ~= testing_labels(i))
           mismatch_MAP = mismatch_MAP + 1;
       end
    end
    
    mismatches_ML(feature_num,p) = mismatch_ML/size_of_test_data(1,2);
    mismatches_MAP(feature_num,p) = mismatch_MAP/size_of_test_data(1,2);
    
    [Prob_false_alarm_ML, Prob_false_alarm_MAP, Prob_Decision_alarm_ML, Prob_Decision_alarm_MAP] = calculate_prob_false_alarm( size_of_test_data(1,2), test_alarms, testing_labels);
    [Prob_miss_detection_ML, Prob_miss_detection_MAP, Prob_Decision_no_alarm_ML, Prob_Decision_no_alarm_MAP] = calculate_prob_miss_detection( size_of_test_data(1,2), test_alarms, testing_labels);
    Prob_Error_ML = Prob_Decision_alarm_ML + Prob_Decision_no_alarm_ML;
    Prob_Error_MAP = Prob_Decision_alarm_MAP + Prob_Decision_no_alarm_MAP;
    
    Error_table(1:2,1:3) = 0;
    Error_table(1,1) = Prob_false_alarm_ML;
    Error_table(1,2) = Prob_miss_detection_ML;
    Error_table(1,3) = Prob_Error_ML;
    Error_table(2,1) = Prob_false_alarm_MAP;
    Error_table(2,2) = Prob_miss_detection_MAP;
    Error_table(2,3) = Prob_Error_MAP;
    
    Error_table_array{p, feature_num} = Error_table;
    
    feature_num = feature_num + 1;
            
    % feature 4: Blood Pressure Peak-to-Peak
    min_x = min(floored_data(4,:));
    max_x = max(floored_data(4,:));
    range = max_x - min_x + 1;
    
    H1_tabulate = tabulate(H1_samples(4,1:H1_ptr-1));
    H0_tabulate = tabulate(H0_samples(4,1:H0_ptr-1));
        
    H1_tabulate_size = size(H1_tabulate);
    H1_tab_ptr = H1_tabulate_size(1,1) + 1;
    H0_tabulate_size = size(H0_tabulate);
    H0_tab_ptr = H0_tabulate_size(1,1) + 1;
    
    % format matrix appropriately
    [H1_tabulate, H0_tabulate, H1_tab_ptr, H0_tab_ptr] = Format_matrix(H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, min_x, range);
    
    % sort the data
    H1_tabulate = sortrows(H1_tabulate,1);
    H0_tabulate = sortrows(H0_tabulate,1);
    
    % plot the data
    plot_pmf(H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, plot_num, p);
    plot_num = plot_num + 1;
    
    % Calculate ML and MAP rule values
    [ML_Vector] = fill_ML_Vector(range, H1_tab_ptr, H1_tabulate, H0_tabulate, min_x, max_x);
    [MAP_Vector] = fill_MAP_Vector(range, H1_tab_ptr, H1_tabulate, H0_tabulate, Prob_H1, Prob_H0, min_x, max_x);
        
    % fill HT_table_array
    HT_table(1:range,1:5) = 0;
    
    x = min_x;
    H1_val = 0;
    H0_val = 0;
    HT_table = 0;
    
    for i = 1:range
        HT_table(i,1) = x;  % insert x value
        [H1_val, H0_val] = get_pmf_vals(x, H1_tabulate, H0_tabulate); 
        HT_table(i,2) = H1_val;    % insert pmf @ X given H1
        HT_table(i,3) = H0_val;    % insert pmf @ X given H0
        HT_table(i,4) = ML_Vector(i);
        HT_table(i,5) = MAP_Vector(i);
        x = x + 1;
    end
    
    HT_table_array{p,feature_num} = HT_table;
    
    % task 1.2 stuff
    size_of_alarm_vec = size(testing_labels);
    test_alarms(1:2,1:size_of_alarm_vec(1,2)) = 0;
    size_of_test_data = size(testing_data);
    
    ML_rule_vector = HT_table(:,4);
    MAP_rule_vector = HT_table(:,5);
 
    [ML_table] = create_hash_table(range, HT_table, ML_rule_vector);
    [MAP_table] = create_hash_table(range, HT_table, MAP_rule_vector);
    
    mismatch_ML = 0;
    mismatch_MAP = 0;
    
    for i = 1:size_of_test_data(1,2)
       test_alarms(1,i) = ML_table(int2str(testing_data(feature_num,i)));
       test_alarms(2,i) = MAP_table(int2str(testing_data(feature_num,i)));
       if (test_alarms(1,i) ~= testing_labels(i))
           mismatch_ML = mismatch_ML + 1;
       end
       
       if (test_alarms(2,i) ~= testing_labels(i))
           mismatch_MAP = mismatch_MAP + 1;
       end
    end
    
    mismatches_ML(feature_num,p) = mismatch_ML/size_of_test_data(1,2);
    mismatches_MAP(feature_num,p) = mismatch_MAP/size_of_test_data(1,2);
    
    [Prob_false_alarm_ML, Prob_false_alarm_MAP, Prob_Decision_alarm_ML, Prob_Decision_alarm_MAP] = calculate_prob_false_alarm( size_of_test_data(1,2), test_alarms, testing_labels);
    [Prob_miss_detection_ML, Prob_miss_detection_MAP, Prob_Decision_no_alarm_ML, Prob_Decision_no_alarm_MAP] = calculate_prob_miss_detection( size_of_test_data(1,2), test_alarms, testing_labels);
    Prob_Error_ML = Prob_Decision_alarm_ML + Prob_Decision_no_alarm_ML;
    Prob_Error_MAP = Prob_Decision_alarm_MAP + Prob_Decision_no_alarm_MAP;
    
    Error_table(1:2,1:3) = 0;
    Error_table(1,1) = Prob_false_alarm_ML;
    Error_table(1,2) = Prob_miss_detection_ML;
    Error_table(1,3) = Prob_Error_ML;
    Error_table(2,1) = Prob_false_alarm_MAP;
    Error_table(2,2) = Prob_miss_detection_MAP;
    Error_table(2,3) = Prob_Error_MAP;
    
    Error_table_array{p, feature_num} = Error_table;
    
    feature_num = feature_num + 1;
    
    % feature 5: Systolic Blood Pressure
    min_x = min(floored_data(5,:));
    max_x = max(floored_data(5,:));
    range = max_x - min_x + 1;
    
    H1_tabulate = tabulate(H1_samples(5,1:H1_ptr-1));
    H0_tabulate = tabulate(H0_samples(5,1:H0_ptr-1));
        
    H1_tabulate_size = size(H1_tabulate);
    H1_tab_ptr = H1_tabulate_size(1,1) + 1;
    H0_tabulate_size = size(H0_tabulate);
    H0_tab_ptr = H0_tabulate_size(1,1) + 1;
    
    % format matrix appropriately
    [H1_tabulate, H0_tabulate, H1_tab_ptr, H0_tab_ptr] = Format_matrix(H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, min_x, range);
    
    % sort the data
    H1_tabulate = sortrows(H1_tabulate,1);
    H0_tabulate = sortrows(H0_tabulate,1);
    
    % plot the data
    plot_pmf(H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, plot_num, p);
    plot_num = plot_num + 1;
    
   % Calculate ML and MAP rule values
    [ML_Vector] = fill_ML_Vector(range, H1_tab_ptr, H1_tabulate, H0_tabulate, min_x, max_x);
    [MAP_Vector] = fill_MAP_Vector(range, H1_tab_ptr, H1_tabulate, H0_tabulate, Prob_H1, Prob_H0, min_x, max_x);
        
    % fill HT_table_array
    HT_table(1:range,1:5) = 0;
    
    x = min_x;
    H1_val = 0;
    H0_val = 0;
    HT_table = 0;
    
    for i = 1:range
        HT_table(i,1) = x;  % insert x value
        [H1_val, H0_val] = get_pmf_vals(x, H1_tabulate, H0_tabulate); 
        HT_table(i,2) = H1_val;    % insert pmf @ X given H1
        HT_table(i,3) = H0_val;    % insert pmf @ X given H0
        HT_table(i,4) = ML_Vector(i);
        HT_table(i,5) = MAP_Vector(i);
        x = x + 1;
    end
    
    HT_table_array{p,feature_num} = HT_table;
    
    % task 1.2 stuff
    size_of_alarm_vec = size(testing_labels);
    test_alarms(1:2,1:size_of_alarm_vec(1,2)) = 0;
    size_of_test_data = size(testing_data);
    
    ML_rule_vector = HT_table(:,4);
    MAP_rule_vector = HT_table(:,5);
 
    [ML_table] = create_hash_table(range, HT_table, ML_rule_vector);
    [MAP_table] = create_hash_table(range, HT_table, MAP_rule_vector);
    
    mismatch_ML = 0;
    mismatch_MAP = 0;
    
    for i = 1:size_of_test_data(1,2)
       test_alarms(1,i) = ML_table(int2str(testing_data(feature_num,i)));
       test_alarms(2,i) = MAP_table(int2str(testing_data(feature_num,i)));
       if (test_alarms(1,i) ~= testing_labels(i))
           mismatch_ML = mismatch_ML + 1;
       end
       
       if (test_alarms(2,i) ~= testing_labels(i))
           mismatch_MAP = mismatch_MAP + 1;
       end
    end
    
    mismatches_ML(feature_num,p) = mismatch_ML/size_of_test_data(1,2);
    mismatches_MAP(feature_num,p) = mismatch_MAP/size_of_test_data(1,2);
    
    [Prob_false_alarm_ML, Prob_false_alarm_MAP, Prob_Decision_alarm_ML, Prob_Decision_alarm_MAP] = calculate_prob_false_alarm( size_of_test_data(1,2), test_alarms, testing_labels);
    [Prob_miss_detection_ML, Prob_miss_detection_MAP, Prob_Decision_no_alarm_ML, Prob_Decision_no_alarm_MAP] = calculate_prob_miss_detection( size_of_test_data(1,2), test_alarms, testing_labels);
    Prob_Error_ML = Prob_Decision_alarm_ML + Prob_Decision_no_alarm_ML;
    Prob_Error_MAP = Prob_Decision_alarm_MAP + Prob_Decision_no_alarm_MAP;
    
    Error_table(1:2,1:3) = 0;
    Error_table(1,1) = Prob_false_alarm_ML;
    Error_table(1,2) = Prob_miss_detection_ML;
    Error_table(1,3) = Prob_Error_ML;
    Error_table(2,1) = Prob_false_alarm_MAP;
    Error_table(2,2) = Prob_miss_detection_MAP;
    Error_table(2,3) = Prob_Error_MAP;
    
    Error_table_array{p, feature_num} = Error_table;
    
    feature_num = feature_num + 1;
        
    % feature 6: Diastolic Blood Pressure
    min_x = min(floored_data(6,:));
    max_x = max(floored_data(6,:));
    range = max_x - min_x + 1;
    
    H1_tabulate = tabulate(H1_samples(6,1:H1_ptr-1));
    H0_tabulate = tabulate(H0_samples(6,1:H0_ptr-1));
        
    H1_tabulate_size = size(H1_tabulate);
    H1_tab_ptr = H1_tabulate_size(1,1) + 1;
    H0_tabulate_size = size(H0_tabulate);
    H0_tab_ptr = H0_tabulate_size(1,1) + 1;
    
    % format matrix appropriately
    [H1_tabulate, H0_tabulate, H1_tab_ptr, H0_tab_ptr] = Format_matrix(H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, min_x, range);
    
    % sort the data
    H1_tabulate = sortrows(H1_tabulate,1);
    H0_tabulate = sortrows(H0_tabulate,1);
    
    % plot the data
    plot_pmf(H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, plot_num, p);
    plot_num = plot_num + 1;
    
    % Calculate ML and MAP rule values
    [ML_Vector] = fill_ML_Vector(range, H1_tab_ptr, H1_tabulate, H0_tabulate, min_x, max_x);
    [MAP_Vector] = fill_MAP_Vector(range, H1_tab_ptr, H1_tabulate, H0_tabulate, Prob_H1, Prob_H0, min_x, max_x);
        
    % fill HT_table_array
    HT_table(1:range,1:5) = 0;
    
    x = min_x;
    H1_val = 0;
    H0_val = 0;
    HT_table = 0;
    
    for i = 1:range
        HT_table(i,1) = x;  % insert x value
        [H1_val, H0_val] = get_pmf_vals(x, H1_tabulate, H0_tabulate); 
        HT_table(i,2) = H1_val;    % insert pmf @ X given H1
        HT_table(i,3) = H0_val;    % insert pmf @ X given H0
        HT_table(i,4) = ML_Vector(i);
        HT_table(i,5) = MAP_Vector(i);
        x = x + 1;
    end
    
    HT_table_array{p,feature_num} = HT_table;
    
    % task 1.2 stuff
    size_of_alarm_vec = size(testing_labels);
    test_alarms(1:2,1:size_of_alarm_vec(1,2)) = 0;
    size_of_test_data = size(testing_data);
    
    ML_rule_vector = HT_table(:,4);
    MAP_rule_vector = HT_table(:,5);
 
    [ML_table] = create_hash_table(range, HT_table, ML_rule_vector);
    [MAP_table] = create_hash_table(range, HT_table, MAP_rule_vector);
    
    mismatch_ML = 0;
    mismatch_MAP = 0;
    
    for i = 1:size_of_test_data(1,2)
       test_alarms(1,i) = ML_table(int2str(testing_data(feature_num,i)));
       test_alarms(2,i) = MAP_table(int2str(testing_data(feature_num,i)));
       if (test_alarms(1,i) ~= testing_labels(i))
           mismatch_ML = mismatch_ML + 1;
       end
       
       if (test_alarms(2,i) ~= testing_labels(i))
           mismatch_MAP = mismatch_MAP + 1;
       end
    end
    
    mismatches_ML(feature_num,p) = mismatch_ML/size_of_test_data(1,2);
    mismatches_MAP(feature_num,p) = mismatch_MAP/size_of_test_data(1,2);
    
    [Prob_false_alarm_ML, Prob_false_alarm_MAP, Prob_Decision_alarm_ML, Prob_Decision_alarm_MAP] = calculate_prob_false_alarm( size_of_test_data(1,2), test_alarms, testing_labels);
    [Prob_miss_detection_ML, Prob_miss_detection_MAP, Prob_Decision_no_alarm_ML, Prob_Decision_no_alarm_MAP] = calculate_prob_miss_detection( size_of_test_data(1,2), test_alarms, testing_labels);
    Prob_Error_ML = Prob_Decision_alarm_ML + Prob_Decision_no_alarm_ML;
    Prob_Error_MAP = Prob_Decision_alarm_MAP + Prob_Decision_no_alarm_MAP;
    
    Error_table(1:2,1:3) = 0;
    Error_table(1,1) = Prob_false_alarm_ML;
    Error_table(1,2) = Prob_miss_detection_ML;
    Error_table(1,3) = Prob_Error_ML;
    Error_table(2,1) = Prob_false_alarm_MAP;
    Error_table(2,2) = Prob_miss_detection_MAP;
    Error_table(2,3) = Prob_Error_MAP;
    
    Error_table_array{p, feature_num} = Error_table;
    
    feature_num = feature_num + 1;
    
    % feature 7: Pulse Pressure
    min_x = min(floored_data(7,:));
    max_x = max(floored_data(7,:));
    range = max_x - min_x + 1;
    
    H1_tabulate = tabulate(H1_samples(7,1:H1_ptr-1));
    H0_tabulate = tabulate(H0_samples(7,1:H0_ptr-1));
        
    H1_tabulate_size = size(H1_tabulate);
    H1_tab_ptr = H1_tabulate_size(1,1) + 1;
    H0_tabulate_size = size(H0_tabulate);
    H0_tab_ptr = H0_tabulate_size(1,1) + 1;
    
    % format matrix appropriately
    [H1_tabulate, H0_tabulate, H1_tab_ptr, H0_tab_ptr] = Format_matrix(H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, min_x, range);
    
    % sort the data
    H1_tabulate = sortrows(H1_tabulate,1);
    H0_tabulate = sortrows(H0_tabulate,1);
    
    % plot the data
    plot_pmf(H1_tab_ptr, H0_tab_ptr, H1_tabulate, H0_tabulate, plot_num, p);
    plot_num = plot_num + 1;
    
    % Calculate ML and MAP rule values
    [ML_Vector] = fill_ML_Vector(range, H1_tab_ptr, H1_tabulate, H0_tabulate, min_x, max_x);
    [MAP_Vector] = fill_MAP_Vector(range, H1_tab_ptr, H1_tabulate, H0_tabulate, Prob_H1, Prob_H0, min_x, max_x);
      
    % fill HT_table_array
    HT_table(1:range,1:5) = 0;
    
    x = min_x;
    H1_val = 0;
    H0_val = 0;
    HT_table = 0;
    
    for i = 1:range
        HT_table(i,1) = x;  % insert x value
        [H1_val, H0_val] = get_pmf_vals(x, H1_tabulate, H0_tabulate); 
        HT_table(i,2) = H1_val;    % insert pmf @ X given H1
        HT_table(i,3) = H0_val;    % insert pmf @ X given H0
        HT_table(i,4) = ML_Vector(i);
        HT_table(i,5) = MAP_Vector(i);
        x = x + 1;
    end
    
    HT_table_array{p,feature_num} = HT_table;
    
    % task 1.2 stuff
    size_of_alarm_vec = size(testing_labels);
    test_alarms(1:2,1:size_of_alarm_vec(1,2)) = 0;
    size_of_test_data = size(testing_data);
    
    ML_rule_vector = HT_table(:,4);
    MAP_rule_vector = HT_table(:,5);
 
    [ML_table] = create_hash_table(range, HT_table, ML_rule_vector);
    [MAP_table] = create_hash_table(range, HT_table, MAP_rule_vector);
    
    mismatch_ML = 0;
    mismatch_MAP = 0;
    
    for i = 1:size_of_test_data(1,2)
       test_alarms(1,i) = ML_table(int2str(testing_data(feature_num,i)));
       test_alarms(2,i) = MAP_table(int2str(testing_data(feature_num,i)));
       if (test_alarms(1,i) ~= testing_labels(i))
           mismatch_ML = mismatch_ML + 1;
       end
       
       if (test_alarms(2,i) ~= testing_labels(i))
           mismatch_MAP = mismatch_MAP + 1;
       end
    end
        
    mismatches_ML(feature_num,p) = mismatch_ML/size_of_test_data(1,2);
    mismatches_MAP(feature_num,p) = mismatch_MAP/size_of_test_data(1,2);
    
    [Prob_false_alarm_ML, Prob_false_alarm_MAP, Prob_Decision_alarm_ML, Prob_Decision_alarm_MAP] = calculate_prob_false_alarm( size_of_test_data(1,2), test_alarms, testing_labels);
    [Prob_miss_detection_ML, Prob_miss_detection_MAP, Prob_Decision_no_alarm_ML, Prob_Decision_no_alarm_MAP] = calculate_prob_miss_detection( size_of_test_data(1,2), test_alarms, testing_labels);
    Prob_Error_ML = Prob_Decision_alarm_ML + Prob_Decision_no_alarm_ML;
    Prob_Error_MAP = Prob_Decision_alarm_MAP + Prob_Decision_no_alarm_MAP;
    
    Error_table(1:2,1:3) = 0;
    Error_table(1,1) = Prob_false_alarm_ML;
    Error_table(1,2) = Prob_miss_detection_ML;
    Error_table(1,3) = Prob_Error_ML;
    Error_table(2,1) = Prob_false_alarm_MAP;
    Error_table(2,2) = Prob_miss_detection_MAP;
    Error_table(2,3) = Prob_Error_MAP;
    
    Error_table_array{p, feature_num} = Error_table;
    
    feature_num = feature_num + 1;
    
    switch p
        case 1
            display('Dat runtime tho...');
        case 2
            display('Go get a cup of coffee.');
        case 3
            display('Yup, still waiting.');
        case 4
            display('I think Ill go make lunch now.');
        case 5
            display('Just about half way there...');
        case 6
            display('Food is gone now.');
        case 7
            display('I think Im gonna fall asleep...');
        case 8
            display('Zzz...');
        case 9
            display('Hooray! We made it!');
            display('Rendering figures...');
            display('Moving on to task 3...');
            
    end
end

% plot % mismatches
features = [1,2,3,4,5,6,7];
for p = 1:9
    figure(p+9);
    bar(features, mismatches_ML(:,p), 'r', 'stacked');
    hold on;
    bar(features, mismatches_MAP(:,p), 'b', 'stacked');
    hold off;
    legend('ML_Rule','MAP_Rule');
    xlabel('Feature Number');
    ylabel('% of mismatches');
end

% plot sum of % of mismatches for two features we chose for each patient
% helped us pick patients
sum_mismatches_ML   =   [.0349+.0551, .0497+.1072, .0196+.0216, .0269+.0488, .0662+.0662, .2140+.2182, .0935+.1598, .034+.0807, .2140+.2182];
sum_mismatches_MAP  =   [.0195+.0202, .005+.005, .0014+.0014, .004+.0129, .0021+.0063, .0133+.0140, .0202+.0202, .0170+.0170, .0133+.0140];
patients            =   [1,2,3,4,5,6,7,8,9];

figure (20);
bar(patients, sum_mismatches_ML, 'b', 'stacked');
hold on;
bar(patients, sum_mismatches_MAP, 'g', 'stacked');
hold off;
legend('ML_Rule','MAP_Rule');
xlabel('Patient Number');
ylabel('Sum of mismatches - Best Two Features');

% break 3 way tie patient 2 features 1,5,6
% Result: Pick features 1 and 6
load('Patient_Data/2_a42126.mat');

floored_data = floor(all_data);
size_of_data = size(floored_data);
cutoff = floor((2/3)*size_of_data(1,2));

training_data = floored_data(:,1:cutoff);

corr1_5 = corrcoef(training_data(1,:), training_data(5,:));
corr1_6 = corrcoef(training_data(1,:), training_data(6,:));
corr5_6 = corrcoef(training_data(5,:), training_data(6,:));

corr_coeffs = [corr1_5(1,2),corr1_6(1,2),corr5_6(1,2)];

% break tie for patient 7
load('Patient_Data/7_a41846.mat');
floored_data = floor(all_data);
size_of_data = size(floored_data);
cutoff = floor((2/3)*size_of_data(1,2));

training_data = floored_data(:,1:cutoff);

corr3_5 = corrcoef(training_data(3,:), training_data(5,:));
corr3_6 = corrcoef(training_data(3,:), training_data(6,:));
corr3_7 = corrcoef(training_data(3,:), training_data(7,:));
corr5_6 = corrcoef(training_data(5,:), training_data(6,:));
corr5_7 = corrcoef(training_data(5,:), training_data(7,:));
corr6_7 = corrcoef(training_data(6,:), training_data(7,:));

corr_coeffs = [corr3_5(1,2),corr3_6(1,2),corr3_7(1,2),corr5_6(1,2),corr5_7(1,2),corr6_7(1,2)];

% break tie for patient 8
load('Patient_Data/8_a42008.mat');
floored_data = floor(all_data);
size_of_data = size(floored_data);
cutoff = floor((2/3)*size_of_data(1,2));

training_data = floored_data(:,1:cutoff);

corr1_3 = corrcoef(training_data(3,:), training_data(5,:));
corr1_6 = corrcoef(training_data(3,:), training_data(6,:));
corr3_6 = corrcoef(training_data(3,:), training_data(7,:));

corr_coeffs = [corr1_3(1,2),corr1_6(1,2),corr3_6(1,2)];

% task 3.1 stuff
% get liklihood matrices for features entered by user
% & create joint liklihood matrices for each patient
feature_X = input('Please enter a feature number between 1 & 7:\n');
while (feature_X < 1 || feature_X > 7) 
    feature_X = input('Invalid input. Please enter a feature number between 1 & 7:\n');
end

feature_Y = input('Please enter a unique second feature number:\n');
while (feature_Y < 1 || feature_Y > 7 || feature_X == feature_Y)
    if (feature_X == feature_Y)
        feature_Y = input('Invalid input. Cannot have two identical features. Try again.\n');
    else
        feature_Y = input('Invalid input. Please enter a unique second feature number. Try again.\n');
    end
end
display('Computing average ML and MAP errors for two entered features...');

for i = 1:9
   feature_X_liklihood = cell2mat(HT_table_array(i,feature_X));
   feature_Y_liklihood = cell2mat(HT_table_array(i,feature_Y));
   [joint_liklihood_matrix] = create_joint_matrix(feature_X_liklihood, feature_Y_liklihood, priors, i);
   joint_HT_table{i,1} = joint_liklihood_matrix;
end

patients = [1,2,3,4,5,6,7,8,9];
for i = 1:9
    feature_X_liklihood = cell2mat(HT_table_array(patients(i),feature_X));
    feature_Y_liklihood = cell2mat(HT_table_array(patients(i),feature_Y));
    joint_liklihood_matrix = cell2mat(joint_HT_table(patients(i),1));

    x_vals = feature_X_liklihood(:,1);
    x_vals = transpose(x_vals);
    y_vals = feature_Y_liklihood(:,1);
    [mesh_z_param_H1, mesh_z_param_H0] = reformat_joint_matrix(joint_liklihood_matrix, feature_X_liklihood, feature_Y_liklihood);
    
    figure(i+20);
    subplot(2,1,1);
    mesh(y_vals,x_vals,mesh_z_param_H1);
    hold on;
    title('joint conditional pmf values H1');
    subplot(2,1,2);
    mesh(y_vals,x_vals,mesh_z_param_H0);
    hold off;
    title('joint conditional pmf values H0');
end

% task 3.2 stuff
for i = 1:9
    load(patient_list{patients(i)});
    
    floored_data = floor(all_data);
    size_of_data = size(floored_data);
    cutoff = floor((2/3)*size_of_data(1,2));

    training_data = floored_data(:,1:cutoff);
    training_labels = all_labels(:,1:cutoff);

    testing_data = floored_data(:,cutoff+1:size_of_data(1,2));
    testing_labels = all_labels(:,cutoff+1:size_of_data(1,2));
    
    joint_test_alarms = 0;
    joint_ML_table = MapN;
    joint_MAP_table = MapN;
    joint_liklihood_matrix = cell2mat(joint_HT_table(patients(i),1));
    size_of_joint_matrix = size(joint_liklihood_matrix);
    
    % make some hash tables
    for j = 1:size_of_joint_matrix(1,1)
        joint_ML_table(joint_liklihood_matrix(j,1), joint_liklihood_matrix(j,2)) = joint_liklihood_matrix(j,5);
        joint_MAP_table(joint_liklihood_matrix(j,1), joint_liklihood_matrix(j,2)) = joint_liklihood_matrix(j,6);
    end
    
    % create joint test alarm vector for ML & MAP rules
    size_of_test_data = size(testing_data);
    for j = 1:size_of_test_data(1,2)
       joint_test_alarms(1,j) = joint_ML_table(testing_data(feature_X,j),testing_data(feature_Y,j));
       joint_test_alarms(2,j) = joint_MAP_table(testing_data(feature_X,j),testing_data(feature_Y,j));
    end
    
    % calculate probabilites for false alarm, miss detection, and error
    joint_error_table = 0;
    
    [Prob_false_alarm_ML, Prob_false_alarm_MAP, Prob_Decision_alarm_ML, Prob_Decision_alarm_MAP] = calculate_prob_false_alarm( size_of_test_data(1,2), joint_test_alarms, testing_labels);
    [Prob_miss_detection_ML, Prob_miss_detection_MAP, Prob_Decision_no_alarm_ML, Prob_Decision_no_alarm_MAP] = calculate_prob_miss_detection( size_of_test_data(1,2), joint_test_alarms, testing_labels);
    Prob_Error_ML = Prob_Decision_alarm_ML + Prob_Decision_no_alarm_ML;
    Prob_Error_MAP = Prob_Decision_alarm_MAP + Prob_Decision_no_alarm_MAP;
    
    joint_error_table(1,1) = Prob_false_alarm_ML;
    joint_error_table(1,2) = Prob_miss_detection_ML;
    joint_error_table(1,3) = Prob_Error_ML;
    joint_error_table(2,1) = Prob_false_alarm_MAP;
    joint_error_table(2,2) = Prob_miss_detection_MAP;
    joint_error_table(2,3) = Prob_Error_MAP;
    
    joint_error_table_array{i} = joint_error_table;
    
    figure (i+30);
    subplot(3,1,1);
    bar(joint_test_alarms(1,:));
    hold on;
    xlabel('x values');
    title('ML generated alarms');
    subplot(3,1,2);
    bar(joint_test_alarms(2,:));
    hold on;
    xlabel('x values');
    title('MAP generated alarms');
    subplot(3,1,3);
    bar(testing_labels);
    hold off;
    xlabel('x values');
    title('golden alarms');
end

% task 3.3 stuff
patient_1_error_table = cell2mat(joint_error_table_array(1));
patient_2_error_table = cell2mat(joint_error_table_array(2));
patient_3_error_table = cell2mat(joint_error_table_array(3));
patient_4_error_table = cell2mat(joint_error_table_array(4));
patient_5_error_table = cell2mat(joint_error_table_array(5));
patient_6_error_table = cell2mat(joint_error_table_array(6));
patient_7_error_table = cell2mat(joint_error_table_array(7));
patient_8_error_table = cell2mat(joint_error_table_array(8));
patient_9_error_table = cell2mat(joint_error_table_array(9));

Average_ML_Error = (patient_1_error_table(1,3) + patient_2_error_table(1,3) + patient_3_error_table(1,3) + patient_4_error_table(1,3) + patient_5_error_table(1,3) + patient_6_error_table(1,3) + patient_7_error_table(1,3) + patient_8_error_table(1,3) + patient_9_error_table(1,3)) / 9;
Average_MAP_Error = (patient_1_error_table(2,3) + patient_2_error_table(2,3) + patient_3_error_table(2,3) + patient_4_error_table(2,3) + patient_5_error_table(2,3) + patient_6_error_table(2,3) + patient_7_error_table(2,3) + patient_8_error_table(2,3) + patient_9_error_table(2,3)) / 9;

display(Average_ML_Error);
display(Average_MAP_Error);
display('Done!');
