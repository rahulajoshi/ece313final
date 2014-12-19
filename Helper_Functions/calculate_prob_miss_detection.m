function [ Prob_miss_detection_ML, Prob_miss_detection_MAP, Prob_Decision_no_alarm_ML, Prob_Decision_no_alarm_MAP ] = calculate_prob_miss_detection ( size_of_test_data, test_alarms, testing_labels )

    Prob_Decision_no_alarm_ML = 0;
    Prob_Decision_no_alarm_MAP = 0;
    Prob_Physician_alarm = 0;
    
    count_ML = 0;
    count_MAP = 0;
    testing_label_count = 0;
    for i = 1:size_of_test_data
        if (test_alarms(1,i) == 0 && testing_labels(i) == 1)
            count_ML = count_ML + 1;
        end
        
        if (test_alarms(2,i) == 0 && testing_labels(i) == 1)
            count_MAP = count_MAP + 1;
        end
        
        if (testing_labels(i) == 1)
            testing_label_count = testing_label_count + 1;
        end
    end
    
    Prob_Decision_no_alarm_ML = count_ML / size_of_test_data;
    Prob_Decision_no_alarm_MAP = count_MAP / size_of_test_data;
    Prob_Physician_alarm = testing_label_count / size_of_test_data;
    
    Prob_miss_detection_ML = Prob_Decision_no_alarm_ML / Prob_Physician_alarm;
    Prob_miss_detection_MAP = Prob_Decision_no_alarm_MAP / Prob_Physician_alarm;
    
end

