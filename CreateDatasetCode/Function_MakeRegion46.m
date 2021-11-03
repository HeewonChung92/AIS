function [Data_ID, Data_Label, Data_Input, Data_ISS] = Function_MakeRegion46(Input_datasets, Input_ais)
%%% 1: id
%%% 2: death
%%% 3: ais_head or neck
%%% 4: ais_face
%%% 5: ais_chest
%%% 6: ais_abdominal or peivic contents
%%% 7: ais_extremities or peivic girdle
%%% 8: ais_external

%%% 1: number
%%% 2: AIS code
%%% 3: score
%%% 4: organ 46

%%%=== Data_ID = {N, 2};
%%%=== Data_Input = {N, size_class};
%%%=== Data_Label = {N, 1};

size_class = 46;
size_data = size(Input_datasets, 1);

Data_ID = Input_datasets.vID;
Data_Label = Input_datasets.vLabel;
Data_Input = zeros(size_data, size_class);
Data_ISS = zeros(size_data, 1);

dataset_AIS = table2cell(Input_datasets(:, 3:8));
for idx_sub = 1:size_data
    input_value = zeros(1, size_class);
    input_iss = zeros(1, 6);
    
    %%%===== Create region 46 datasets
    for idx_diagnosis = 1:6
        getDiagnosis = cell2mat(dataset_AIS(idx_sub, idx_diagnosis));
        getCount = getDiagnosis(1);  %%% number of code
        if getCount > 0
            arr_score = [];
            getCode = getDiagnosis(2:end);    %%% get code data
            for idx_code = 1:getCount
                now_code = getCode(idx_code);
                find_code = find(Input_ais(:, 2) == now_code);
                idx_organ = Input_ais(find_code, 4);
                organ_score = Input_ais(find_code, 3);
                if organ_score == 9
                    organ_score = 0; %%% score 9 is same to 0
                end
                input_score = organ_score.^2;   %%% Use Score Square value
                
                input_value(1, idx_organ) = input_value(1, idx_organ) + input_score;
                arr_score = [arr_score; input_score];
                clear now_code find_code idx_organ organ_score input_score
            end
            input_iss(idx_diagnosis) = max(arr_score);
            clear idx_code getCode score_array inputScore arr_score
        end
        clear getDiagnosis getCount
    end
    
    %%%===== Calculate ISS score (use Top3 score)
    input_iss = sort(input_iss, 'descend');
    vISS = sum(input_iss(1:3));
    Data_ISS(idx_sub) = vISS;
    
    Data_Input(idx_sub, :) = input_value;
    clear idx_diagnosis input_value idx_cc input_iss vISS
end
clear idx_sub size_class size_data