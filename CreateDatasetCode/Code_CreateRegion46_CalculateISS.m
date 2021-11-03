
%%%===== Load dataset
load('miniData.mat');

%%%===== Create Region 46 and Calucate ISS value
[miniID, miniLabel, miniData, miniISS] = Function_MakeRegion46(miniTable, ais_code_score);
miniTable{:, end+1} = miniISS;
miniTable.Properties.VariableNames = {'vID' 'vLabel', 'vHead', 'vFace', 'vChest', 'vAbdoman', 'vExtrem', 'vExternal', 'vISS'};
fprintf('[Message: %s] Save Organ46 - square datasets.   \n', datetime(now, 'ConvertFrom', 'datenum'));

