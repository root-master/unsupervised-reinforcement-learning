function [states_list] = create_states_list()

[x_vec , y_vec, ~, ~] = create_descritized_vec();

% the number of states -- This is the gross mesh states ; the 1st tiling 
nStates = length(x_vec) * length(y_vec); 
states_list = zeros(nStates,2);

index = 1;
for i=1:length(x_vec),
    for j =1:length(y_vec) 
        states_list(index,:) = [ x_vec(i), y_vec(j)];
        index = index + 1;
    end
end

