function experiences_list = create_greedy_experiences_list()

[states_list] = create_states_list();
[actions_list] = create_actions_list();

size_states = size(states_list);

% e = [s,a,r,sp]
size_experiences_list = ...
    [size_states(1) , size_states(2)+1+1+size_states(2)];

experiences_list = zeros(size_experiences_list);

index = 1;
for i = 1:length(states_list)
    s = states_list(i,:);
    rmax = -inf;
    for a = 1:length(actions_list)
        etemp = experience(s,a);
        r = etemp(4);
        if (r > rmax)
            e = etemp;
        end                
    end
    experiences_list(index,:) = e;
    index = index + 1;
end
        
        