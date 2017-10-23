function [sp,bumped2wall,reached2goal] = update_state(s,a)

[x_vec , y_vec,xgrid,ygrid] = create_descritized_vec();
goal_state = [1,1];
% convert to row/column notation: 
x = s(1); y = s(2); 
bumped2wall = 0;
reached2goal = 0;

% Goal is a terminal point
if (s(1) == goal_state(1) && s(2) == goal_state(2))
    bumped2wall = 0;
    reached2goal = 1;
    sp = s;
    return
end

% incorporate any actions and fix our position if we end up outside the grid:
% 
switch a
 case 1, % action = UP 
     sp = [x,y+ygrid];
 case 2, % action = DOWN
     sp = [x,y-ygrid];
 case 3, % action = RIGHT
     sp = [x+xgrid,y];
 case 4  % action = LEFT 
     sp = [x-xgrid,y];
end

% bound position inside the grid:
sp(1) = bound( sp(1) , x_vec );
sp(2) = bound( sp(2) , y_vec );

if (sp(1) == s(1) && sp(2) == s(2))
   bumped2wall = 1;
end

if (sp(1) == goal_state(1) && sp(2) == goal_state(2))
    bumped2wall = 0;
    reached2goal = 1;
end







