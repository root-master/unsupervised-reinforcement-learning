function r = get_reward(s,a)
[sp,bumped2wall,reached2goal] = update_state(s,a);
% convert to row/column notation: 
[agentinPuddle,dist2Edge] = CreatePuddle(sp);

r = 0;
if reached2goal,
    r = 0;
elseif agentinPuddle,
    r = min(-400 * dist2Edge,-1);
elseif bumped2wall,
    r = -2;
else
    r = -1;
end