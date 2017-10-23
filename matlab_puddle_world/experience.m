function e = experience(s,a)

r = get_reward(s,a);
sp = update_state(s,a);

e = [s,a,r,sp];