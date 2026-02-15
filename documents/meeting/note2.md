?v=0; a=0 (The car can move slowly but that is not ideal)
?v <= ep; a:= v/T (This will help the car still move)

initial state predicate, follow it to reset the state

Reward problem
Calculate the reward using the state before and after control
(Code is already available in KeyMaeraX, boundaryDist)
Can postpone til next term

z3 smt solver use it on the bound predicate to solve the state init problem and for the reset function

Add warning if ep==0






